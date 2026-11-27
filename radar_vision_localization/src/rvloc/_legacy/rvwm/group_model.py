"""Single-frame radar/vision fusion for continuous joint member localization.

The learned interface contains ray origins/directions, an independent observer
position and two spectra with physical axes and masks.  Neither native IQ nor
appearance, velocities, depth grids or target labels enter ``conditional``.
RA/RE share range-row tokens; they are not treated as independent likelihoods.
Each member stays on its calibrated visual ray; the model learns joint depths.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

GROUP_SCHEMA = 'observer_referenced_ra_re_continuous_v1'
FORMATIONS = ('line', 'v', 'ring', 'grid', 'staggered_3d')


def _mlp(a, h, b):
    return nn.Sequential(nn.Linear(a, h), nn.SiLU(), nn.Linear(h, b))


def set_position_loss(predicted, target):
    """Training-only unordered reconstruction without persistent identities."""
    if not len(predicted) or not len(target):
        return predicted.sum() * 0.
    distance = torch.cdist(predicted, target).square()
    a, b = linear_sum_assignment(distance.detach().cpu().numpy())
    return distance[torch.as_tensor(a, device=distance.device),
                    torch.as_tensor(b, device=distance.device)].mean()


def match_ray_labels(features, positions, config=None):
    """Supervision/evaluation-only partial assignment, never an inference input.

    The configurable angle gate is an engineering rejection threshold, not a
    calibrated confidence interval. Infeasible angle/support edges are removed
    *before* assignment. Dummy columns permit unmatched detections; their cost
    makes maximum feasible cardinality precede minimum total angular error.
    Only the five geometric feature fields below are required by this helper.
    """
    config = {} if config is None else config
    gate = float(config.get('group_supervision_max_angle_deg', 3.))
    if not np.isfinite(gate) or not 0. < gate < 90.:
        raise ValueError('Supervision angle gate must be finite and in (0,90) degrees')
    origins = np.asarray(features['ray_origins'], float)
    directions = np.asarray(features['ray_directions'], float)
    valid = np.asarray(features['ray_valid'], bool)
    bounds = np.asarray(features['ray_depth_bounds'], float)
    scale = float(features['range_scale_m'])
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    true = np.asarray(positions, float)
    n = len(origins)
    if (origins.shape != (n, 3) or directions.shape != (n, 3) or valid.shape != (n,)
            or bounds.shape != (n, 2) or not np.isfinite(scale) or scale <= 0
            or not np.isfinite(origins).all() or not np.isfinite(directions).all()
            or not np.isfinite(bounds).all()):
        raise ValueError('Finite calibrated ray geometry is required for label assignment')
    if true.ndim != 2 or true.shape[1] != 3 or not np.isfinite(true).all():
        raise ValueError('Separate finite 3D supervision [N,3] required')
    if (not np.allclose(np.linalg.norm(directions[valid], axis=-1), 1., atol=1e-6, rtol=1e-6)
            or np.any(bounds[valid, 0] < 0) or np.any(bounds[valid, 1] <= bounds[valid, 0])):
        raise ValueError('Unit valid rays with nonempty positive depth support required')
    active = np.flatnonzero(valid)
    # This tolerance covers coordinate roundoff only, not measurement noise.
    tolerance = max(1e-6, scale * 1e-7)
    label_domain = (true[:, 0] > 0.) & (np.linalg.norm(true, axis=-1) <= scale + tolerance)
    delta = true[None] - origins[active, None]
    distance = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(distance[..., None], 1e-12)
    cosine = np.clip(np.sum(unit * directions[active, None], axis=-1), -1., 1.)
    angular = np.degrees(np.arccos(cosine))
    depth = np.sum(delta * directions[active, None], axis=-1)
    angle_ok = (distance > 1e-12) & (angular <= gate + 1e-10)
    depth_ok = ((depth >= bounds[active, 0, None] - tolerance)
                & (depth <= bounds[active, 1, None] + tolerance))
    support_ok = label_domain[None] & depth_ok
    feasible = angle_ok & support_ok
    row = np.empty(0, np.int64)
    col = np.empty(0, np.int64)
    if len(active) and len(true):
        # A lost match costs more than every feasible angular cost combined.
        unmatched_cost = float(len(active) + 1)
        cost = np.full((len(active), len(true) + len(active)), unmatched_cost)
        cost[:, :len(true)] = np.where(feasible, angular / gate, unmatched_cost * (len(active) + 2))
        assigned_rows, assigned_cols = linear_sum_assignment(cost)
        keep = assigned_cols < len(true)
        row, col = assigned_rows[keep], assigned_cols[keep]
        if len(row):
            keep = feasible[row, col]
            row, col = row[keep], col[keep]
    rows = active[row]
    diagnostics = dict(
        supervision_usable=bool(len(rows)), matched_training_count=len(rows),
        training_detected_count=n, valid_training_rays=len(active), label_member_count=len(true),
        unmatched_detection_count=len(active) - len(rows), invalid_detection_count=n - len(active),
        unmatched_label_count=len(true) - len(rows), angle_gate_deg=gate,
        angle_gate_status='configured engineering rejection threshold; not calibrated',
        rejected_angle_pairs=int((~angle_ok).sum()), rejected_support_pairs=int((~support_ok).sum()),
        depth_outside_support_pair_count=int((~depth_ok).sum()),
        outside_support_label_count=int((~label_domain).sum()),
        matched_angle_errors_deg=angular[row, col].tolist(),
        rejection_reason=None if len(rows) else 'no_matches_within_angle_and_physical_support')
    return dict(rows=rows.astype(np.int64), cols=col.astype(np.int64),
                depths_m=depth[row, col], diagnostics=diagnostics)


def _masked_mean(values, valid):
    if not len(values):
        return values.new_zeros(values.shape[-1])
    return (values * valid[:, None]).sum(0) / valid.sum().clamp_min(1)


def _masked_max(values, valid):
    if not len(values) or not bool(valid.any()):
        return values.new_zeros(values.shape[-1])
    return values.masked_fill(~valid[:, None], -torch.inf).max(0).values


class _SpectrumEncoder(nn.Module):
    """A small CNN retaining every observed range row, with angle pooling.

    Pooling angular *observations* is a feature compression, not a ray-depth
    search grid.  The input includes physical range, sine/cosine of the actual
    angle, and the valid mask.  Missing pixels never contribute pooled power.
    """
    def __init__(self, hidden, angle_tokens):
        super().__init__()
        self.angle_tokens = int(angle_tokens)
        width = max(hidden // 2, 8)
        self.cnn = nn.Sequential(nn.Conv2d(5, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, hidden, 3, padding=1), nn.SiLU())

    def forward(self, power, mask, ranges, angles, scale, power_scale):
        nr, na = power.shape
        log_power = torch.log1p(power.masked_fill(~mask, 0.) / power_scale)
        radial = (ranges / scale)[:, None].expand(nr, na)
        sine = angles.sin()[None].expand(nr, na)
        cosine = angles.cos()[None].expand(nr, na)
        inputs = torch.stack((log_power, radial, sine, cosine, mask.to(power.dtype)))
        encoded = self.cnn(inputs[None])[0] * mask[None]
        row_valid = mask.any(1)
        row_features = encoded.sum(2).T / mask.sum(1).clamp_min(1)[:, None]
        slots = min(self.angle_tokens, na)
        pooled = F.adaptive_avg_pool2d(encoded[None], (nr, slots))[0]
        fraction = F.adaptive_avg_pool2d(mask[None, None].to(power.dtype), (nr, slots))[0, 0]
        pooled = pooled / fraction.clamp_min(1e-8)[None]
        tokens = pooled.permute(1, 2, 0).reshape(nr * slots, -1)
        return tokens, (fraction > 0).reshape(-1), row_features, row_valid


class GroupSceneModel(nn.Module):
    """Single-frame conditional joint position generator.

    A shared latent couples the continuous depth vector.  Object self-attention
    makes the deterministic decoder permutation-equivariant. Alternative layouts
    represent uncalibrated spatial ambiguity in the current frame.
    """
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        from .single_frame import require_single_frame
        self.schema = self.config.get('architecture_schema', GROUP_SCHEMA)
        require_single_frame(config)
        self.hidden_dim = int(config.get('group_hidden_dim', 64))
        self.latent_dim = int(config.get('group_latent_dim', 8))
        self.max_members = int(config.get('group_max_members', 24))
        h, z = self.hidden_dim, self.latent_dim
        heads = int(config.get('group_attention_heads', 4))
        layers = int(config.get('group_interaction_layers', 2))
        angle_tokens = int(config.get('group_spectrum_angle_tokens', 8))
        if min(h, z, self.max_members, heads, layers, angle_tokens) < 1 or h % heads:
            raise ValueError('Invalid continuous RA/RE model dimensions')
        self.ra_encoder = _SpectrumEncoder(h, angle_tokens)
        self.re_encoder = _SpectrumEncoder(h, angle_tokens)
        self.range_pair_encoder = _mlp(2 * h + 3, h, h)
        self.ray_encoder = _mlp(11, h, h)
        self.observer_encoder = _mlp(3, h, h)
        self.cross_attention = nn.MultiheadAttention(h, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(h)
        self.context_encoder = _mlp(3 * h, h, h)
        layer = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.latent_prior = _mlp(3 * h, h, 2 * z)
        self.point_encoder = _mlp(3, h, h)
        self.latent_posterior = _mlp(3 * h, h, 2 * z)
        self.joint_head = _mlp(2 * h + z, h, 2)
        self.formation_head = _mlp(z, h, len(FORMATIONS))
        self.trained = False
        self.last_proposal_diagnostics = None

    def _tensor(self, value):
        parameter = next(self.parameters())
        return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)

    def _check_features(self, features):
        from .group_observation import validate_features
        validate_features(features)
        if len(features['ray_origins']) > self.max_members:
            raise ValueError('Detected ray count exceeds supported group_max_members')
        forbidden = {'radar', 'visual_tokens', 'geometry', 'depth_candidates', 'depth_valid',
                     'ray_evidence', 'candidate_radial_velocity', 'center_candidates',
                     'iq', 'appearance', 'targets', 'target_positions', 'true_count', 'future', 'future_frames'}
        if forbidden.intersection(features):
            raise ValueError('Only rays, observer position and calibrated RA/RE observations are model inputs')

    def conditional(self, features):
        from .single_frame import validate_single_frame_features
        validate_single_frame_features(features)
        return self._single_condition(features)

    def _single_condition(self, features):
        """Shared per-frame encoder; the baseline numerical path is unchanged."""
        self._check_features(features)
        origins = self._tensor(features['ray_origins'])
        directions = self._tensor(features['ray_directions'])
        bounds = self._tensor(features['ray_depth_bounds'])
        observer = self._tensor(features['observer_position'])
        scale = float(features['range_scale_m'])
        valid = torch.as_tensor(features['ray_valid'], dtype=torch.bool, device=origins.device)
        ra = self._tensor(features['ra_power'])
        re = self._tensor(features['re_power'])
        ra_valid = torch.as_tensor(features['ra_valid'], dtype=torch.bool, device=origins.device)
        re_valid = torch.as_tensor(features['re_valid'], dtype=torch.bool, device=origins.device)
        ranges = self._tensor(features['range_m'])
        # One common linear-power scale. Independent per-map maxima would erase
        # relative amplitudes and create a misleading correspondence cue.
        count = (ra_valid.sum() + re_valid.sum()).clamp_min(1)
        power_scale = ((ra.masked_fill(~ra_valid, 0.).sum() +
                        re.masked_fill(~re_valid, 0.).sum()) / count).clamp_min(1e-20)
        ra_tokens, ra_mask, ra_rows, ra_row_mask = self.ra_encoder(
            ra, ra_valid, ranges, self._tensor(features['azimuth_rad']), scale, power_scale)
        re_tokens, re_mask, re_rows, re_row_mask = self.re_encoder(
            re, re_valid, ranges, self._tensor(features['elevation_rad']), scale, power_scale)
        shared_range = torch.stack((ranges / scale, ra_row_mask.to(ra.dtype), re_row_mask.to(ra.dtype)), -1)
        paired = self.range_pair_encoder(torch.cat((ra_rows, re_rows, shared_range), -1))
        paired_valid = ra_row_mask & re_row_mask
        radar_tokens = torch.cat((ra_tokens, re_tokens, paired), 0)
        radar_valid = torch.cat((ra_mask, re_mask, paired_valid), 0)
        reference = self.observer_encoder(observer / scale)
        # The reference token keeps attention well-defined even if every
        # spectral cell is masked; publication still requires data validity.
        memory = torch.cat((radar_tokens, reference[None]), 0)
        memory_valid = torch.cat((radar_valid, valid.new_ones(1)), 0)
        ray_values = torch.cat((origins / scale, directions,
                                (origins - observer[None]) / scale, bounds / scale), -1)
        tokens = self.ray_encoder(ray_values)
        if len(tokens):
            evidence, _ = self.cross_attention(tokens[None], memory[None], memory[None],
                                               key_padding_mask=~memory_valid[None], need_weights=False)
            tokens = self.cross_norm(tokens + evidence[0])
            # Invalid/padded detections neither influence other members nor
            # participate in the conditional latent. Keep their output slots.
            if bool(valid.any()):
                tokens = self.interaction(tokens[None], src_key_padding_mask=~valid[None])[0]
            tokens = tokens * valid[:, None]
        radar_summary = _masked_mean(radar_tokens, radar_valid)
        context = self.context_encoder(torch.cat((radar_summary,
                                                  _masked_mean(tokens, valid), reference)))
        mean, logvar = self.latent_prior(torch.cat((context, _masked_mean(tokens, valid),
                                                    _masked_max(tokens, valid)))).chunk(2)
        return dict(context=context, tokens=tokens, valid=valid, origins=origins,
                    directions=directions, lo=bounds[:, 0], hi=bounds[:, 1], scale=scale,
                    latent_mean=mean, latent_logvar=logvar.clamp(-8., 3.),
                    radar_summary=radar_summary, reference=reference,
                    radar_token_count=len(radar_tokens), shared_range_rows=len(ranges))

    def encode(self, features):
        return self.conditional(features)['context']

    def formation_logits(self, latent):
        return self.formation_head(latent if isinstance(latent, torch.Tensor) else self._tensor(latent))

    def _decode(self, condition, latent, rng=None, stochastic=False):
        n = len(condition['tokens'])
        values = self.joint_head(torch.cat((condition['tokens'], condition['context'].expand(n, -1),
                                            latent.expand(n, -1)), -1))
        logit_mean = values[:, 0]
        logit_std = (F.softplus(values[:, 1]) + .02).clamp_max(3.)
        logits = logit_mean
        if stochastic and n:
            if rng is None:
                logits = logits + logit_std * torch.randn_like(logits)
            else:
                logits = logits + logit_std * self._tensor(rng.normal(size=n))
        width = (condition['hi'] - condition['lo']).clamp_min(0.)
        depths = condition['lo'] + width * torch.sigmoid(logits)
        valid = condition['valid']
        depths = torch.where(valid, depths, torch.zeros_like(depths))
        points = condition['origins'] + depths[:, None] * condition['directions']
        points = torch.where(valid[:, None], points, torch.zeros_like(points))
        scene = dict(depths=depths, member_positions=points, center=_masked_mean(points, valid),
                     valid_mask=valid, latent=latent, depth_logit_mean=logit_mean,
                     depth_logit_std=logit_std)
        return scene

    def prior_prediction(self, features):
        condition = self.conditional(features)
        return self._decode(condition, condition['latent_mean'])

    def propose(self, features, count=32, rng=None, *, allow_untrained_validation=False):
        if not self.trained and not allow_untrained_validation:
            raise RuntimeError('RA/RE inference requires trained matching weights; use explicit untrained validation only for interface checks')
        if int(count) < 1:
            raise ValueError('Positive joint hypothesis count required')
        rng = np.random.default_rng() if rng is None else rng
        self.eval()
        result = []
        with torch.no_grad():
            condition = self.conditional(features)
            mean = condition['latent_mean']
            std = (.5 * condition['latent_logvar']).exp()
            for k in range(int(count)):
                latent = mean if k == 0 else mean + std * self._tensor(rng.normal(size=self.latent_dim))
                scene = self._decode(condition, latent, rng, k > 0)
                row = {key: value.cpu().numpy() for key, value in scene.items()
                       if key not in ('depth_logit_mean', 'depth_logit_std')}
                row.update(source='learned_ra_re_continuous_joint_depth' if self.trained else 'untrained_validation_only',
                           mode_source='conditional_latent_mean_decoder_not_map_or_mode' if k == 0 else
                                       'conditional_joint_continuous_sample',
                           model_trained=bool(self.trained))
                result.append(row)
        self.last_proposal_diagnostics = dict(
            schema=self.schema, group_hypotheses=len(result), latent_dim=self.latent_dim,
            detected_members=len(features['ray_origins']), valid_rays=int(condition['valid'].sum()),
            decoder='shared latent, set interaction, continuous logistic-normal joint depths',
            inputs=['ray_origins_base', 'ray_directions_base', 'observer_position_base',
                    'ra_power_with_physical_axes', 're_power_with_physical_axes', 'valid_masks'],
            range_association='RA/RE row fusion on the same physical range axis; no hard peak assignment',
            ray_depth_grid=False, global_depth_grid=False, appearance_used=False, iq_input_used=False,
            radar_token_count=condition['radar_token_count'], shared_range_rows=condition['shared_range_rows'],
            model_trained=bool(self.trained), untrained_validation=not self.trained,
            first_hypothesis='conditional latent mean with deterministic decoder; not a MAP estimate',
            exact_visual_rays=True, temporal_transition=False, reads_labels=False, reads_future=False,
            input_frames=1, history_used=False,
            formation_count_lookup=False, support_kind='uncalibrated conditional generation; not posterior probability',
            uncertainty_calibrated=False)
        return result

    def _label_assignment(self, condition, positions):
        true = self._tensor(positions)
        if true.ndim != 2 or true.shape[1] != 3 or not len(true) or not bool(torch.isfinite(true).all()):
            raise ValueError('Separate nonempty finite 3D supervision required')
        indices = condition['valid'].nonzero(as_tuple=False).flatten()
        if not len(indices):
            raise ValueError('No valid observed rays for supervised training')
        geometry = {key: condition[name].detach().cpu().numpy() for key, name in
                    (('ray_origins', 'origins'), ('ray_directions', 'directions'), ('ray_valid', 'valid'))}
        geometry.update(ray_depth_bounds=torch.stack((condition['lo'], condition['hi']), -1).detach().cpu().numpy(),
                        range_scale_m=condition['scale'])
        matched = match_ray_labels(geometry, true, self.config)
        row = torch.as_tensor(matched['rows'], device=true.device)
        col = torch.as_tensor(matched['cols'], device=true.device)
        depth = ((true[col] - condition['origins'][row]) * condition['directions'][row]).sum(-1)
        return true, row, col, depth, matched['diagnostics']

    def training_loss(self, features, positions, velocities=None, formation_index=None):
        """Supervise current positions; velocities are not model observations."""
        condition = self.conditional(features)
        true, row, col, depth, matching = self._label_assignment(condition, positions)
        prior = self._decode(condition, condition['latent_mean'])
        if not len(row):
            # No synthetic match or posterior label is constructed. Callers
            # must exclude this sample from optimizer/gradient denominators.
            zero = prior['depth_logit_mean'].sum() * 0.
            return zero, dict(matching, depth_nll=zero, depth_regression=zero,
                              shape_reconstruction=zero, prior_shape_loss=zero,
                              latent_kl=zero, formation_loss=zero, center_nll=zero,
                              supervised_depth_outside_support=0, prediction=prior,
                              posterior_latent=None, uncertainty_calibrated=False)
        point = self.point_encoder(true[col] / condition['scale'])
        qm, ql = self.latent_posterior(torch.cat((condition['context'], point.mean(0), point.max(0).values))).chunk(2)
        ql = ql.clamp(-8., 3.)
        latent = qm + torch.randn_like(qm) * (.5 * ql).exp()
        posterior = self._decode(condition, latent)
        width = (condition['hi'][row] - condition['lo'][row]).clamp_min(1e-6)
        # Matching already rejected unsupported labels. This remaining clamp
        # only keeps the logistic density finite at a valid interval endpoint.
        unit = ((depth - condition['lo'][row]) / width).clamp(1e-5, 1. - 1e-5)
        target_logit = torch.logit(unit)
        def supervised(decoded):
            mean = decoded['depth_logit_mean'][row]
            std = decoded['depth_logit_std'][row]
            # Exact change of variables for logistic-normal density in metres.
            nll = (.5 * ((target_logit - mean) / std).square() + std.log() +
                   .5 * math.log(2. * math.pi) + width.log() + unit.log() + torch.log1p(-unit)).mean()
            regression = ((decoded['depths'][row] - depth) / condition['scale']).square().mean()
            position_loss = ((decoded['member_positions'][row] - true[col]) / condition['scale']).square().mean()
            return nll, regression, position_loss
        nll, regression, state_loss = supervised(posterior)
        prior_nll, prior_regression, prior_state_loss = supervised(prior)
        kl = .5 * (condition['latent_logvar'] - ql +
                    (ql.exp() + (qm - condition['latent_mean']).square()) / condition['latent_logvar'].exp() - 1.).mean()
        formation = nll * 0.
        formation_weight = float(self.config.get('group_formation_loss_weight', 0.))
        if formation_weight and formation_index is not None:
            label = torch.tensor([int(formation_index)], device=true.device)
            formation = .5 * (F.cross_entropy(self.formation_logits(latent)[None], label) +
                              F.cross_entropy(self.formation_logits(condition['latent_mean'])[None], label))
        loss = .5 * (nll + prior_nll) + 20. * (state_loss + prior_state_loss) + .03 * kl + formation_weight * formation
        return loss, dict(matching, depth_nll=prior_nll, depth_regression=prior_regression,
                          shape_reconstruction=state_loss, prior_shape_loss=prior_state_loss,
                          latent_kl=kl, formation_loss=formation, center_nll=prior_nll,
                          supervised_depth_outside_support=0,
                          prediction=prior, posterior_latent=latent,
                          uncertainty_calibrated=False)


def checkpoint(model, steps, **metadata):
    if int(steps) < 1:
        raise ValueError('At least one real optimization step required')
    reserved = {'kind', 'group_schema', 'steps', 'model_config', 'state_dict', 'uncertainty_calibrated'}
    if reserved.intersection(metadata):
        raise ValueError('Checkpoint metadata cannot override architecture or training fields')
    return dict(kind='group_scene_world_model', group_schema=model.schema, steps=int(steps),
                model_config=dict(model.config), state_dict=model.state_dict(), uncertainty_calibrated=False, **metadata)


def load_group_model(path, config=None, device='cpu'):
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    from .single_frame import require_single_frame
    if value.get('kind') != 'group_scene_world_model' or value.get('group_schema') != GROUP_SCHEMA:
        raise ValueError('Unsupported RA/RE checkpoint schema; previous IQ/depth-grid weights require retraining')
    if int(value.get('steps', 0)) < 1:
        raise ValueError('Untrained RA/RE checkpoint')
    model = GroupSceneModel(value['model_config'])
    if value['group_schema'] != model.schema:
        raise ValueError('Checkpoint schema differs from its stored model configuration')
    if config is not None:
        if config.get('architecture_schema', GROUP_SCHEMA) != model.schema:
            raise ValueError('RA/RE checkpoint architecture mismatch: architecture_schema')
        defaults = dict(group_latent_dim=8, group_hidden_dim=64, group_max_members=24,
                        group_attention_heads=4, group_interaction_layers=2, group_spectrum_angle_tokens=8)
        for name, default in defaults.items():
            if name in config and int(config[name]) != int(model.config.get(name, default)):
                raise ValueError('RA/RE checkpoint architecture mismatch: ' + name)
        require_single_frame(config)
    model.load_state_dict(value['state_dict'], strict=True)
    model.to(device).eval()
    model.trained = True
    model.checkpoint_metadata = {key: item for key, item in value.items() if key != 'state_dict'}
    return model
