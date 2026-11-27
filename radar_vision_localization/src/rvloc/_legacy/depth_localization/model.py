"""Deterministic single-frame, geometry-guided, metre-supervised localization.

Only visual observations and mixed RA/RE enter forward. GT is a separate
training_loss argument. There is no latent, posterior, NLL, mode sampling,
per-member peak assignment, spectrum renderer, or claimed uncertainty.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rvwm.group_model import match_ray_labels
from .geometry import candidate_geometry, sample_paired_spectra

SCHEMA = 'single_frame_geometry_guided_depth_v2'


def _mlp(a, h, b):
    return nn.Sequential(nn.Linear(a, h), nn.SiLU(), nn.Linear(h, b))


def _mean(value, valid):
    return (value*valid[:, None]).sum(0)/valid.sum().clamp_min(1)


def _interaction(hidden, heads, layers):
    block = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0.,
        activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, enable_nested_tensor=False)


def _interact(module, tokens, valid):
    # No all-masked softmax, and padded detections never affect real members.
    if len(tokens) and bool(valid.any()):
        tokens = module(tokens[None], src_key_padding_mask=~valid[None])[0]
    return tokens*valid[:, None]


class _NativeSpectrumEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(5, hidden//2, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden//2, hidden, 3, padding=1), nn.SiLU())
        self.row = _mlp(2*hidden, hidden, hidden)

    def forward(self, power, mask, ranges, angles, scale, power_scale):
        nr, na = power.shape
        values = torch.stack((torch.log1p(power.masked_fill(~mask, 0.)/power_scale),
            (ranges/scale)[:, None].expand(nr, na), angles.sin()[None].expand(nr, na),
            angles.cos()[None].expand(nr, na), mask.to(power.dtype)))
        native = self.cnn(values[None])[0]*mask[None]
        mean = native.sum(-1).T/mask.sum(-1).clamp_min(1)[:, None]
        maximum = native.masked_fill(~mask[None], -torch.inf).max(-1).values.T
        maximum = torch.where(mask.any(-1)[:, None], maximum, torch.zeros_like(maximum))
        rows = self.row(torch.cat((mean, maximum), -1))
        return native, rows, mask.any(-1)


class DepthLocalizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.schema = self.config.get('architecture_schema', SCHEMA)
        if self.schema != SCHEMA or type(self.config.get('window', 1)) is not int or self.config.get('window', 1) != 1:
            raise ValueError('DepthLocalizer requires the single-frame geometry-guided depth v2 architecture')
        if self.config.get('group_temporal_enabled', False) is not False or any(
                key in self.config for key in ('group_temporal_layers', 'group_temporal_heads')):
            raise ValueError('Historical or temporal model inputs are unsupported')
        h = int(config.get('depth_hidden_dim', config.get('group_hidden_dim', 64)))
        heads = int(config.get('depth_attention_heads', 4))
        layers = int(config.get('depth_interaction_layers', 2))
        self.steps = int(config.get('depth_refinement_steps', 2))
        self.max_correction = float(config.get('depth_max_correction_m', 16.))
        self.huber_delta = float(config.get('depth_huber_delta_m', 1.))
        self.initial_weight = float(config.get('depth_initial_loss_weight', .3))
        raw_mse_weight = config.get('depth_loss_mse_weight', 0.)
        if isinstance(raw_mse_weight, (bool, np.bool_)):
            raise ValueError('Depth MSE weight must be a real nonnegative number')
        try:
            self.mse_weight = float(raw_mse_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Depth MSE weight must be a real nonnegative number') from exc
        if h < 8 or heads < 1 or h % heads or layers < 1 or self.steps not in (1, 2):
            raise ValueError('Invalid deterministic depth model dimensions or refinement count')
        if not math.isfinite(self.max_correction) or self.max_correction <= 0 or self.huber_delta <= 0 or not math.isfinite(self.huber_delta) or self.initial_weight < 0 or not math.isfinite(self.initial_weight) or not math.isfinite(self.mse_weight) or self.mse_weight < 0:
            raise ValueError('Invalid metre loss or refinement settings')
        offsets = torch.as_tensor(config.get('depth_offsets_m', [0., -2., 2., -4., 4., -8., 8., -16., 16.]), dtype=torch.float32)
        if offsets.ndim != 1 or not len(offsets) or not bool(torch.isfinite(offsets).all()) or not bool((offsets == 0).any()):
            raise ValueError('Depth offsets must be finite and include zero')
        self.register_buffer('depth_offsets_m', offsets)
        # The interface schema still describes a single continuous depth per
        # current-frame ray. Exact source/config hashes bind this optional
        # coarse/fine design separately from the previous shared-scale model.
        self.fine_enabled = 'depth_fine_offsets_m' in config
        self.fine_max_correction = self.max_correction
        fine_offsets = None
        if self.fine_enabled:
            if self.steps != 2:
                raise ValueError('Fine depth offsets require exactly two refinement steps')
            try:
                raw_fine = np.asarray(config['depth_fine_offsets_m'])
                if raw_fine.dtype == np.bool_ or not np.issubdtype(raw_fine.dtype, np.number):
                    raise ValueError('Fine depth offsets must be real metre values')
                if np.iscomplexobj(raw_fine):
                    raise ValueError('Fine depth offsets must be real metre values')
                fine_offsets = torch.as_tensor(raw_fine, dtype=torch.float32)
                fine_correction = config.get('depth_fine_max_correction_m', 2.)
                if isinstance(fine_correction, (bool, np.bool_)):
                    raise ValueError('Fine correction must be a real metre value')
                self.fine_max_correction = float(fine_correction)
            except (TypeError, OverflowError) as exc:
                raise ValueError('Invalid fine depth refinement settings') from exc
            if (not math.isfinite(self.fine_max_correction) or self.fine_max_correction <= 0
                    or self.fine_max_correction > self.max_correction):
                raise ValueError('Fine correction must be positive and no larger than coarse correction')
            if (fine_offsets.ndim != 1 or not len(fine_offsets)
                    or not bool(torch.isfinite(fine_offsets).all())
                    or not bool((fine_offsets == 0).any())
                    or len(torch.unique(fine_offsets)) != len(fine_offsets)
                    or bool((fine_offsets.abs() > self.fine_max_correction).any())):
                raise ValueError('Fine depth offsets must be finite, unique, include zero and fit the fine correction interval')
        elif 'depth_fine_max_correction_m' in config:
            raise ValueError('Fine correction requires explicit fine depth offsets')
        self.register_buffer('depth_fine_offsets_m', fine_offsets)
        self.register_buffer('depth_prior_m', torch.tensor(20.))
        self.scale_enabled = bool(config.get('depth_visual_scale_enabled', True))
        self.geometry_enabled = bool(config.get('depth_geometry_enabled', True))
        self.max_members = int(config.get('group_max_members', 24))
        self.visual_encoder = _mlp(18, h, h)
        self.visual_interaction = _interaction(h, heads, layers)
        self.coarse_head = _mlp(h, h, 1)
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        self.ra_encoder = _NativeSpectrumEncoder(h)
        self.re_encoder = _NativeSpectrumEncoder(h)
        self.observer_encoder = _mlp(3, h, h)
        self.global_attention = nn.MultiheadAttention(h, heads, dropout=0., batch_first=True)
        self.global_norm = nn.LayerNorm(h)
        self.global_head = _mlp(h, h, 2)
        nn.init.zeros_(self.global_head[-1].weight)
        with torch.no_grad():
            self.global_head[-1].bias.copy_(torch.tensor([0., -4.]))
        self.sample_encoder = _mlp(2*h+10, h, h)
        self.sample_score = _mlp(2*h, h, 1)
        self.refine_encoder = _mlp(3*h+1, h, h)
        self.refine_interaction = _interaction(h, heads, layers)
        self.refine_head = _mlp(h, h, 1)
        self.trained = False
        self.last_proposal_diagnostics = None

    def initialize_depth_prior(self, depth_m):
        """Optional train-only scalar initialization, recorded by the trainer."""
        value = float(depth_m)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Training depth prior must be positive finite metres')
        with torch.no_grad():
            self.depth_prior_m.fill_(value)

    def _tensor(self, value):
        p = next(self.parameters())
        return torch.as_tensor(value, dtype=p.dtype, device=p.device)

    def _bool(self, value):
        return torch.as_tensor(value, dtype=torch.bool, device=self.depth_prior_m.device)

    def _validate(self, features):
        from .features import validate_depth_features
        # Preserve autograd on supplied spectrum tensors while checking their
        # detached values with the same public feature contract as NumPy data.
        view = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        validate_depth_features(view)
        if len(features['ray_origins']) > self.max_members:
            raise ValueError('Detected count exceeds configured model support')
        forbidden = {'positions', 'target_positions', 'true_positions', 'true_depths',
                     'ground_truth', 'gt', 'formation_index', 'formation_type',
                     'depth_labels', 'target_velocities', 'true_count', 'targets', 'future',
                     'future_frames', 'iq', 'appearance', 'visual_tokens', 'labels',
                     'depth_candidates', 'ray_evidence', 'member_velocities'}
        if forbidden.intersection(features):
            raise ValueError('Only observed visual scale, rays, observer and RA/RE may enter inference')

    @staticmethod
    def _interior_bounds(bounds):
        # Reserve a few floating-point ULPs so even saturated sigmoid/tanh and
        # repeated refinements cannot round onto the array-front plane. This
        # numerical inset is <=1% of support, usually a few millimetres.
        width = (bounds[:, 1]-bounds[:, 0]).clamp_min(0.)
        magnitude = bounds.abs().amax(-1).clamp_min(1.)
        inset = torch.minimum(width*.01, 64.*torch.finfo(bounds.dtype).eps*magnitude)
        return torch.stack((bounds[:, 0]+inset, bounds[:, 1]-inset), -1)

    @classmethod
    def _bounded(cls, anchor, residual_logit, bounds):
        interior = cls._interior_bounds(bounds)
        width = (interior[:, 1]-interior[:, 0]).clamp_min(1e-6)
        fraction = ((anchor-interior[:, 0])/width).clamp(1e-5, 1.-1e-5)
        return interior[:, 0]+width*torch.sigmoid(torch.logit(fraction)+residual_logit)

    def _support_residual(self, base, raw, bounds, max_correction=None):
        """Directional, support-aware metre residual without output clipping."""
        correction = self.max_correction if max_correction is None else max_correction
        interior = self._interior_bounds(bounds)
        negative = torch.minimum((base-interior[:, 0]).clamp_min(0.),
                                 torch.full_like(base, correction))*.99
        positive = torch.minimum((interior[:, 1]-base).clamp_min(0.),
                                 torch.full_like(base, correction))*.99
        return base+torch.where(raw >= 0., positive, negative)*raw

    def forward(self, features):
        self._validate(features)
        origin = self._tensor(features['ray_origins']); direction = self._tensor(features['ray_directions'])
        bounds = self._tensor(features['ray_depth_bounds']); valid = self._bool(features['ray_valid'])
        observer = self._tensor(features['observer_position']); visual = self._tensor(features['visual_scale'])
        scale = float(features['range_scale_m']); n = len(origin)
        if not self.scale_enabled:
            visual = torch.zeros_like(visual)
        visual_input = torch.cat((origin/scale, direction, (origin-observer[None])/scale,
                                  bounds/scale, visual), -1)
        tokens = _interact(self.visual_interaction, self.visual_encoder(visual_input), valid)
        # The train-initialized scalar has metre units. 0.05 is only a
        # dimensionless reference angular size, not a known UAV body size.
        anchor = self.depth_prior_m.expand(n)
        if self.scale_enabled:
            anchor = anchor*.05/visual[:, 2].clamp_min(1e-5)
        initial = self._bounded(anchor, self.coarse_head(tokens)[:, 0], bounds)
        initial = initial*valid
        if not n or not bool(valid.any()):
            zero = origin*0.
            return dict(depths=initial, initial_depths=initial, member_positions=zero,
                initial_positions=zero, valid_mask=valid, center=origin.new_zeros(3),
                sampling_diagnostics=dict(local_valid_fraction=[], global_gate=[],
                    stage_candidate_offsets_m=[], stage_max_correction_m=[], stage_depths_m=[],
                    refinement_mode='coarse_fine' if self.fine_enabled else 'shared_scale',
                    geometry_enabled=self.geometry_enabled, global_proposal_is_search=False))
        ra, re = self._tensor(features['ra_power']), self._tensor(features['re_power'])
        av, ev = self._bool(features['ra_valid']), self._bool(features['re_valid'])
        ranges = self._tensor(features['range_m']); az = self._tensor(features['azimuth_rad']); el = self._tensor(features['elevation_rad'])
        power_scale = ((ra.masked_fill(~av, 0.).sum()+re.masked_fill(~ev, 0.).sum())/
                       (av.sum()+ev.sum()).clamp_min(1)).clamp_min(1e-20)
        ra_map, ra_rows, ra_rows_valid = self.ra_encoder(ra, av, ranges, az, scale, power_scale)
        re_map, re_rows, re_rows_valid = self.re_encoder(re, ev, ranges, el, scale, power_scale)
        reference = self.observer_encoder(observer/scale)
        memory = torch.cat((ra_rows, re_rows, reference[None]), 0)
        memory_valid = torch.cat((ra_rows_valid, re_rows_valid, valid.new_ones(1)))
        context, _ = self.global_attention(tokens[None], memory[None], memory[None],
                                          key_padding_mask=~memory_valid[None], need_weights=False)
        global_token = self.global_norm(tokens+context[0])*valid[:, None]
        global_values = self.global_head(global_token)
        global_depth = self._bounded(self.depth_prior_m.expand(n), global_values[:, 0], bounds)*valid
        global_gate = torch.sigmoid(global_values[:, 1])*valid
        depth = initial; fractions=[]; gates=[]; stage_offsets=[]; stage_limits=[]; stage_depths=[]
        for step in range(self.steps):
            fine_stage = self.fine_enabled and step == 1
            offsets = self.depth_fine_offsets_m if fine_stage else self.depth_offsets_m
            correction = self.fine_max_correction if fine_stage else self.max_correction
            hypotheses = depth[:, None]+offsets[None]
            geometry = candidate_geometry(origin, direction, hypotheses, bounds, valid)
            sampled = sample_paired_spectra(ra_map, re_map, ranges=geometry['ranges'],
                azimuth=geometry['azimuth'], elevation=geometry['elevation'], range_axis=ranges,
                azimuth_axis=az, elevation_axis=el, ra_valid=av, re_valid=ev,
                candidate_valid=geometry['valid'])
            usable = sampled['valid'] if self.geometry_enabled else torch.zeros_like(sampled['valid'])
            angular = torch.stack((geometry['azimuth'].sin(), geometry['azimuth'].cos(),
                geometry['elevation'].sin(), geometry['elevation'].cos()), -1)
            scalars = torch.cat((hypotheses[..., None]/scale, geometry['ranges'][..., None]/scale,
                (offsets/correction)[None, :, None].expand(n, -1, -1), angular,
                sampled['ra_valid'][..., None].to(ra.dtype), sampled['re_valid'][..., None].to(ra.dtype),
                geometry['valid'][..., None].to(ra.dtype)), -1)
            candidate = self.sample_encoder(torch.cat((sampled['ra'], sampled['re'], scalars), -1))
            logits = self.sample_score(torch.cat((candidate, global_token[:, None].expand(-1, len(offsets), -1)), -1))[..., 0]
            # All-invalid rows have exactly zero local evidence, never NaNs or
            # a fabricated match to zero-filled out-of-bounds power.
            weights = torch.softmax(logits.masked_fill(~usable, -1e4), -1)*usable
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
            local = (candidate*weights[..., None]).sum(1)
            refine = self.refine_encoder(torch.cat((tokens, global_token, local, depth[:, None]/scale), -1))
            refine = _interact(self.refine_interaction, refine, valid)
            raw = torch.tanh(self.refine_head(refine)[:, 0])
            # The global branch is a learned proposal, not a guaranteed global
            # depth search. It can recover beyond the local residual interval.
            # A fine pass only refines the coarse result locally. Reapplying
            # the global proposal here would bypass its metre correction limit.
            applied_gate = torch.zeros_like(global_gate) if fine_stage else global_gate
            base = (1.-applied_gate)*depth+applied_gate*global_depth
            depth = self._support_residual(base, raw, bounds, correction)*valid
            fractions.append(usable.to(ra.dtype).mean(-1)); gates.append(applied_gate)
            stage_offsets.append(offsets); stage_limits.append(correction); stage_depths.append(depth)
            tokens = refine
        positions = (origin+depth[:, None]*direction)*valid[:, None]
        initial_positions = (origin+initial[:, None]*direction)*valid[:, None]
        return dict(depths=depth, initial_depths=initial, member_positions=positions,
            initial_positions=initial_positions, valid_mask=valid, center=_mean(positions, valid),
            sampling_diagnostics=dict(local_valid_fraction=fractions, global_gate=gates,
                global_depths=global_depth, geometry_enabled=self.geometry_enabled,
                candidate_offsets_m=self.depth_offsets_m, global_proposal_is_search=False,
                stage_candidate_offsets_m=stage_offsets, stage_max_correction_m=stage_limits,
                stage_depths_m=stage_depths,
                refinement_mode='coarse_fine' if self.fine_enabled else 'shared_scale',
                mixed_observer_preserved=True, independent_peak_assignment=False))

    def training_loss(self, features, positions, formation_index=None):
        prediction = self(features)
        matched = match_ray_labels(features, positions, self.config)
        rows = torch.as_tensor(matched['rows'], dtype=torch.long, device=self.depth_prior_m.device)
        cols = torch.as_tensor(matched['cols'], dtype=torch.long, device=self.depth_prior_m.device)
        target_depth = self._tensor(matched['depths_m'])
        zero = prediction['depths'].sum()*0.
        if not len(rows):
            return zero, dict(matched['diagnostics'], initial_depth_huber=zero,
                final_depth_huber=zero, final_depth_mse=zero, matched_rmse_m=zero, prediction=prediction,
                loss_units='Huber plus weighted squared metre depth residuals; projected visual-ray labels; no range-scale normalization')
        initial = F.huber_loss(prediction['initial_depths'][rows], target_depth,
                               reduction='mean', delta=self.huber_delta)
        final = F.huber_loss(prediction['depths'][rows], target_depth,
                             reduction='mean', delta=self.huber_delta)
        final_mse = F.mse_loss(prediction['depths'][rows], target_depth, reduction='mean')
        true = self._tensor(positions)
        rmse = ((prediction['member_positions'][rows]-true[cols]).square().sum(-1).mean()).sqrt()
        return final+self.mse_weight*final_mse+self.initial_weight*initial, dict(matched['diagnostics'],
            initial_depth_huber=initial, final_depth_huber=final, final_depth_mse=final_mse, matched_rmse_m=rmse,
            prediction=prediction, loss_units='Huber plus weighted squared metre depth residuals; projected visual-ray labels; no range-scale normalization')

    def predict(self, features, allow_untrained_validation=False):
        if not self.trained and not allow_untrained_validation:
            raise RuntimeError('A trained depth model is required; allow_untrained_validation is diagnostics only')
        def detached(value):
            if isinstance(value, torch.Tensor):return value.detach().cpu().numpy()
            if isinstance(value, dict):return {k:detached(v) for k,v in value.items()}
            if isinstance(value, list):return [detached(v) for v in value]
            return value
        with torch.no_grad():
            result = detached(self(features))
        result.update(model_trained=bool(self.trained), source='trained_depth_localizer' if self.trained else 'untrained_validation_only',
            uncertainty_calibrated=False, hypothesis_index=0, input_frames=1,
            decoder='deterministic metric depth regression with geometry-guided RA/RE features',
            mode_source='single deterministic prediction; not posterior or MAP',
            depth_unit='camera slant metres', velocities_predicted=False)
        return result

    def propose(self, features, count=1, rng=None, allow_untrained_validation=False):
        if isinstance(count, bool) or count != 1:
            raise ValueError('Deterministic depth localization returns exactly one prediction')
        result = self.predict(features, allow_untrained_validation=allow_untrained_validation)
        self.last_proposal_diagnostics = dict(schema=self.schema, input_frames=1,
            group_hypotheses=1, random_sampling=False, model_trained=bool(self.trained),
            untrained_validation=not self.trained, uncertainty_calibrated=False,
            geometry_guided=self.geometry_enabled, independent_peak_assignment=False,
            global_proposal_is_search=False)
        return [result]
