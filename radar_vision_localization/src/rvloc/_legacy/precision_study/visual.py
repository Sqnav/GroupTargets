"""Trained visual/navigation control with no radar inputs or radar modules."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from depth_localization.model import (_mlp, _interaction, _interact, _mean,
                                      _NativeSpectrumEncoder, DepthLocalizer)
from rvwm.group_model import match_ray_labels

SCHEMA = 'visual_bbox_base_rays_v1'
ARRAY_FIELDS = ('ray_origins', 'ray_directions', 'ray_valid', 'ray_depth_bounds',
                'observer_position', 'bbox_xyxy', 'detection_scores', 'visual_scale',
                'camera_K', 'camera_image_size_wh')
ALLOWED_FIELDS = frozenset(ARRAY_FIELDS) | {'schema', 'depth_schema', 'range_scale_m', 'metadata'}


def validate_visual_features(features):
    """Strict allowlist: radar and target fields are rejected, never ignored."""
    if set(features) != ALLOWED_FIELDS or features.get('schema') != SCHEMA:
        raise ValueError('Only the explicit visual/navigation feature contract is accepted')
    n = len(features['ray_directions'])
    shapes = dict(ray_origins=(n, 3), ray_directions=(n, 3), ray_valid=(n,),
                  ray_depth_bounds=(n, 2), observer_position=(3,), bbox_xyxy=(n, 4),
                  detection_scores=(n,), visual_scale=(n, 7), camera_K=(3, 3), camera_image_size_wh=(2,))
    values = {}
    for name, shape in shapes.items():
        value = features[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != shape or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError('Invalid visual input: ' + name)
        values[name] = value
    valid = values['ray_valid']
    if valid.dtype != np.bool_:
        raise ValueError('ray_valid must be boolean; it is a geometry mask, not confidence')
    if not np.allclose(np.linalg.norm(values['ray_directions'][valid], axis=-1), 1., rtol=1e-6, atol=1e-6):
        raise ValueError('Valid visual rays must be unit vectors')
    bounds = values['ray_depth_bounds'][valid]
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Positive nonempty geometric depth support required')
    scale = features['range_scale_m']
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Public positive sensor range support required')
    boxes, k = values['bbox_xyxy'], values['camera_K']
    w, h = (boxes[:, 2:] - boxes[:, :2]).T
    if np.any(w <= 0) or np.any(h <= 0) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Positive observed boxes and calibrated focal lengths required')
    scores = values['detection_scores']
    expected = np.column_stack((w/k[0, 0], h/k[1, 1], np.sqrt(w*h/(k[0, 0]*k[1, 1])),
                                np.log(w/k[0, 0]), np.log(h/k[1, 1]), np.log(w/h), scores))
    if not np.allclose(expected, values['visual_scale'], rtol=1e-6, atol=1e-7):
        raise ValueError('Visual scales do not match observed boxes and calibration')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Detection scores must be in [0,1]')
    return features


def drop_visual_detections(features, keep):
    from copy import deepcopy
    validate_visual_features(features)
    keep = np.asarray(keep)
    if keep.shape != (len(features['ray_directions']),) or keep.dtype != np.bool_:
        raise ValueError('Detection retention mask must be boolean and detection-aligned')
    result = deepcopy(features)
    for key in ('ray_origins', 'ray_directions', 'ray_valid', 'ray_depth_bounds',
                'bbox_xyxy', 'detection_scores', 'visual_scale'):
        result[key] = np.asarray(features[key])[keep].copy()
    return validate_visual_features(result)


class VisualOnlyDepthLocalizer(nn.Module):
    """Same visual/set refinement width, depth budgets and loss as current v3.

    The RA/RE encoders and local spectrum query networks are absent. Navigation
    remains the only global memory. The former local radar slot is explicitly
    zero inside the network, not a synthetic spectrum accepted by a validator.
    """
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if type(config.get('window', 1)) is not int or config.get('window', 1) != 1:
            raise ValueError('Visual control accepts one current frame')
        h = int(config.get('depth_hidden_dim', 64))
        heads = int(config.get('depth_attention_heads', 4))
        layers = int(config.get('depth_interaction_layers', 2))
        self.steps = int(config.get('depth_refinement_steps', 2))
        self.max_correction = float(config.get('depth_max_correction_m', 16.))
        self.fine_enabled = 'depth_fine_offsets_m' in config
        self.fine_max_correction = float(config.get('depth_fine_max_correction_m', self.max_correction))
        self.huber_delta = float(config.get('depth_huber_delta_m', 1.))
        self.initial_weight = float(config.get('depth_initial_loss_weight', .3))
        self.mse_weight = float(config.get('depth_loss_mse_weight', .05))
        if h < 8 or h % heads or min(heads, layers, self.steps) < 1 or self.steps > 2:
            raise ValueError('Invalid visual control dimensions')
        if (not all(math.isfinite(x) for x in (self.max_correction, self.fine_max_correction,
                self.huber_delta, self.initial_weight, self.mse_weight)) or self.max_correction <= 0
                or not 0 < self.fine_max_correction <= self.max_correction or self.huber_delta <= 0
                or min(self.initial_weight, self.mse_weight) < 0):
            raise ValueError('Invalid metre-level visual-control settings')
        self.max_members = int(config.get('group_max_members', 24))
        self.register_buffer('depth_prior_m', torch.tensor(20.))
        self.visual_encoder = _mlp(18, h, h)
        self.visual_interaction = _interaction(h, heads, layers)
        self.coarse_head = _mlp(h, h, 1)
        nn.init.zeros_(self.coarse_head[-1].weight); nn.init.zeros_(self.coarse_head[-1].bias)
        # Match fresh v3's random initialization for every retained block.
        # These temporary modules only consume RNG during construction. They
        # are discarded immediately, never registered or given observations.
        _NativeSpectrumEncoder(h); _NativeSpectrumEncoder(h)
        self.observer_encoder = _mlp(3, h, h)
        self.global_attention = nn.MultiheadAttention(h, heads, dropout=0., batch_first=True)
        self.global_norm = nn.LayerNorm(h)
        self.global_head = _mlp(h, h, 2)
        nn.init.zeros_(self.global_head[-1].weight)
        with torch.no_grad(): self.global_head[-1].bias.copy_(torch.tensor([0., -4.]))
        _mlp(2*h+10, h, h); _mlp(2*h, h, 1)
        self.refine_encoder = _mlp(3*h+1, h, h)
        self.refine_interaction = _interaction(h, heads, layers)
        self.refine_head = _mlp(h, h, 1)
        self.trained = False
        self.initialization_contract = 'retained blocks match same-seed fresh v3; no weights loaded'

    _interior_bounds = staticmethod(DepthLocalizer._interior_bounds)
    _bounded = classmethod(DepthLocalizer._bounded.__func__)
    _support_residual = DepthLocalizer._support_residual

    def _tensor(self, value):
        p = next(self.parameters())
        return torch.as_tensor(value, dtype=p.dtype, device=p.device)

    def initialize_depth_prior(self, depth_m):
        depth_m = float(depth_m)
        if not math.isfinite(depth_m) or depth_m <= 0: raise ValueError('Positive train-only prior required')
        with torch.no_grad(): self.depth_prior_m.fill_(depth_m)

    def forward(self, features):
        validate_visual_features(features)
        origin, direction = self._tensor(features['ray_origins']), self._tensor(features['ray_directions'])
        bounds = self._tensor(features['ray_depth_bounds'])
        valid = torch.as_tensor(features['ray_valid'], dtype=torch.bool, device=origin.device)
        if len(origin) > self.max_members: raise ValueError('Too many observed detections')
        observer = self._tensor(features['observer_position']); scale = float(features['range_scale_m'])
        visual = self._tensor(features['visual_scale'])
        values = torch.cat((origin/scale, direction, (origin-observer[None])/scale, bounds/scale, visual), -1)
        tokens = _interact(self.visual_interaction, self.visual_encoder(values), valid)
        anchor = self.depth_prior_m.expand(len(origin))*.05/visual[:, 2].clamp_min(1e-5)
        initial = self._bounded(anchor, self.coarse_head(tokens)[:, 0], bounds)*valid
        if not len(origin) or not bool(valid.any()):
            return dict(depths=initial, initial_depths=initial, member_positions=origin*0.,
                        initial_positions=origin*0., valid_mask=valid, center=origin.new_zeros(3),
                        sampling_diagnostics=dict(stage_depths_m=[], radar_used=False))
        reference = self.observer_encoder(observer/scale)
        context, _ = self.global_attention(tokens[None], reference[None, None], reference[None, None], need_weights=False)
        global_token = self.global_norm(tokens+context[0])*valid[:, None]
        proposal = self.global_head(global_token)
        global_depth = self._bounded(self.depth_prior_m.expand(len(origin)), proposal[:, 0], bounds)*valid
        gate = torch.sigmoid(proposal[:, 1])*valid
        depth = initial; stages=[]; gates=[]
        for step in range(self.steps):
            fine = self.fine_enabled and step == 1
            applied_gate = torch.zeros_like(gate) if fine else gate
            refine = self.refine_encoder(torch.cat((tokens, global_token, torch.zeros_like(tokens), depth[:, None]/scale), -1))
            refine = _interact(self.refine_interaction, refine, valid)
            base = (1.-applied_gate)*depth+applied_gate*global_depth
            depth = self._support_residual(base, torch.tanh(self.refine_head(refine)[:, 0]), bounds,
                self.fine_max_correction if fine else self.max_correction)*valid
            stages.append(depth); gates.append(applied_gate); tokens = refine
        return dict(depths=depth, initial_depths=initial, member_positions=(origin+depth[:, None]*direction)*valid[:, None],
            initial_positions=(origin+initial[:, None]*direction)*valid[:, None], valid_mask=valid,
            center=_mean(origin+depth[:, None]*direction, valid),
            sampling_diagnostics=dict(stage_depths_m=stages, global_depths=global_depth, global_gate=gates,
                radar_used=False, refinement_mode='visual_navigation_control'))

    def training_loss(self, features, positions, formation_index=None):
        p = self(features); matched = match_ray_labels(features, positions, self.config)
        rows = torch.as_tensor(matched['rows'], dtype=torch.long, device=p['depths'].device)
        cols = torch.as_tensor(matched['cols'], dtype=torch.long, device=p['depths'].device)
        zero = p['depths'].sum()*0.
        if not len(rows): return zero, dict(matched['diagnostics'], initial_depth_huber=zero,
            final_depth_huber=zero, final_depth_mse=zero, matched_rmse_m=zero, prediction=p)
        target = self._tensor(matched['depths_m'])
        initial = F.huber_loss(p['initial_depths'][rows], target, delta=self.huber_delta)
        final = F.huber_loss(p['depths'][rows], target, delta=self.huber_delta)
        mse = F.mse_loss(p['depths'][rows], target)
        rmse = (p['member_positions'][rows]-self._tensor(positions)[cols]).square().sum(-1).mean().sqrt()
        return final+self.mse_weight*mse+self.initial_weight*initial, dict(matched['diagnostics'],
            initial_depth_huber=initial, final_depth_huber=final, final_depth_mse=mse,
            matched_rmse_m=rmse, prediction=p)

    def predict(self, features, allow_untrained_validation=False):
        if not self.trained and not allow_untrained_validation: raise RuntimeError('Train visual control before inference')
        def convert(x):
            if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
            if isinstance(x, dict): return {k:convert(v) for k,v in x.items()}
            if isinstance(x, list): return [convert(v) for v in x]
            return x
        with torch.no_grad(): result = convert(self(features))
        result.update(model_trained=self.trained, source='trained_visual_navigation_control',
                      depth_unit='camera slant metres', input_frames=1, radar_used=False,
                      uncertainty_calibrated=False)
        return result
