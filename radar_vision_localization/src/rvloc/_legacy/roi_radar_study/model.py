"""Frozen-ROI visual/navigation control and unchanged 32/64 MHz ROI fusion.

The radar arms delegate to the original scalar and tensor-batched executor.
The visual arm has no radar encoders or spectrum inputs. Its original two
nine-candidate updates encode only candidate geometry and public navigation.
"""
from __future__ import annotations

import numpy as np
import torch

from d_evidence_speed.model import FastDepthLocalizer as OriginalFastDepthLocalizer
from d_evidence_speed.model import _bounded, _geometry, _interact
from depth_localization.model import _mlp
from precision_study.visual import ARRAY_FIELDS, validate_visual_features

ARMS = ('visual', '32mhz', '64mhz')
VISUAL_PACKED_FIELDS = frozenset(ARRAY_FIELDS) | {
    'range_scale_m', 'member_mask', 'detector_roi_features',
}


class StudyDepthLocalizer(OriginalFastDepthLocalizer):
    def __init__(self, config):
        arm = config.get('study_arm')
        if arm not in ARMS:
            raise ValueError('An explicit visual/32mhz/64mhz study_arm is required')
        if config.get('d_evidence_arm') != 'roi':
            raise ValueError('All study arms require the same frozen ROI visual input branch')
        super().__init__(config)
        if not self.roi_enabled or self.sequence_enabled or self.aligned_enabled:
            raise ValueError('Keep the original ROI architecture without other evidence changes')
        self.study_arm = arm
        self.radar_used = arm != 'visual'
        if not self.radar_used:
            self.config['precision_mode'] = 'visual'
            # Construction above preserves the exact same-seed initialization
            # of every retained original block. Unused radar modules are then
            # removed, not frozen alongside fictitious zero-valued observations.
            del self.ra_encoder
            del self.re_encoder
            del self.sample_encoder
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261125)
                hidden = self.visual_encoder[0].out_features
                self.geometry_encoder = _mlp(8, hidden, hidden)

    def pack_batch(self, features_list, positions=None, matches=None):
        from .data import pack_batch
        return pack_batch(features_list, positions=positions, matches=matches,
                          device=self.depth_prior_m.device, config=self.config)

    def _packed(self, value, positions=None, matches=None):
        batch = super()._packed(value, positions, matches)
        if not self.radar_used and set(batch.inputs) != VISUAL_PACKED_FIELDS:
            extra = set(batch.inputs) - VISUAL_PACKED_FIELDS
            missing = VISUAL_PACKED_FIELDS - set(batch.inputs)
            raise ValueError('Visual packed input forbids radar/GT/unknown fields and requires its exact '
                             'contract; extra='+str(sorted(extra))+' missing='+str(sorted(missing)))
        return batch

    def _validate_visual(self, features):
        if not isinstance(features, dict) or 'detector_roi_features' not in features:
            raise ValueError('A visual observation requires frozen detection-aligned ROI vectors')
        validate_visual_features({key: value for key, value in features.items()
                                  if key != 'detector_roi_features'})
        n = len(features['ray_directions'])
        values = features['detector_roi_features']
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.shape != (n, 1024) or not np.isfinite(values).all():
            raise ValueError('Finite frozen ROI vectors must align with the current detections')
        if n > self.max_members:
            raise ValueError('Detected count exceeds configured model support')

    def forward_batch(self, value):
        if self.radar_used:
            return super().forward_batch(value)
        batch = self._packed(value); x = batch.inputs
        origin = x['ray_origins']; direction = x['ray_directions']; bounds = x['ray_depth_bounds']
        valid = x['ray_valid'] & x['member_mask']; observer = x['observer_position']
        visual = x['visual_scale']; scale = x['range_scale_m']; b, n = valid.shape
        if not self.scale_enabled:
            visual = torch.zeros_like(visual)
        visual_input = torch.cat((origin/scale[:, None, None], direction,
            (origin-observer[:, None])/scale[:, None, None], bounds/scale[:, None, None], visual), -1)
        encoded = self.visual_encoder(visual_input)
        encoded = encoded+self.roi_projection(x['detector_roi_features'].detach())
        tokens = _interact(self.visual_interaction, encoded, valid)
        anchor = self.depth_prior_m.expand(b, n)
        if self.scale_enabled:
            anchor = anchor*.05/visual[..., 2].clamp_min(1e-5)
        initial = _bounded(anchor, self.coarse_head(tokens)[..., 0], bounds)*valid
        # Only measured public navigation enters this global memory. This is
        # not a zero spectrum, recovered observer echo, or simulated radar token.
        reference = self.observer_encoder(observer/scale[:, None])
        context, _ = self.global_attention(tokens, reference[:, None], reference[:, None],
                                           need_weights=False)
        global_token = self.global_norm(tokens+context)*valid[..., None]
        values = self.global_head(global_token)
        global_depth = _bounded(self.depth_prior_m.expand(b, n), values[..., 0], bounds)*valid
        global_gate = torch.sigmoid(values[..., 1])*valid
        depth = initial
        fractions = []; gates = []; stage_offsets = []; stage_limits = []; stages = []
        queries = []; bases = []; hypotheses_trace = []; geometries = []
        for step in range(self.steps):
            fine = self.fine_enabled and step == 1
            offsets = self.depth_fine_offsets_m if fine else self.depth_offsets_m
            correction = self.fine_max_correction if fine else self.max_correction
            gate = torch.zeros_like(global_gate) if fine else global_gate
            base = (1.-gate)*depth+gate*global_depth
            query = depth
            hypotheses = query[..., None]+offsets[None, None]
            geometry = _geometry(origin, direction, hypotheses, bounds, valid)
            usable = geometry['valid']
            angular = torch.stack((geometry['azimuth'].sin(), geometry['azimuth'].cos(),
                                   geometry['elevation'].sin(), geometry['elevation'].cos()), -1)
            scalars = torch.cat((hypotheses[..., None]/scale[:, None, None, None],
                geometry['ranges'][..., None]/scale[:, None, None, None],
                (offsets/correction)[None, None, :, None].expand(b, n, -1, -1), angular,
                usable[..., None].to(origin.dtype)), -1)
            candidate = self.geometry_encoder(scalars)
            logits = self.sample_score(torch.cat((candidate, global_token[:, :, None].expand(
                -1, -1, len(offsets), -1)), -1))[..., 0]
            weights = torch.softmax(logits.masked_fill(~usable, -1e4), -1)*usable
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
            local = (candidate*weights[..., None]).sum(2)
            refine = self.refine_encoder(torch.cat((tokens, global_token, local,
                                                    depth[..., None]/scale[:, None, None]), -1))
            refine = _interact(self.refine_interaction, refine, valid)
            raw = torch.tanh(self.refine_head(refine)[..., 0])
            # Reuse the original update method so the unchanged stage observer
            # records both actual global-blend bases without altering tensors.
            depth = self._support_residual(base.reshape(-1), raw.reshape(-1),
                bounds.reshape(-1, 2), correction).reshape_as(base)*valid
            fractions.append(usable.to(origin.dtype).mean(-1)); gates.append(gate)
            stage_offsets.append(offsets); stage_limits.append(correction); stages.append(depth)
            if self.capture_candidates:
                queries.append(query); bases.append(base); hypotheses_trace.append(hypotheses)
                geometries.append(dict(offsets_m=offsets, depths_m=hypotheses,
                    ranges_m=geometry['ranges'], azimuth_rad=geometry['azimuth'],
                    elevation_rad=geometry['elevation'], geometry_valid=usable, usable=usable))
            tokens = refine
        positions = (origin+depth[..., None]*direction)*valid[..., None]
        diagnostics = dict(local_valid_fraction=fractions, global_gate=gates, global_depths=global_depth,
            candidate_offsets_m=self.depth_offsets_m, stage_candidate_offsets_m=stage_offsets,
            stage_max_correction_m=stage_limits, stage_depths_m=stages,
            refinement_mode='visual_navigation_geometry_control', global_proposal_is_search=False,
            geometry_enabled=True, independent_peak_assignment=False, mixed_observer_preserved=False,
            radar_used=False, study_arm=self.study_arm,
            candidate_evidence='public geometry only; no measured or zero-filled radar features',
            observer_reference='public measured navigation coordinates only',
            stage_sequence_residual_logits=[torch.zeros_like(initial) for _ in stages])
        if self.capture_candidates:
            diagnostics.update(stage_query_centers_m=queries, stage_update_bases_m=bases,
                stage_candidate_depths_m=hypotheses_trace, stage_candidate_geometry=geometries,
                sequence_details=[])
        return dict(depths=depth, initial_depths=initial, member_positions=positions,
            initial_positions=(origin+initial[..., None]*direction)*valid[..., None], valid_mask=valid,
            center=positions.sum(1)/valid.sum(1).clamp_min(1)[:, None],
            sampling_diagnostics=diagnostics, member_mask=x['member_mask'])

    def forward(self, features):
        if self.radar_used:
            return super().forward(features)
        self._validate_visual(features)
        batch = self.pack_batch([features]); value = self.forward_batch(batch); n = batch.lengths[0]
        output = {key: value[key][0, :n] for key in
                  ('depths', 'initial_depths', 'member_positions', 'initial_positions', 'valid_mask')}
        output['center'] = value['center'][0]
        lists = {'local_valid_fraction', 'global_gate', 'stage_depths_m',
                 'stage_sequence_residual_logits', 'stage_query_centers_m',
                 'stage_update_bases_m', 'stage_candidate_depths_m'}
        diagnostic = {}
        for key, item in value['sampling_diagnostics'].items():
            if key == 'global_depths':
                diagnostic[key] = item[0, :n]
            elif key in lists:
                diagnostic[key] = [v[0, :n] for v in item]
            elif key == 'stage_candidate_geometry':
                diagnostic[key] = [{name: (v if name == 'offsets_m' else v[0, :n])
                                    for name, v in stage.items()} for stage in item]
            else:
                diagnostic[key] = item
        output['sampling_diagnostics'] = diagnostic
        return output

    def predict(self, features, allow_untrained_validation=False):
        value = super().predict(features, allow_untrained_validation=allow_untrained_validation)
        if not self.radar_used:
            value.update(source='trained_visual_roi_geometry' if self.trained else 'untrained_visual_roi_diagnostic',
                         decoder='two-round continuous visual/navigation depth; no radar observations',
                         radar_used=False, study_arm=self.study_arm)
        return value

    def predict_batch(self, value, allow_untrained_validation=False):
        scenes = super().predict_batch(value, allow_untrained_validation=allow_untrained_validation)
        if not self.radar_used:
            for scene in scenes:
                scene.update(source='trained_visual_roi_geometry' if self.trained else 'untrained_visual_roi_diagnostic',
                             decoder='two-round continuous visual/navigation depth; no radar observations',
                             radar_used=False, study_arm=self.study_arm)
        return scenes


DepthLocalizer = StudyDepthLocalizer
FastDepthLocalizer = StudyDepthLocalizer
