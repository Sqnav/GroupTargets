"""Single-frame v3 backbone with an isolated native three-dimensional query adapter.

The baseline module stays byte-for-byte unchanged. The copied forward below
retains its global RA/RE encoders, depth bounds, losses and two-pass budgets;
only local evidence reads the joint pre-projection power at (R,u,v).
"""
from __future__ import annotations
import numpy as np
import torch
from depth_localization.model import DepthLocalizer as BaseDepthLocalizer
from depth_localization.model import _mlp, _interact, _mean
from depth_localization.geometry import candidate_geometry
from .volume import positions_to_ruv, sample_power_volume, VERSION


class JointPowerDepthLocalizer(BaseDepthLocalizer):
    def __init__(self, config):
        super().__init__(config)
        h = int(config.get('depth_hidden_dim', config.get('group_hidden_dim', 64)))
        self.joint_neighbourhood = config.get('precision_joint_neighbourhood', 'center')
        if self.joint_neighbourhood not in ('center','cross7'):
            raise ValueError('Joint neighbourhood must be center or cross7')
        # The same small adapter is used for both controls. Unqueried slots
        # are zero with mask=False, rather than a fabricated measurement.
        self.volume_adapter = _mlp(14, h//2, 2*h)
        torch.nn.init.normal_(self.volume_adapter[-1].weight, std=.02)
        torch.nn.init.zeros_(self.volume_adapter[-1].bias)

    def _validate(self, features):
        base = {k:v for k,v in features.items() if k != 'joint_volume'}
        super()._validate(base)
        if 'joint_volume' not in features:
            raise ValueError('Joint radar mode requires an authenticated measured power volume')
        value = features['joint_volume']
        if set(value) != {'volume_power','range_m','u_axis','v_axis','angular_valid','metadata'}:
            raise ValueError('Only calibrated power/axes/mask/provenance may enter joint volume')
        if value['metadata'].get('version') != VERSION or value['metadata'].get('uses_labels') is not False:
            raise ValueError('Joint volume must declare its measured, label-free input contract')
        a = value['volume_power']
        shape = tuple(a.shape)
        if shape != (len(value['range_m']),len(value['u_axis']),len(value['v_axis'])):
            raise ValueError('Joint volume and axes differ')
        if any(n < 2 for n in shape):
            raise ValueError('Joint axes need at least two bins')

    def _volume(self, features):
        value = features['joint_volume']
        raw = value['volume_power']
        # Cache arrays are deliberately read-only. Give torch writable private
        # storage rather than exposing a non-writable mmap as a mutable tensor.
        if isinstance(raw, np.ndarray) and not raw.flags.writeable:
            raw = np.array(raw,copy=True)
        volume = self._tensor(raw)
        if not bool(torch.isfinite(volume).all()) or bool((volume < 0).any()):
            raise ValueError('Joint input must be finite nonnegative measured power')
        return volume, value

    def _joint_query(self, volume, meta, geometry, power_scale):
        ruv = positions_to_ruv(geometry['positions'])
        valid = geometry['valid'] & ruv['valid']
        axes = [self._tensor(meta[k]) for k in ('range_m','u_axis','v_axis')]
        steps = [a[1]-a[0] for a in axes]
        zero = steps[0]*0.
        offsets = torch.stack((torch.stack((zero,zero,zero)),
            torch.stack((-steps[0],zero,zero)),torch.stack((steps[0],zero,zero)),
            torch.stack((zero,-steps[1],zero)),torch.stack((zero,steps[1],zero)),
            torch.stack((zero,zero,-steps[2])),torch.stack((zero,zero,steps[2]))))
        count = 1 if self.joint_neighbourhood == 'center' else 7
        offsets = offsets[:count]
        queried = sample_power_volume(volume,
            ranges=ruv['ranges'][...,None]+offsets[:,0],
            u=ruv['u'][...,None]+offsets[:,1],v=ruv['v'][...,None]+offsets[:,2],
            range_axis=axes[0],u_axis=axes[1],v_axis=axes[2],
            angular_valid=meta['angular_valid'],
            candidate_valid=valid[...,None].expand(*valid.shape,count))
        powers = queried['values'][...,0]; mask=queried['valid']
        logs = torch.log1p(powers.clamp_min(0.)/power_scale)
        if count == 1:
            logs = torch.nn.functional.pad(logs,(0,6))
            mask_input = torch.nn.functional.pad(mask.to(logs.dtype),(0,6))
        else:
            mask_input = mask.to(logs.dtype)
        embedded=self.volume_adapter(torch.cat((logs,mask_input),-1))*mask[...,0,None]
        return embedded, mask[...,0], powers[...,0], dict(powers=powers,valid=mask)

    def forward(self, features):
        self._validate(features)
        volume, volume_meta = self._volume(features)
        joint_powers = []; joint_masks = []
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
            # Keep RA/RE global memory unchanged, but use the paired native
            # R/u/v evidence instead of max-projected local feature pairs.
            embedded, joint_valid, joint_power, neighbourhood = self._joint_query(
                volume, volume_meta, geometry, power_scale)
            half = embedded.shape[-1]//2
            sampled = dict(ra=embedded[..., :half], re=embedded[..., half:],
                ra_valid=joint_valid, re_valid=joint_valid, valid=joint_valid)
            joint_powers.append(joint_power); joint_masks.append(joint_valid)
            usable = joint_valid if self.geometry_enabled else torch.zeros_like(joint_valid)
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
                joint_power_per_stage=joint_powers, joint_valid_per_stage=joint_masks,
                joint_neighbourhood=self.joint_neighbourhood,
                local_evidence="native_joint_Ruv_power",
                power_scale=power_scale,
                mixed_observer_preserved=True, independent_peak_assignment=False))


    def predict(self, features, allow_untrained_validation=False):
        result = super().predict(features, allow_untrained_validation)
        result.update(source='trained_joint_power_depth_localizer' if self.trained else 'untrained_joint_interface',
            radar_mode='joint_volume', decoder='continuous depths from global RA/RE and local native R/u/v power')
        return result


def make_model(config):
    mode = config.get('precision_radar_mode', 'ra_re')
    if mode == 'ra_re':
        return BaseDepthLocalizer(config)
    if mode == 'joint_volume':
        return JointPowerDepthLocalizer(config)
    if mode == 'none':
        from .visual import VisualOnlyDepthLocalizer
        return VisualOnlyDepthLocalizer(config)
    raise ValueError('precision_radar_mode must be ra_re, joint_volume or none')


DepthLocalizer = make_model
