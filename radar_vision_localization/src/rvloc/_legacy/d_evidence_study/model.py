"""Three isolated single-frame D ablations; production D is never modified.

The sequence adapter reads the nine observations *inside* each original
refinement, before tanh and the original metre-bounded update. It never queries
a third set of candidates or post-processes D's final positions.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn

from depth_localization.model import DepthLocalizer as BaseDepthLocalizer
from depth_localization.model import SCHEMA as FRAME_SCHEMA, _interact, _mean
from depth_localization.geometry import candidate_geometry, sample_paired_spectra

SCHEMA = 'd_evidence_depth_v1'
ARMS = ('control','sequence','aligned','roi')
FLAGS = ('d_evidence_candidate_sequence','d_evidence_aligned_first_query','d_evidence_frozen_roi')
ARM_FLAG = dict(zip(ARMS[1:],FLAGS))


def configured_flags(config):
    arm = config.get('d_evidence_arm','control')
    if arm not in ARMS:
        raise ValueError('Unknown single-change D evidence arm')
    flags = {name:config.get(name,ARM_FLAG.get(arm)==name) for name in FLAGS}
    if any(type(value) is not bool for value in flags.values()) or sum(flags.values()) > 1:
        raise ValueError('D evidence flags must be booleans with at most one active change')
    expected = ARM_FLAG.get(arm)
    if any(value != (name==expected) for name,value in flags.items()):
        raise ValueError('The evidence arm and its single independent switch disagree')
    return arm, flags


def frame_features(features):
    if not isinstance(features,dict):
        raise TypeError('One measured D feature dictionary is required')
    return {key:value for key,value in features.items() if key!='detector_roi_features'}


class CandidateSequenceResidual(nn.Module):
    """Masked nonuniform-depth sequence -> an extra continuous residual logit."""
    def __init__(self, feature_dim=64, hidden=32, candidates=9):
        super().__init__()
        self.candidates=candidates
        self.first=nn.Conv1d(3*feature_dim+8,hidden,3,padding=1)
        self.second=nn.Conv1d(hidden,hidden,3,padding=1)
        self.activation=nn.SiLU()
        self.readout=nn.Linear(candidates*hidden,1,bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self,candidate,ra,re,*,offsets_m,depths_m,ranges_m,azimuth_rad,elevation_rad,
                ra_valid,re_valid,geometry_valid,range_scale_m):
        if candidate.ndim!=3 or candidate.shape[1]!=self.candidates or ra.shape!=candidate.shape or re.shape!=candidate.shape:
            raise ValueError('Complete matched nine-candidate feature sequences are required')
        shape=candidate.shape[:2]
        if (offsets_m.ndim!=1 or len(offsets_m)!=self.candidates or not bool(torch.isfinite(offsets_m).all())
                or len(torch.unique(offsets_m))!=self.candidates):
            raise ValueError('Actual finite unique metre offsets, rather than candidate indices, are required')
        for value in (depths_m,ranges_m,azimuth_rad,elevation_rad,ra_valid,re_valid,geometry_valid):
            if value.shape!=shape:
                raise ValueError('Candidate geometry and independent masks must align')
        if any(value.dtype!=torch.bool for value in (ra_valid,re_valid,geometry_valid)):
            raise ValueError('Separate boolean RA, RE and physical geometry masks are required')
        if not math.isfinite(float(range_scale_m)) or range_scale_m<=0:
            raise ValueError('A positive public radar range scale is required')
        usable=(ra_valid|re_valid)&geometry_valid
        # The offsets are deliberately nonuniform in D. Sort by actual metres
        # and retain those numbers, instead of treating the original [0,-,+]
        # storage order as a uniformly spaced depth lattice.
        order=torch.argsort(offsets_m,stable=True)
        scalars=torch.stack((offsets_m[None].expand(shape)/16.,depths_m/range_scale_m,
            ranges_m/range_scale_m,azimuth_rad,elevation_rad,ra_valid.to(candidate.dtype),
            re_valid.to(candidate.dtype),geometry_valid.to(candidate.dtype)),-1)
        safe_ra=torch.where(ra_valid[...,None],ra,torch.zeros_like(ra))
        safe_re=torch.where(re_valid[...,None],re,torch.zeros_like(re))
        values=torch.cat((candidate,safe_ra,safe_re,scalars),-1)
        values=torch.where(usable[...,None],values,torch.zeros_like(values))
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Valid candidate evidence must be finite')
        mask=usable[:,order,None].transpose(1,2).to(values.dtype)
        encoded=values[:,order].transpose(1,2)
        encoded=self.activation(self.first(encoded))*mask
        encoded=self.activation(self.second(encoded))*mask
        residual=self.readout(encoded.flatten(1))[:,0]*usable.any(-1).to(values.dtype)
        return residual,dict(sorted_offsets_m=offsets_m[order],sort_indices=order,
            sorted_candidate_depths_m=depths_m[:,order],sorted_ranges_m=ranges_m[:,order],
            sorted_azimuth_rad=azimuth_rad[:,order],sorted_elevation_rad=elevation_rad[:,order],
            sorted_ra_valid=ra_valid[:,order],sorted_re_valid=re_valid[:,order],
            sorted_geometry_valid=geometry_valid[:,order],usable_fraction=usable.to(values.dtype).mean(-1))


class EvidenceDepthLocalizer(BaseDepthLocalizer):
    def __init__(self,config):
        config=deepcopy(config)
        if config.get('architecture_schema',FRAME_SCHEMA)!=FRAME_SCHEMA or config.get('d_evidence_schema',SCHEMA)!=SCHEMA:
            raise ValueError('D per-frame geometry and independent evidence schema are required')
        arm,flags=configured_flags(config)
        config.setdefault('architecture_schema',FRAME_SCHEMA)
        config.update(d_evidence_schema=SCHEMA,d_evidence_arm=arm,**flags)
        super().__init__(config)
        self.schema=SCHEMA
        self.evidence_arm=arm
        self.sequence_enabled=flags[FLAGS[0]]
        self.aligned_enabled=flags[FLAGS[1]]
        self.roi_enabled=flags[FLAGS[2]]
        self.capture_candidates=config.get('d_evidence_capture_candidates',False)
        if type(self.capture_candidates) is not bool:
            raise ValueError('Candidate capture must be an explicit diagnostic boolean')
        self.roi_dim=int(config.get('d_evidence_roi_dim',1024))
        seed=config.get('d_evidence_init_seed',20261124)
        if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
            raise ValueError('An independent nonnegative evidence initialization seed is required')
        if self.roi_dim!=1024:
            raise ValueError('This study fixes the frozen detector box-head representation to 1024 dimensions')
        if any(flags.values()):
            if (int(config.get('depth_hidden_dim',64))!=64 or int(config.get('depth_attention_heads',4))!=4
                    or self.steps!=2 or not self.fine_enabled or len(self.depth_offsets_m)!=9
                    or len(self.depth_fine_offsets_m)!=9 or self.max_correction!=16. or self.fine_max_correction!=16.):
                raise ValueError('Keep D width/heads, original two nine-point rounds and 16m correction limits')
        # Additional initialization must not perturb either original D weights
        # or the random stream consumed after constructing the baseline model.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if self.sequence_enabled:
                hidden=int(config.get('d_evidence_sequence_hidden_dim',32))
                if hidden!=32:
                    raise ValueError('The registered candidate Conv1d width is 32')
                self.candidate_sequence=CandidateSequenceResidual(64,hidden,9)
            if self.roi_enabled:
                self.roi_projection=nn.Linear(self.roi_dim,64,bias=False)
                nn.init.zeros_(self.roi_projection.weight)

    def _roi(self,features,n):
        if 'detector_roi_features' not in features:
            raise ValueError('The ROI arm requires authenticated frozen detector ROI features')
        value=self._tensor(features['detector_roi_features']).detach()
        if value.shape!=(n,self.roi_dim) or not bool(torch.isfinite(value).all()):
            raise ValueError('One finite 1024-dimensional frozen ROI vector per current detection is required')
        return value

    def forward(self,features):
        frame=frame_features(features)
        # A true flags-off route executes the original forward, not a copied
        # implementation, and has exactly the original state-dict keys.
        if not (self.sequence_enabled or self.aligned_enabled or self.roi_enabled):
            return super().forward(frame)
        self._validate(frame)
        origin=self._tensor(frame['ray_origins']); direction=self._tensor(frame['ray_directions'])
        bounds=self._tensor(frame['ray_depth_bounds']); valid=self._bool(frame['ray_valid'])
        observer=self._tensor(frame['observer_position']); visual=self._tensor(frame['visual_scale'])
        scale=float(frame['range_scale_m']); n=len(origin)
        if not self.scale_enabled:
            visual=torch.zeros_like(visual)
        visual_input=torch.cat((origin/scale,direction,(origin-observer[None])/scale,bounds/scale,visual),-1)
        encoded=self.visual_encoder(visual_input)
        if self.roi_enabled:
            encoded=encoded+self.roi_projection(self._roi(features,n))
        tokens=_interact(self.visual_interaction,encoded,valid)
        anchor=self.depth_prior_m.expand(n)
        if self.scale_enabled:
            anchor=anchor*.05/visual[:,2].clamp_min(1e-5)
        initial=self._bounded(anchor,self.coarse_head(tokens)[:,0],bounds)*valid
        if not n or not bool(valid.any()):
            zero=origin*0.
            return dict(depths=initial,initial_depths=initial,member_positions=zero,initial_positions=zero,
                valid_mask=valid,center=origin.new_zeros(3),sampling_diagnostics=dict(local_valid_fraction=[],
                global_gate=[],stage_candidate_offsets_m=[],stage_max_correction_m=[],stage_depths_m=[],
                refinement_mode='coarse_fine',geometry_enabled=self.geometry_enabled,
                global_proposal_is_search=False,evidence_arm=self.evidence_arm))
        ra,re=self._tensor(frame['ra_power']),self._tensor(frame['re_power'])
        av,ev=self._bool(frame['ra_valid']),self._bool(frame['re_valid'])
        ranges=self._tensor(frame['range_m']); az=self._tensor(frame['azimuth_rad']); el=self._tensor(frame['elevation_rad'])
        power_scale=((ra.masked_fill(~av,0.).sum()+re.masked_fill(~ev,0.).sum())/
            (av.sum()+ev.sum()).clamp_min(1)).clamp_min(1e-20)
        ra_map,ra_rows,ra_rows_valid=self.ra_encoder(ra,av,ranges,az,scale,power_scale)
        re_map,re_rows,re_rows_valid=self.re_encoder(re,ev,ranges,el,scale,power_scale)
        reference=self.observer_encoder(observer/scale)
        memory=torch.cat((ra_rows,re_rows,reference[None]),0)
        memory_valid=torch.cat((ra_rows_valid,re_rows_valid,valid.new_ones(1)))
        context,_=self.global_attention(tokens[None],memory[None],memory[None],
            key_padding_mask=~memory_valid[None],need_weights=False)
        global_token=self.global_norm(tokens+context[0])*valid[:,None]
        global_values=self.global_head(global_token)
        global_depth=self._bounded(self.depth_prior_m.expand(n),global_values[:,0],bounds)*valid
        global_gate=torch.sigmoid(global_values[:,1])*valid
        depth=initial
        fractions=[];gates=[];stage_offsets=[];stage_limits=[];stage_depths=[]
        query_centers=[];update_bases=[];candidate_depths=[];sequence_logits=[];sequence_details=[];candidate_geometry_trace=[]
        for step in range(self.steps):
            fine_stage=self.fine_enabled and step==1
            offsets=self.depth_fine_offsets_m if fine_stage else self.depth_offsets_m
            correction=self.fine_max_correction if fine_stage else self.max_correction
            applied_gate=torch.zeros_like(global_gate) if fine_stage else global_gate
            base=(1.-applied_gate)*depth+applied_gate*global_depth
            query_center=base if self.aligned_enabled and step==0 else depth
            hypotheses=query_center[:,None]+offsets[None]
            geometry=candidate_geometry(origin,direction,hypotheses,bounds,valid)
            sampled=sample_paired_spectra(ra_map,re_map,ranges=geometry['ranges'],
                azimuth=geometry['azimuth'],elevation=geometry['elevation'],range_axis=ranges,
                azimuth_axis=az,elevation_axis=el,ra_valid=av,re_valid=ev,candidate_valid=geometry['valid'])
            usable=sampled['valid'] if self.geometry_enabled else torch.zeros_like(sampled['valid'])
            if self.capture_candidates:
                candidate_geometry_trace.append(dict(offsets_m=offsets,depths_m=hypotheses,
                    ranges_m=geometry['ranges'],azimuth_rad=geometry['azimuth'],elevation_rad=geometry['elevation'],
                    ra_valid=sampled['ra_valid'],re_valid=sampled['re_valid'],
                    geometry_valid=geometry['valid'],usable=usable))
            angular=torch.stack((geometry['azimuth'].sin(),geometry['azimuth'].cos(),
                geometry['elevation'].sin(),geometry['elevation'].cos()),-1)
            scalars=torch.cat((hypotheses[...,None]/scale,geometry['ranges'][...,None]/scale,
                (offsets/correction)[None,:,None].expand(n,-1,-1),angular,
                sampled['ra_valid'][...,None].to(ra.dtype),sampled['re_valid'][...,None].to(ra.dtype),
                geometry['valid'][...,None].to(ra.dtype)),-1)
            candidate=self.sample_encoder(torch.cat((sampled['ra'],sampled['re'],scalars),-1))
            logits=self.sample_score(torch.cat((candidate,global_token[:,None].expand(-1,len(offsets),-1)),-1))[...,0]
            weights=torch.softmax(logits.masked_fill(~usable,-1e4),-1)*usable
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
            local=(candidate*weights[...,None]).sum(1)
            refine=self.refine_encoder(torch.cat((tokens,global_token,local,depth[:,None]/scale),-1))
            refine=_interact(self.refine_interaction,refine,valid)
            raw_logit=self.refine_head(refine)[:,0]
            sequence_logit=torch.zeros_like(raw_logit)
            if self.sequence_enabled:
                sequence_logit,detail=self.candidate_sequence(candidate,sampled['ra'],sampled['re'],
                    offsets_m=offsets,depths_m=hypotheses,ranges_m=geometry['ranges'],
                    azimuth_rad=geometry['azimuth'],elevation_rad=geometry['elevation'],
                    ra_valid=sampled['ra_valid'] & usable,re_valid=sampled['re_valid'] & usable,
                    geometry_valid=geometry['valid'] & usable,range_scale_m=scale)
                if self.capture_candidates:
                    sequence_details.append(detail)
            raw=torch.tanh(raw_logit+sequence_logit)
            depth=self._support_residual(base,raw,bounds,correction)*valid
            fractions.append(usable.to(ra.dtype).mean(-1));gates.append(applied_gate)
            stage_offsets.append(offsets);stage_limits.append(correction);stage_depths.append(depth)
            query_centers.append(query_center);update_bases.append(base);candidate_depths.append(hypotheses)
            sequence_logits.append(sequence_logit);tokens=refine
        positions=(origin+depth[:,None]*direction)*valid[:,None]
        result=dict(depths=depth,initial_depths=initial,member_positions=positions,
            initial_positions=(origin+initial[:,None]*direction)*valid[:,None],valid_mask=valid,
            center=_mean(positions,valid),sampling_diagnostics=dict(local_valid_fraction=fractions,
                global_gate=gates,global_depths=global_depth,geometry_enabled=self.geometry_enabled,
                candidate_offsets_m=self.depth_offsets_m,global_proposal_is_search=False,
                stage_candidate_offsets_m=stage_offsets,stage_max_correction_m=stage_limits,
                stage_depths_m=stage_depths,refinement_mode='coarse_fine',mixed_observer_preserved=True,
                independent_peak_assignment=False,evidence_arm=self.evidence_arm,
                stage_sequence_residual_logits=sequence_logits))
        if self.capture_candidates:
            result['sampling_diagnostics'].update(stage_query_centers_m=query_centers,
                stage_update_bases_m=update_bases,stage_candidate_depths_m=candidate_depths,
                stage_candidate_geometry=candidate_geometry_trace,sequence_details=sequence_details)
        return result

    def predict(self,features,allow_untrained_validation=False):
        result=super().predict(features,allow_untrained_validation)
        result.update(source='trained_d_evidence_depth' if self.trained else 'untrained_d_evidence_validation',
            d_evidence_schema=SCHEMA,d_evidence_arm=self.evidence_arm,
            d_evidence_flags={name:self.config[name] for name in FLAGS},input_frames=1,
            detector_frozen=True,roi_features_used=self.roi_enabled,
            decoder='original two-round D continuous camera-ray depth with one isolated internal evidence change')
        return result


DepthLocalizer=EvidenceDepthLocalizer
