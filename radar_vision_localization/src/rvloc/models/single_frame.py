"""Visual mean/uncertainty -> one radar query -> one final depth output.

Visual priors use only current ROI/navigation geometry. Radar CNN maps are
encoded once and reused by full-map attention and one candidate query. There
is no intermediate corrected depth, re-query, or iterative refinement.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from rvloc.legacy import activate

activate()
from depth_localization.model import (DepthLocalizer, _NativeSpectrumEncoder,
                                      _mlp, _interaction, _interact, _mean)
from precision_study.visual import ALLOWED_FIELDS, SCHEMA, validate_visual_features

ARCHITECTURE = "single_query_depth_v3"
DEFAULT_MULTIPLIERS = (-3., -2., -1., -.5, 0., .5, 1., 2., 3.)
SEQUENCE_FIELDS = frozenset({"episode_id", "member_ids", "frame_index", "timestamp_s"})
FORBIDDEN_FIELDS = frozenset({
    "iq", "raw_iq", "risk_score", "acceptance", "accept", "search_features",
    "positions", "target_positions", "true_positions", "true_depths", "ground_truth",
    "gt", "formation_index", "formation_type", "depth_labels", "target_velocities",
    "true_count", "targets", "future", "future_frames", "labels", "member_velocities",
})
CURRENT_FIELDS = ALLOWED_FIELDS | frozenset({"detector_roi_features", "ra_power",
    "re_power", "ra_valid", "re_valid", "range_m", "azimuth_rad", "elevation_rad"})
ITERATIVE_SETTINGS = frozenset({"depth_offsets_m", "depth_fine_offsets_m",
    "depth_max_correction_m", "depth_fine_max_correction_m", "depth_geometry_enabled",
    "d_evidence_capture_candidates", "depth_initial_loss_weight"})


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def model_config(config=None, *, visual=False):
    """Only zero iterative updates and one uncertainty-scaled radar query."""
    value = dict(config or {})
    if ITERATIVE_SETTINGS.intersection(value):
        raise ValueError("Single-query depth forbids iterative refinement settings: "
                         + str(sorted(ITERATIVE_SETTINGS.intersection(value))))
    for name in ("d_evidence_candidate_sequence", "d_evidence_aligned_first_query"):
        if name in value and value[name] is not False:
            raise ValueError("Single-query depth forbids legacy iterative setting: " + name)
    if type(value.get("depth_refinement_steps", 0)) is not int or value.get("depth_refinement_steps", 0) != 0:
        raise ValueError("Single-query depth requires depth_refinement_steps=0")
    for name in ("architecture", "architecture_schema"):
        if name in value and value[name] != ARCHITECTURE:
            raise ValueError("Expected single_query_depth_v3 architecture")
    expected = 0 if visual else 1
    if "query_rounds" in value and (type(value['query_rounds']) is not int or value['query_rounds'] != expected):
        raise ValueError(f"This model group requires query_rounds={expected}")
    if "local_query_enabled" in value and value['local_query_enabled'] is not (not visual):
        raise ValueError("local_query_enabled is inconsistent with the model group")
    value.update(architecture=ARCHITECTURE,architecture_schema=ARCHITECTURE,
                 depth_refinement_steps=0,query_rounds=expected,local_query_enabled=not visual,
                 precision_radar_mode="none" if visual else "ra_re")
    value.setdefault("depth_hidden_dim",64)
    value.setdefault("depth_attention_heads",4)
    value.setdefault("depth_interaction_layers",2)
    value.setdefault("candidate_multipliers",list(DEFAULT_MULTIPLIERS))
    value.setdefault("visual_std_min_m",.25)
    value.setdefault("visual_std_max_m",16.)
    value.setdefault("visual_std_initial_m",2.)
    z=value['candidate_multipliers']
    if (not isinstance(z,(list,tuple)) or not z or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in z)
            or len(set(z))!=len(z) or 0. not in z or list(z)!=sorted(z)):
        raise ValueError("Candidate multipliers must be finite, unique, sorted and include zero")
    value['candidate_multipliers']=[float(x) for x in z]
    for name in ('visual_std_min_m','visual_std_max_m','visual_std_initial_m'):
        item=value[name]
        if isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(item) or item<=0:
            raise ValueError(name+' must be positive finite metres')
    if not value['visual_std_min_m']<value['visual_std_initial_m']<value['visual_std_max_m']:
        raise ValueError("Initial visual standard deviation must lie inside its metre bounds")
    return value


def current_features(features: Mapping, *, visual=False):
    """Use only public current inputs; RGB never accesses radar values."""
    if not isinstance(features,Mapping):
        raise TypeError("features must be a mapping")
    forbidden=FORBIDDEN_FIELDS.intersection(features)
    if forbidden:
        raise ValueError("Inference forbids label, IQ, risk or search inputs: "+str(sorted(forbidden)))
    unknown=features.keys()-CURRENT_FIELDS-SEQUENCE_FIELDS
    if unknown:
        raise ValueError("Unknown inference features: "+str(sorted(unknown)))
    if visual:
        selected={key:features[key] for key in ALLOWED_FIELDS if key!='schema'}
        selected['schema']=SCHEMA
        selected['detector_roi_features']=features['detector_roi_features']
        return selected
    return {key:value for key,value in features.items() if key not in SEQUENCE_FIELDS}


class _SingleQueryDepthCore(nn.Module):
    architecture=ARCHITECTURE

    def __init__(self,config,*,visual):
        super().__init__()
        self.config=dict(config);self.visual=bool(visual)
        h=_integer(config['depth_hidden_dim'],'depth_hidden_dim',8)
        heads=_integer(config['depth_attention_heads'],'depth_attention_heads')
        layers=_integer(config['depth_interaction_layers'],'depth_interaction_layers')
        if h%heads:raise ValueError('depth_hidden_dim must be divisible by depth_attention_heads')
        self.hidden_size=self.hidden_dim=h
        self.max_members=_integer(config.get('group_max_members',24),'group_max_members')
        self.scale_enabled=config.get('depth_visual_scale_enabled',True)
        if type(self.scale_enabled) is not bool:raise ValueError('depth_visual_scale_enabled must be boolean')
        self.visual_encoder=_mlp(18,h,h)
        self.roi_projection=nn.Linear(1024,h,bias=False)
        self.visual_interaction=_interaction(h,heads,layers)
        self.visual_prior_head=_mlp(h,h,2)
        fraction=(config['visual_std_initial_m']-config['visual_std_min_m'])/(config['visual_std_max_m']-config['visual_std_min_m'])
        with torch.no_grad():self.visual_prior_head[-1].bias.copy_(torch.tensor([0.,math.log(fraction/(1.-fraction))]))
        self.std_min_m=float(config['visual_std_min_m']);self.std_max_m=float(config['visual_std_max_m'])
        if not self.visual:
            from .single_query import SingleRadarQuery
            self.ra_encoder=_NativeSpectrumEncoder(h)
            self.re_encoder=_NativeSpectrumEncoder(h)
            self.observer_encoder=_mlp(3,h,h)
            self.global_attention=nn.MultiheadAttention(h,heads,dropout=0.,batch_first=True)
            self.global_norm=nn.LayerNorm(h)
            self.local_query=SingleRadarQuery(hidden=h,heads=heads,candidate_multipliers=config['candidate_multipliers'])
            self.fusion_encoder=_mlp(3*h+2,h,h)
            self.fusion_interaction=_interaction(h,heads,layers)
            self.depth_head=_mlp(h,h,1)
            nn.init.zeros_(self.depth_head[-1].bias)
        self.register_buffer('depth_prior_m',torch.tensor(20.))
        self.trained=False

    def initialize_depth_prior(self,depth_m):
        depth_m=float(depth_m)
        if not math.isfinite(depth_m) or depth_m<=0:raise ValueError('Positive train-only depth prior required')
        with torch.no_grad():self.depth_prior_m.fill_(depth_m)

    def _tensor(self,value):
        tensor=torch.as_tensor(value,device=self.depth_prior_m.device)
        if tensor.is_complex():raise ValueError('Public depth features must be real')
        return tensor.to(dtype=self.depth_prior_m.dtype)

    def _validate(self,features):
        from rvloc.data import DIRECT_DEPTH_SCHEMA
        if features.get('depth_schema')!=DIRECT_DEPTH_SCHEMA:
            raise ValueError('Expected the public optical-depth geometry contract')
        if self.visual:
            view={key:features[key] for key in ALLOWED_FIELDS if key!='schema'};view['schema']=SCHEMA
            validate_visual_features(view)
        else:
            from rvloc.data import validate_direct_fusion_features
            validate_direct_fusion_features(features)
        n=len(features['ray_directions'])
        if n>self.max_members:raise ValueError('Detected count exceeds configured model support')
        roi=self._tensor(features['detector_roi_features'])
        if roi.shape!=(n,1024) or not bool(torch.isfinite(roi).all()):
            raise ValueError('Finite frozen ROI vectors must align with detections')

    def forward(self,features,*,fusion_prior=None):
        self._validate(features)
        origin=self._tensor(features['ray_origins']);direction=self._tensor(features['ray_directions'])
        bounds=self._tensor(features['ray_depth_bounds'])
        valid=torch.as_tensor(features['ray_valid'],device=origin.device,dtype=torch.bool)
        observer=self._tensor(features['observer_position']);visual=self._tensor(features['visual_scale'])
        scale=float(features['range_scale_m']);n=len(origin)
        if not self.scale_enabled:visual=torch.zeros_like(visual)
        geometry=torch.cat((origin/scale,direction,(origin-observer[None])/scale,bounds/scale,visual),-1)
        encoded=self.visual_encoder(geometry)+self.roi_projection(self._tensor(features['detector_roi_features']).detach())
        visual_token=_interact(self.visual_interaction,encoded,valid)
        prior_values=self.visual_prior_head(visual_token)
        anchor=self.depth_prior_m.expand(n)
        if self.scale_enabled:anchor=anchor*.05/visual[:,2].clamp_min(1e-5)
        # Both parameters depend on current visual inputs only. Their metric
        # coordinates are never conditioned on radar or world memory.
        mean=DepthLocalizer._bounded(anchor,prior_values[:,0],bounds)*valid
        std=self.std_min_m+(self.std_max_m-self.std_min_m)*torch.sigmoid(prior_values[:,1])
        diagnostics=dict(refinement_mode='single_query_depth',refinement_steps=0,
                         query_rounds=0,query_rounds_configured=self.config['query_rounds'],
                         local_query_enabled=not self.visual,stage_depths_m=[])
        if fusion_prior is not None:
            if self.visual:raise ValueError('RGB visual prior cannot receive world/radar fusion state')
            if fusion_prior.shape!=(n,self.hidden_size) or not bool(torch.isfinite(fusion_prior).all()):
                raise ValueError('fusion_prior must be a finite detection-aligned fusion token')
        if self.visual:
            depth=mean
        else:
            global_token=visual_token
            local_token=torch.zeros_like(visual_token)
            if n and bool(valid.any()):
                ra,re=self._tensor(features['ra_power']),self._tensor(features['re_power'])
                av=torch.as_tensor(features['ra_valid'],device=origin.device,dtype=torch.bool)
                ev=torch.as_tensor(features['re_valid'],device=origin.device,dtype=torch.bool)
                ranges=self._tensor(features['range_m']);az=self._tensor(features['azimuth_rad']);el=self._tensor(features['elevation_rad'])
                power_scale=((ra.masked_fill(~av,0.).sum()+re.masked_fill(~ev,0.).sum())/(av.sum()+ev.sum()).clamp_min(1)).clamp_min(1e-20)
                ra_map,ra_rows,ra_rows_valid=self.ra_encoder(ra,av,ranges,az,scale,power_scale)
                re_map,re_rows,re_rows_valid=self.re_encoder(re,ev,ranges,el,scale,power_scale)
                reference=self.observer_encoder(observer/scale)
                memory=torch.cat((ra_rows,re_rows,reference[None]),0)
                memory_valid=torch.cat((ra_rows_valid,re_rows_valid,valid.new_ones(1)))
                context,_=self.global_attention(visual_token[None],memory[None],memory[None],key_padding_mask=~memory_valid[None],need_weights=False)
                global_token=self.global_norm(visual_token+context[0])*valid[:,None]
                local_token,query=self.local_query(origin=origin,direction=direction,bounds=bounds,ray_valid=valid,
                    visual_mean=mean,visual_std=std,ra_map=ra_map,re_map=re_map,range_axis=ranges,
                    azimuth_axis=az,elevation_axis=el,ra_valid=av,re_valid=ev,range_scale_m=scale)
                diagnostics.update(query)
                for src,dst in [('candidate_depths','candidate_depths_m'),('candidate_offsets','candidate_offsets_m')]:
                    if src in diagnostics and dst not in diagnostics:diagnostics[dst]=diagnostics[src]
                diagnostics['query_rounds']=1
            fused=self.fusion_encoder(torch.cat((visual_token,global_token,local_token,mean[:,None]/scale,std[:,None]/scale),-1))
            if fusion_prior is not None:fused=fused+fusion_prior.to(device=fused.device,dtype=fused.dtype)
            fused=_interact(self.fusion_interaction,fused,valid)
            depth=DepthLocalizer._bounded(mean,self.depth_head(fused)[:,0],bounds)*valid
        positions=(origin+depth[:,None]*direction)*valid[:,None]
        return dict(depths=depth,visual_depth_mean_m=mean,visual_depth_std_m=std,
                    member_positions=positions,valid_mask=valid,center=_mean(positions,valid),
                    sampling_diagnostics=diagnostics,
                    aux=dict(architecture=ARCHITECTURE,radar_used=not self.visual,
                             query_rounds=diagnostics['query_rounds'],depth_head_count=1,
                             final_depth_head_count=0 if self.visual else 1,visual_prior_head_count=1,
                             visual_prior_conditioning='current_roi_geometry_only'))


class SingleFrameLocalizer(nn.Module):
    """RGB prior or one-query global/local radar/vision depth fusion."""
    architecture=ARCHITECTURE

    def __init__(self,config=None,*,visual=False):
        super().__init__();self.visual=bool(visual)
        self.group='rgb' if self.visual else 'single_frame'
        self.config=model_config(config,visual=self.visual)
        self.core=_SingleQueryDepthCore(self.config,visual=self.visual)

    def forward(self,features,memory=None):
        output=self.core(current_features(features,visual=self.visual))
        output['state']=None;output['aux']['group']=self.group
        return output

    def initialize_depth_prior(self,depth_m):self.core.initialize_depth_prior(depth_m)


class RGBLocalizer(SingleFrameLocalizer):
    def __init__(self,config=None):super().__init__(config,visual=True)


class RadarVisionLocalizer(SingleFrameLocalizer):
    def __init__(self,config=None):super().__init__(config,visual=False)
