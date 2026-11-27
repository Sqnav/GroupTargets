"""One-query shared-evidence graph branch using the original G3 backbone."""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from torch import nn

from rvloc.models import build_model
from .graph import SharedEvidenceGraph,VARIANTS

# Reuse the tested batched extension without editing the running source/tools.
_TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(_TOOLS) not in sys.path:sys.path.insert(0,str(_TOOLS))
import accelerated_batch_model as batch_engine

ARCHITECTURE='vision_conditioned_shared_radar_gnn_v1'
CHECKPOINT_SCHEMA='rvloc.shared_evidence_gnn.v1'
pack_features=batch_engine.pack_features


class SharedRadarEvidenceLocalizer(nn.Module):
    graph_architecture=ARCHITECTURE
    def __init__(self,config=None,*,variant='shared',rf=None):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unknown graph variant')
        # Build every original G3 module first, preserving shared-backbone RNG
        # initialisation. Only then replace its local evidence reader.
        original=build_model(3,config or {})
        self.core=original.core;self.visual=False;self.group='shared_evidence_gnn'
        self.config=dict(original.config,graph_variant=variant,graph_rf=dict(rf or {}),
                         graph_architecture=ARCHITECTURE)
        self.core.local_query=SharedEvidenceGraph(hidden=self.core.hidden_size,
            candidate_multipliers=self.config['candidate_multipliers'],rf=rf,variant=variant)

    def forward(self,features,memory=None):
        if memory is not None:raise ValueError('Single-frame graph does not accept temporal state')
        dtype=self.core.depth_prior_m.dtype;device=self.core.depth_prior_m.device
        packed=pack_features([features],device=device,visual=False,dtype=dtype)
        result=forward_batch(self,packed);n=len(features['ray_directions'])
        output={name:result[name][0,:n] for name in ('depths','visual_depth_mean_m','visual_depth_std_m','member_positions','valid_mask')}
        output.update(center=result['center'][0],state=None,aux=result['aux'],sampling_diagnostics=result['sampling_diagnostics'])
        return output


def build_graph_model(config=None,variant='shared',rf=None):
    return SharedRadarEvidenceLocalizer(config,variant=variant,rf=rf)


def forward_batch(model,batch):
    core=model.core
    if batch.get('schema')!=batch_engine.SCHEMA or batch['visual']:
        raise ValueError('Graph requires validated padded visual+RA/RE inputs')
    origin,direction,bounds,valid=(batch[name] for name in ('ray_origins','ray_directions','ray_depth_bounds','ray_valid'))
    observer,scales=batch['observer_position'],batch['range_scale_m'];scale=scales[:,None,None]
    visual=batch['visual_scale'] if core.scale_enabled else torch.zeros_like(batch['visual_scale'])
    b,n=valid.shape
    geometry=torch.cat((origin/scale,direction,(origin-observer[:,None])/scale,bounds/scale,visual),-1)
    encoded=core.visual_encoder(geometry)+core.roi_projection(batch['detector_roi_features'].detach())
    visual_token=batch_engine._interact(core.visual_interaction,encoded,valid,has_active=bool(batch['active_frames']))
    prior=core.visual_prior_head(visual_token);anchor=core.depth_prior_m.expand(b,n)
    if core.scale_enabled:anchor=anchor*.05/visual[...,2].clamp_min(1e-5)
    mean=batch_engine._bounded(anchor,prior[...,0],bounds)*valid
    std=core.std_min_m+(core.std_max_m-core.std_min_m)*torch.sigmoid(prior[...,1])
    global_token=visual_token;local=torch.zeros_like(visual_token);active=batch['active_indices']
    diagnostic=dict(query_rounds=1,graph_variant=model.config['graph_variant'])
    if batch['active_frames']:
        ra,re,av,ev=(batch[name].index_select(0,active) for name in ('ra_power','re_power','ra_valid','re_valid'))
        ranges,az,el=(batch[name] for name in batch_engine.AXIS_FIELDS);sc=scales.index_select(0,active)
        power_scale=((ra.masked_fill(~av,0.).sum((1,2))+re.masked_fill(~ev,0.).sum((1,2)))/(av.sum((1,2))+ev.sum((1,2))).clamp_min(1)).clamp_min(1e-20)
        ra_map,ra_rows,ra_rows_valid=batch_engine._native_encoder(core.ra_encoder,ra,av,ranges,az,sc,power_scale)
        re_map,re_rows,re_rows_valid=batch_engine._native_encoder(core.re_encoder,re,ev,ranges,el,sc,power_scale)
        reference=core.observer_encoder(observer.index_select(0,active)/sc[:,None])
        memory=torch.cat((ra_rows,re_rows,reference[:,None]),1)
        memory_valid=torch.cat((ra_rows_valid,re_rows_valid,torch.ones((len(batch['active_frames']),1),device=valid.device,dtype=torch.bool)),1)
        token=visual_token.index_select(0,active)
        context,_=core.global_attention(token,memory,memory,key_padding_mask=~memory_valid,need_weights=False)
        global_token=global_token.index_copy(0,active,core.global_norm(token+context)*valid.index_select(0,active)[...,None])
        loc,diagnostic=core.local_query(origin=origin.index_select(0,active),direction=direction.index_select(0,active),
            bounds=bounds.index_select(0,active),ray_valid=valid.index_select(0,active),visual_mean=mean.index_select(0,active),
            visual_std=std.index_select(0,active),visual_token=token,ra_map=ra_map,re_map=re_map,
            range_axis=ranges,azimuth_axis=az,elevation_axis=el,ra_valid=av,re_valid=ev,
            range_scale_m=sc,sampler=batch_engine._sample)
        local=local.index_copy(0,active,loc)
        # Associate diagnostics with original packed rows, including empty ones.
        diagnostic['source_batch_indices']=active
    fused=core.fusion_encoder(torch.cat((visual_token,global_token,local,mean[...,None]/scale,std[...,None]/scale),-1))
    fused=batch_engine._interact(core.fusion_interaction,fused,valid,has_active=bool(batch['active_frames']))
    depth=batch_engine._bounded(mean,core.depth_head(fused)[...,0],bounds)*valid
    positions=(origin+depth[...,None]*direction)*valid[...,None]
    return dict(depths=depth,visual_depth_mean_m=mean,visual_depth_std_m=std,member_positions=positions,
        valid_mask=valid,center=positions.sum(1)/valid.sum(1).clamp_min(1)[:,None],state=None,
        sampling_diagnostics=diagnostic,graph_diagnostics=diagnostic,counts=batch['counts'],
        aux=dict(architecture='single_query_depth_v3',graph_architecture=ARCHITECTURE,group=model.group,
                 radar_used=True,query_rounds=1,depth_refinement_steps=0))
