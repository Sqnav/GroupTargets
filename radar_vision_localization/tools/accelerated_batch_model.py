"""Batched execution of the existing single-query model, sharing every weight.

CPU inputs are validated once by ``pack_features``. Frames are independent
batch elements; padded members cannot attend to another frame. The geometry
is the trusted candidate_geometry implementation. Physical grid interpolation
and sampling are its exact batched tensor extension, tested against the trusted
single-frame sampler (including nonuniform axes and independent RA/RE masks).
No scattering model, new trainable module, GT input or radar approximation is
introduced. Save/load the original model, not a new checkpoint architecture.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from rvloc.models.single_frame import current_features, DepthLocalizer
from rvloc.data import validate_direct_fusion_features
from rvloc.legacy import activate
activate()
from precision_study.visual import ALLOWED_FIELDS, validate_visual_features
from depth_localization.geometry import candidate_geometry

SCHEMA = 'rvloc.padded_single_query_inputs.v1'
MEMBER_FIELDS = ('ray_origins','ray_directions','ray_depth_bounds','ray_valid',
                 'visual_scale','detector_roi_features')
RADAR_FIELDS = ('ra_power','re_power','ra_valid','re_valid')
AXIS_FIELDS = ('range_m','azimuth_rad','elevation_rad')


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pack_features(feature_list, *, device='cpu', visual=False, validate=True,
                  max_members=24, dtype=torch.float32):
    """Authenticate/validate CPU feature dictionaries, then pad and upload once.

    The returned tensors may be retained on GPU: these are the frozen ROI,
    raw RA/RE power and public geometry, never CNN outputs. Physical axes are
    required to agree within a batch; range scales may differ by frame.
    """
    if not feature_list:
        raise ValueError('At least one frame is required')
    selected = []
    for features in feature_list:
        value = current_features(features, visual=visual)
        if validate:
            if visual:
                validate_visual_features({k:value[k] for k in ALLOWED_FIELDS})
            else:
                validate_direct_fusion_features(value)
            roi = _numpy(value['detector_roi_features'])
            n = len(value['ray_directions'])
            if roi.shape != (n,1024) or not np.isfinite(roi).all():
                raise ValueError('Finite detection-aligned frozen ROI [N,1024] required')
        selected.append(value)
    counts = tuple(len(x['ray_directions']) for x in selected)
    if max(counts) > max_members:
        raise ValueError('Detected count exceeds model support')
    b, n = len(selected), max(1,max(counts))
    arrays = dict(ray_origins=np.zeros((b,n,3)),ray_directions=np.zeros((b,n,3)),
        ray_depth_bounds=np.broadcast_to([.1,1.],(b,n,2)).copy(),
        ray_valid=np.zeros((b,n),dtype=bool),visual_scale=np.zeros((b,n,7)),
        detector_roi_features=np.zeros((b,n,1024),dtype=np.float32))
    arrays['ray_directions'][...,0] = 1.
    for i,value in enumerate(selected):
        for key in MEMBER_FIELDS:
            arrays[key][i,:counts[i]] = _numpy(value[key])
    active = tuple(i for i in range(b) if arrays['ray_valid'][i].any())
    out = dict(schema=SCHEMA,counts=counts,active_frames=active,visual=visual)
    for key,value in arrays.items():
        out[key] = torch.as_tensor(value,device=device,dtype=torch.bool if key=='ray_valid' else dtype)
    out['observer_position'] = torch.as_tensor(np.stack([_numpy(x['observer_position']) for x in selected]),device=device,dtype=dtype)
    out['range_scale_m'] = torch.tensor([float(x['range_scale_m']) for x in selected],device=device,dtype=dtype)
    out['active_indices'] = torch.tensor(active,device=device,dtype=torch.long)
    if not visual:
        for key in RADAR_FIELDS:
            out[key] = torch.as_tensor(np.stack([_numpy(x[key]) for x in selected]),device=device,
                dtype=torch.bool if key.endswith('_valid') else dtype)
        for key in AXIS_FIELDS:
            first = _numpy(selected[0][key])
            if any(not np.array_equal(first,_numpy(x[key])) for x in selected[1:]):
                raise ValueError('Physical axes must agree within a batch: '+key)
            out[key] = torch.as_tensor(first,device=device,dtype=dtype)
    return out


collate_frames = pack_features


def _interact(module,tokens,valid,*,has_active=True):
    if not has_active:
        return tokens*valid[...,None]
    # Empty frames use an attention-only dummy key; it is removed from every
    # output and cannot leak to another frame or any real member.
    empty = ~valid.any(-1)
    first = torch.arange(valid.shape[1],device=valid.device)==0
    safe = valid | (empty[:,None]&first[None])
    return module(tokens,src_key_padding_mask=~safe)*valid[...,None]


def _bounded(anchor,residual,bounds):
    shape=anchor.shape
    return DepthLocalizer._bounded(anchor.reshape(-1),residual.reshape(-1),bounds.reshape(-1,2)).reshape(shape)


def _native_encoder(module,power,mask,ranges,angles,scales,power_scale):
    b,nr,na=power.shape
    values=torch.stack((torch.log1p(power.masked_fill(~mask,0.)/power_scale[:,None,None]),
        (ranges[None,:,None]/scales[:,None,None]).expand(b,nr,na),
        angles.sin()[None,None].expand(b,nr,na),angles.cos()[None,None].expand(b,nr,na),mask.to(power.dtype)),1)
    native=module.cnn(values)*mask[:,None]
    mean=native.sum(-1).transpose(1,2)/mask.sum(-1).clamp_min(1)[...,None]
    maximum=native.masked_fill(~mask[:,None],-torch.inf).max(-1).values.transpose(1,2)
    row_valid=mask.any(-1)
    maximum=torch.where(row_valid[...,None],maximum,torch.zeros_like(maximum))
    rows=module.row(torch.cat((mean,maximum),-1))
    return native,rows,row_valid


def _axis_index(values,axis):
    # Tensor operations identical to trusted geometry._axis_index. Axis dtype,
    # finiteness and strict monotonicity were checked by pack_features on CPU.
    lower=torch.searchsorted(axis,values.contiguous(),right=True).sub(1).clamp(0,len(axis)-2)
    index=lower.to(values.dtype)+(values-axis[lower])/(axis[lower+1]-axis[lower])
    valid=torch.isfinite(values)&(values>=axis[0])&(values<=axis[-1])
    return 2.*index/(len(axis)-1)-1.,valid


def _sample(feature,mask,ranges,angles,range_axis,angle_axis,candidate_valid):
    y,ry=_axis_index(ranges,range_axis);x,ax=_axis_index(angles,angle_axis)
    grid=torch.stack((x,y),-1);finite=torch.isfinite(grid).all(-1)
    safe=torch.where(finite[...,None],grid,torch.full_like(grid,2.))
    values=F.grid_sample(feature,safe,mode='bilinear',padding_mode='zeros',align_corners=True).permute(0,2,3,1)
    weight=F.grid_sample(mask[:,None].to(feature.dtype),safe,mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]
    valid=ry&ax&finite&candidate_valid&(weight>=1.-1e-6)
    return torch.where(valid[...,None],values,torch.zeros_like(values)),valid


def _local_query(module,origin,direction,bounds,valid,mean,std,ra_map,re_map,ranges,az,el,av,ev,scales):
    b,n=mean.shape;s=len(module.candidate_multipliers)
    z=module.candidate_multipliers.to(dtype=mean.dtype)
    offsets=std[...,None]*z;depths=mean[...,None]+offsets
    geometry=candidate_geometry(origin.reshape(-1,3),direction.reshape(-1,3),depths.reshape(-1,s),bounds.reshape(-1,2),valid.reshape(-1))
    if geometry['positions'].requires_grad:
        mask=geometry['valid']
        geometry['positions'].register_hook(lambda grad:torch.where(mask[...,None],grad,torch.zeros_like(grad)))
    g={k:v.reshape((b,n,s)+(v.shape[2:] if k=='positions' else ())) for k,v in geometry.items()}
    ra,ra_ok=_sample(ra_map,av,g['ranges'],g['azimuth'],ranges,az,g['valid'])
    re,re_ok=_sample(re_map,ev,g['ranges'],g['elevation'],ranges,el,g['valid'])
    usable=(ra_ok|re_ok)&g['valid']
    angular=torch.stack((g['azimuth'].sin(),g['azimuth'].cos(),g['elevation'].sin(),g['elevation'].cos()),-1)
    scale=scales[:,None,None,None]
    scalars=torch.cat((offsets[...,None]/scale,(offsets/std[...,None])[...,None],
        depths[...,None]/scale,g['ranges'][...,None]/scale,angular,
        ra_ok[...,None].to(mean.dtype),re_ok[...,None].to(mean.dtype),g['valid'][...,None].to(mean.dtype)),-1)
    inputs=torch.cat((ra,re,scalars),-1)
    inputs=torch.where(usable[...,None],inputs,torch.zeros_like(inputs))
    tokens=module.candidate_encoder(inputs)
    tokens=torch.where(usable[...,None],tokens,torch.zeros_like(tokens))
    evidence=usable.any(-1)
    # Preserve the original module's grad=None (and hence AdamW behavior) when
    # no candidate Transformer is used anywhere in the optimizer batch. This
    # is one logical branch, not a repeated GPU finiteness validation.
    if bool(evidence.any()):
        interacted=_interact(module.candidate_interaction,tokens.reshape(b*n,s,-1),usable.reshape(b*n,s)).reshape_as(tokens)
    else:
        interacted=torch.zeros_like(tokens)
    scores=module.candidate_score(interacted)[...,0]
    weights=torch.softmax(scores.masked_fill(~usable,torch.finfo(scores.dtype).min),-1)*usable
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    pooled=(interacted*weights[...,None]).sum(2)
    pooled=torch.where(evidence[...,None],pooled,torch.zeros_like(pooled))
    return pooled,dict(query_rounds=1,candidate_depths_m=depths,candidate_offsets_m=offsets,
        geometry_valid=g['valid'],ra_valid=ra_ok,re_valid=re_ok,usable=usable,weights=weights,
        evidence_available=evidence)


def prepare_evidence(model,batch):
    """Batch the current visual/radar evidence before causal memory injection.

    This graph is valid for one optimizer update only. It must be recomputed
    after an optimizer step: this is not a cache of trainable CNN outputs.
    G7 may prepare its four current-frame inputs together, then finish them
    sequentially with priors from its unchanged observed-ID GRU.
    """
    core=model.core
    if batch.get('schema')!=SCHEMA or batch['visual']!=core.visual:
        raise ValueError('Packed input modality/schema does not match model')
    origin,direction,bounds,valid=(batch[k] for k in ('ray_origins','ray_directions','ray_depth_bounds','ray_valid'))
    observer,scales=batch['observer_position'],batch['range_scale_m']
    visual=batch['visual_scale'] if core.scale_enabled else torch.zeros_like(batch['visual_scale'])
    b,n=valid.shape;scale=scales[:,None,None]
    geometry=torch.cat((origin/scale,direction,(origin-observer[:,None])/scale,bounds/scale,visual),-1)
    encoded=core.visual_encoder(geometry)+core.roi_projection(batch['detector_roi_features'].detach())
    visual_token=_interact(core.visual_interaction,encoded,valid,has_active=bool(batch['active_frames']))
    prior=core.visual_prior_head(visual_token)
    anchor=core.depth_prior_m.expand(b,n)
    if core.scale_enabled:anchor=anchor*.05/visual[...,2].clamp_min(1e-5)
    mean=_bounded(anchor,prior[...,0],bounds)*valid
    std=core.std_min_m+(core.std_max_m-core.std_min_m)*torch.sigmoid(prior[...,1])
    diagnostic=dict(query_rounds=0 if core.visual else 1)
    fused=None
    if not core.visual:
        global_token=visual_token;local=torch.zeros_like(visual_token)
        active=batch['active_indices']
        if batch['active_frames']:
            ra,re,av,ev=(batch[k].index_select(0,active) for k in ('ra_power','re_power','ra_valid','re_valid'))
            ranges,az,el=(batch[k] for k in AXIS_FIELDS)
            sc=scales.index_select(0,active)
            power_scale=((ra.masked_fill(~av,0.).sum((1,2))+re.masked_fill(~ev,0.).sum((1,2)))/(av.sum((1,2))+ev.sum((1,2))).clamp_min(1)).clamp_min(1e-20)
            ra_map,ra_rows,ra_rows_valid=_native_encoder(core.ra_encoder,ra,av,ranges,az,sc,power_scale)
            re_map,re_rows,re_rows_valid=_native_encoder(core.re_encoder,re,ev,ranges,el,sc,power_scale)
            reference=core.observer_encoder(observer.index_select(0,active)/sc[:,None])
            memory=torch.cat((ra_rows,re_rows,reference[:,None]),1)
            memory_valid=torch.cat((ra_rows_valid,re_rows_valid,torch.ones((len(batch['active_frames']),1),device=valid.device,dtype=torch.bool)),1)
            token=visual_token.index_select(0,active)
            context,_=core.global_attention(token,memory,memory,key_padding_mask=~memory_valid,need_weights=False)
            ctx=core.global_norm(token+context)*valid.index_select(0,active)[...,None]
            global_token=global_token.index_copy(0,active,ctx)
            loc,diagnostic=_local_query(core.local_query,origin.index_select(0,active),direction.index_select(0,active),
                bounds.index_select(0,active),valid.index_select(0,active),mean.index_select(0,active),std.index_select(0,active),
                ra_map,re_map,ranges,az,el,av,ev,sc)
            local=local.index_copy(0,active,loc)
        fused=core.fusion_encoder(torch.cat((visual_token,global_token,local,mean[...,None]/scale,std[...,None]/scale),-1))
    return dict(schema='rvloc.prepared_batch_evidence.v1',visual=core.visual,
        origin=origin,direction=direction,bounds=bounds,valid=valid,mean=mean,std=std,
        base_fusion=fused,active_frames=batch['active_frames'],counts=batch['counts'],
        sampling_diagnostics=diagnostic)


def slice_prepared(prepared, indices):
    """Select frame rows without detaching their trainable evidence graph."""
    if isinstance(indices,int):indices=[indices]
    indices=tuple(indices)
    if any(type(i) is not int or not 0<=i<len(prepared['counts']) for i in indices):
        raise ValueError('Prepared frame indices must be valid Python integers')
    index=torch.tensor(indices,device=prepared['origin'].device,dtype=torch.long)
    out=dict(prepared)
    for key in ('origin','direction','bounds','valid','mean','std','base_fusion'):
        value=prepared[key]
        if value is not None:out[key]=value.index_select(0,index)
    out['counts']=tuple(prepared['counts'][i] for i in indices)
    out['active_frames']=tuple(j for j,i in enumerate(indices) if i in prepared['active_frames'])
    # Diagnostics are not used by the recurrent loss. Their frame-dependent
    # fields belong to the original evidence batch, so do not mislabel them.
    out['sampling_diagnostics']=dict(query_rounds=0 if prepared['visual'] else 1,
                                    source_frame_indices=indices)
    return out


def finish_depth(model,prepared,*,fusion_prior=None):
    """Finish independent frame rows, optionally injecting a causal GRU prior."""
    core=model.core
    if prepared.get('schema')!='rvloc.prepared_batch_evidence.v1' or prepared['visual']!=core.visual:
        raise ValueError('Prepared evidence modality/schema differs from model')
    if model.group=='world_fusion' and fusion_prior is None:
        raise ValueError('World fusion requires explicit causal fusion_prior; this helper runs its core only')
    valid,mean,std=prepared['valid'],prepared['mean'],prepared['std']
    b,n=valid.shape
    if core.visual:
        if fusion_prior is not None:raise ValueError('RGB must not receive a radar/world prior')
        depth=mean
    else:
        fused=prepared['base_fusion']
        if fusion_prior is not None:
            if fusion_prior.shape!=fused.shape:raise ValueError('fusion_prior must match batched member tokens')
            fused=fused+fusion_prior.to(device=fused.device,dtype=fused.dtype)
        fused=_interact(core.fusion_interaction,fused,valid,has_active=bool(prepared['active_frames']))
        depth=_bounded(mean,core.depth_head(fused)[...,0],prepared['bounds'])*valid
    positions=(prepared['origin']+depth[...,None]*prepared['direction'])*valid[...,None]
    center=positions.sum(1)/valid.sum(1).clamp_min(1)[:,None]
    return dict(depths=depth,visual_depth_mean_m=mean,visual_depth_std_m=std,
        member_positions=positions,valid_mask=valid,center=center,state=None,
        sampling_diagnostics=prepared['sampling_diagnostics'],aux=dict(architecture=core.architecture,
            radar_used=not core.visual,group=model.group,query_rounds=0 if core.visual else 1,batched_frames=b),
        counts=prepared['counts'])


def forward_batch(model,batch,*,fusion_prior=None):
    """Run existing G2/G3 weights over independent padded frame rows."""
    return finish_depth(model,prepare_evidence(model,batch),fusion_prior=fusion_prior)


batch_forward=forward_batch
