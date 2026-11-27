"""Parameter-identical tensor-batched execution of the locked evidence D.

The inherited scalar forward/predict remain the reference evaluation path.
Only an explicitly packed, previously validated batch uses these operations.
No autocast, compile, extra model parameters, new loss or query is introduced.
"""
from __future__ import annotations

from copy import deepcopy
import torch
from torch.nn import functional as F

from d_evidence_study.model import EvidenceDepthLocalizer, FLAGS, SCHEMA


def pack_batch(features_list, positions=None, matches=None, *, device='cpu', config=None):
    from .data import pack_batch as prepare
    return prepare(features_list,positions=positions,matches=matches,device=device,config=config)


def _interact(module,tokens,valid):
    # One safe dummy key in all-invalid frames prevents a masked softmax NaN.
    # Every output of such a frame is subsequently masked to exact zero.
    padding=~valid
    first=padding[:,:1] & valid.any(-1,keepdim=True)
    padding=torch.cat((first,padding[:,1:]),-1)
    return module(tokens,src_key_padding_mask=padding)*valid[...,None]


def _interior(bounds):
    width=(bounds[...,1]-bounds[...,0]).clamp_min(0.)
    magnitude=bounds.abs().amax(-1).clamp_min(1.)
    inset=torch.minimum(width*.01,64.*torch.finfo(bounds.dtype).eps*magnitude)
    return torch.stack((bounds[...,0]+inset,bounds[...,1]-inset),-1)


def _bounded(anchor,residual,bounds):
    interior=_interior(bounds)
    width=(interior[...,1]-interior[...,0]).clamp_min(1e-6)
    fraction=((anchor-interior[...,0])/width).clamp(1e-5,1.-1e-5)
    return interior[...,0]+width*torch.sigmoid(torch.logit(fraction)+residual)


def _support(base,raw,bounds,correction):
    interior=_interior(bounds)
    negative=torch.minimum((base-interior[...,0]).clamp_min(0.),torch.full_like(base,correction))*.99
    positive=torch.minimum((interior[...,1]-base).clamp_min(0.),torch.full_like(base,correction))*.99
    return base+torch.where(raw>=0.,positive,negative)*raw


def _spectrum(encoder,power,mask,ranges,angles,scale,power_scale):
    b,nr,na=power.shape
    values=torch.stack((torch.log1p(power.masked_fill(~mask,0.)/power_scale[:,None,None]),
        (ranges[None,:,None]/scale[:,None,None]).expand(b,nr,na),
        angles.sin()[None,None,:].expand(b,nr,na),angles.cos()[None,None,:].expand(b,nr,na),
        mask.to(power.dtype)),1)
    native=encoder.cnn(values)*mask[:,None]
    mean=native.sum(-1).transpose(1,2)/mask.sum(-1).clamp_min(1)[...,None]
    # Match Tensor.max's first-index tie gradient, rather than amax's shared
    # gradient across tied angular bins.
    maximum=native.masked_fill(~mask[:,None],-torch.inf).max(-1).values.transpose(1,2)
    rows_valid=mask.any(-1)
    maximum=torch.where(rows_valid[...,None],maximum,torch.zeros_like(maximum))
    return native,encoder.row(torch.cat((mean,maximum),-1)),rows_valid


def _axis(values,axis):
    # Physical axes are immutable and validated once in data.pack_batch.
    lower=torch.searchsorted(axis,values.contiguous(),right=True).sub(1).clamp(0,len(axis)-2)
    index=lower.to(values.dtype)+(values-axis[lower])/(axis[lower+1]-axis[lower])
    valid=torch.isfinite(values)&(values>=axis[0])&(values<=axis[-1])
    return 2.*index/(len(axis)-1)-1.,valid


def _sample(feature_map,mask,ranges,angles,range_axis,angle_axis,candidate_valid):
    y,rv=_axis(ranges,range_axis);x,av=_axis(angles,angle_axis)
    grid=torch.stack((x,y),-1)
    finite=torch.isfinite(grid).all(-1)
    grid=torch.where(finite[...,None],grid,torch.full_like(grid,2.))
    sampled=F.grid_sample(feature_map,grid,mode='bilinear',padding_mode='zeros',align_corners=True).permute(0,2,3,1)
    weights=F.grid_sample(mask[:,None].to(feature_map.dtype),grid,mode='bilinear',
        padding_mode='zeros',align_corners=True)[:,0]
    valid=rv&av&finite&candidate_valid&(weights>=1.-1e-6)
    return torch.where(valid[...,None],sampled,torch.zeros_like(sampled)),valid


def _geometry(origin,direction,depths,bounds,valid):
    positions=origin[:,:,None,:]+depths[...,None]*direction[:,:,None,:]
    radius=positions.square().sum(-1).clamp_min(1e-12).sqrt()
    horizontal=positions[...,:2].square().sum(-1).clamp_min(1e-12).sqrt()
    # Padding has exactly x=y=0. atan2(0,0) returns zero forward but its
    # undefined derivative contaminates masked gradients with NaNs. The safe
    # origin yields the same zero azimuth and never changes a physical x>0
    # candidate, including its derivatives.
    has_horizontal=(positions[...,0]!=0)|(positions[...,1]!=0)
    azimuth=torch.atan2(torch.where(has_horizontal,positions[...,1],torch.zeros_like(positions[...,1])),
        torch.where(has_horizontal,positions[...,0],torch.ones_like(positions[...,0])))
    elevation=torch.atan2(positions[...,2],horizontal)
    usable=(valid[:,:,None]&torch.isfinite(depths)&(depths>=bounds[...,0,None])&
        (depths<=bounds[...,1,None])&(depths>0.)&(positions[...,0]>0.)&(radius>1e-6))
    return dict(ranges=radius,azimuth=azimuth,elevation=elevation,valid=usable)


class FastDepthLocalizer(EvidenceDepthLocalizer):
    def __init__(self,config):
        super().__init__(config)
        # Nonpersistent execution constants; do not add state_dict keys.
        self._speed_orders={}
        self._speed_offset_order=[sorted(range(len(config.get('depth_offsets_m',[]))),
            key=lambda i:config['depth_offsets_m'][i])]
        fine=config.get('depth_fine_offsets_m')
        self._speed_offset_order.append(sorted(range(len(fine)),key=fine.__getitem__) if fine is not None else self._speed_offset_order[0])

    def pack_batch(self,features_list,positions=None,matches=None):
        return pack_batch(features_list,positions,matches,device=self.depth_prior_m.device,config=self.config)

    def _packed(self,value,positions=None,matches=None):
        if hasattr(value,'inputs') and hasattr(value,'lengths'):
            if positions is not None or matches is not None:
                raise ValueError('Attach training targets when packing; a packed batch already owns its target identity')
            batch=value
        else:
            batch=self.pack_batch(value,positions,matches)
        # The packed lengths are host integers. Retain the scalar model's
        # supported member cap without reading any CUDA value or revalidating
        # immutable spectra in this hot path.
        if any(length>self.max_members for length in batch.lengths):
            raise ValueError('Detected count exceeds configured model support')
        return batch

    def _order(self,offsets,fine_stage):
        key=(str(offsets.device),bool(fine_stage))
        if key not in self._speed_orders:
            order=self._speed_offset_order[int(fine_stage)]
            if len(order)!=len(offsets):
                # Only the compatibility control may omit explicit offsets.
                order=torch.argsort(offsets.detach().cpu(),stable=True).tolist()
            self._speed_orders[key]=torch.as_tensor(order,dtype=torch.long,device=offsets.device)
        return self._speed_orders[key]

    def _sequence(self,candidate,ra,re,offsets,depths,geometry,av,ev,usable,scale,fine_stage):
        b,n,s,h=candidate.shape
        av=av&usable;ev=ev&usable;gv=geometry['valid']&usable
        scalar=torch.stack((offsets[None,None].expand(b,n,-1)/16.,depths/scale[:,None,None],
            geometry['ranges']/scale[:,None,None],geometry['azimuth'],geometry['elevation'],
            av.to(candidate.dtype),ev.to(candidate.dtype),gv.to(candidate.dtype)),-1)
        ra=torch.where(av[...,None],ra,torch.zeros_like(ra))
        re=torch.where(ev[...,None],re,torch.zeros_like(re))
        values=torch.cat((candidate,ra,re,scalar),-1)
        values=torch.where(usable[...,None],values,torch.zeros_like(values))
        order=self._order(offsets,fine_stage)
        mask=usable.index_select(2,order).reshape(b*n,1,s).to(values.dtype)
        encoded=values.index_select(2,order).reshape(b*n,s,-1).transpose(1,2)
        branch=self.candidate_sequence
        encoded=branch.activation(branch.first(encoded))*mask
        encoded=branch.activation(branch.second(encoded))*mask
        return branch.readout(encoded.flatten(1)).reshape(b,n)*usable.any(-1).to(values.dtype)

    def forward_batch(self,value):
        batch=self._packed(value);x=batch.inputs
        origin=x['ray_origins'];direction=x['ray_directions'];bounds=x['ray_depth_bounds']
        valid=x['ray_valid']&x['member_mask'];observer=x['observer_position'];visual=x['visual_scale']
        scale=x['range_scale_m'];b,n=valid.shape
        if not self.scale_enabled:visual=torch.zeros_like(visual)
        visual_input=torch.cat((origin/scale[:,None,None],direction,
            (origin-observer[:,None])/scale[:,None,None],bounds/scale[:,None,None],visual),-1)
        encoded=self.visual_encoder(visual_input)
        if self.roi_enabled:
            encoded=encoded+self.roi_projection(x['detector_roi_features'].detach())
        tokens=_interact(self.visual_interaction,encoded,valid)
        anchor=self.depth_prior_m.expand(b,n)
        if self.scale_enabled:anchor=anchor*.05/visual[...,2].clamp_min(1e-5)
        initial=_bounded(anchor,self.coarse_head(tokens)[...,0],bounds)*valid
        ra,re=x['ra_power'],x['re_power'];av,ev=x['ra_valid'],x['re_valid']
        ranges,az,el=x['range_m'],x['azimuth_rad'],x['elevation_rad']
        power_scale=((ra.masked_fill(~av,0.).sum((-2,-1))+re.masked_fill(~ev,0.).sum((-2,-1)))/
            (av.sum((-2,-1))+ev.sum((-2,-1))).clamp_min(1)).clamp_min(1e-20)
        ra_map,ra_rows,ra_rows_valid=_spectrum(self.ra_encoder,ra,av,ranges,az,scale,power_scale)
        re_map,re_rows,re_rows_valid=_spectrum(self.re_encoder,re,ev,ranges,el,scale,power_scale)
        reference=self.observer_encoder(observer/scale[:,None])
        memory=torch.cat((ra_rows,re_rows,reference[:,None]),1)
        memory_valid=torch.cat((ra_rows_valid,re_rows_valid,valid.new_ones((b,1))),1)
        context,_=self.global_attention(tokens,memory,memory,key_padding_mask=~memory_valid,need_weights=False)
        global_token=self.global_norm(tokens+context)*valid[...,None]
        global_values=self.global_head(global_token)
        global_depth=_bounded(self.depth_prior_m.expand(b,n),global_values[...,0],bounds)*valid
        global_gate=torch.sigmoid(global_values[...,1])*valid
        depth=initial
        fractions=[];gates=[];stage_offsets=[];stage_limits=[];stage_depths=[];sequence_logits=[]
        queries=[];bases=[];candidate_depths=[];geometry_trace=[]
        for step in range(self.steps):
            fine_stage=self.fine_enabled and step==1
            offsets=self.depth_fine_offsets_m if fine_stage else self.depth_offsets_m
            correction=self.fine_max_correction if fine_stage else self.max_correction
            gate=torch.zeros_like(global_gate) if fine_stage else global_gate
            base=(1.-gate)*depth+gate*global_depth
            center=base if self.aligned_enabled and step==0 else depth
            hypotheses=center[...,None]+offsets[None,None]
            geometry=_geometry(origin,direction,hypotheses,bounds,valid)
            sampled_ra,sav=_sample(ra_map,av,geometry['ranges'],geometry['azimuth'],ranges,az,geometry['valid'])
            sampled_re,sev=_sample(re_map,ev,geometry['ranges'],geometry['elevation'],ranges,el,geometry['valid'])
            usable=(sav|sev) if self.geometry_enabled else torch.zeros_like(sav)
            angular=torch.stack((geometry['azimuth'].sin(),geometry['azimuth'].cos(),
                geometry['elevation'].sin(),geometry['elevation'].cos()),-1)
            scalars=torch.cat((hypotheses[...,None]/scale[:,None,None,None],
                geometry['ranges'][...,None]/scale[:,None,None,None],
                (offsets/correction)[None,None,:,None].expand(b,n,-1,-1),angular,
                sav[...,None].to(ra.dtype),sev[...,None].to(ra.dtype),geometry['valid'][...,None].to(ra.dtype)),-1)
            candidate=self.sample_encoder(torch.cat((sampled_ra,sampled_re,scalars),-1))
            logits=self.sample_score(torch.cat((candidate,global_token[:,:,None].expand(-1,-1,len(offsets),-1)),-1))[...,0]
            weights=torch.softmax(logits.masked_fill(~usable,-1e4),-1)*usable
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
            local=(candidate*weights[...,None]).sum(2)
            refine=self.refine_encoder(torch.cat((tokens,global_token,local,depth[...,None]/scale[:,None,None]),-1))
            refine=_interact(self.refine_interaction,refine,valid)
            raw_logit=self.refine_head(refine)[...,0]
            extra=self._sequence(candidate,sampled_ra,sampled_re,offsets,hypotheses,geometry,sav,sev,usable,scale,fine_stage) if self.sequence_enabled else torch.zeros_like(raw_logit)
            depth=_support(base,torch.tanh(raw_logit+extra),bounds,correction)*valid
            fractions.append(usable.to(ra.dtype).mean(-1));gates.append(gate)
            stage_offsets.append(offsets);stage_limits.append(correction);stage_depths.append(depth);sequence_logits.append(extra)
            if self.capture_candidates:
                queries.append(center);bases.append(base);candidate_depths.append(hypotheses)
                geometry_trace.append(dict(offsets_m=offsets,depths_m=hypotheses,ranges_m=geometry['ranges'],
                    azimuth_rad=geometry['azimuth'],elevation_rad=geometry['elevation'],ra_valid=sav,re_valid=sev,
                    geometry_valid=geometry['valid'],usable=usable))
            tokens=refine
        positions=(origin+depth[...,None]*direction)*valid[...,None]
        diagnostics=dict(local_valid_fraction=fractions,global_gate=gates,global_depths=global_depth,
            geometry_enabled=self.geometry_enabled,candidate_offsets_m=self.depth_offsets_m,
            global_proposal_is_search=False,stage_candidate_offsets_m=stage_offsets,
            stage_max_correction_m=stage_limits,stage_depths_m=stage_depths,
            refinement_mode='coarse_fine' if self.fine_enabled else 'shared_scale',mixed_observer_preserved=True,
            independent_peak_assignment=False,evidence_arm=self.evidence_arm,stage_sequence_residual_logits=sequence_logits)
        if self.capture_candidates:
            diagnostics.update(stage_query_centers_m=queries,stage_update_bases_m=bases,
                stage_candidate_depths_m=candidate_depths,stage_candidate_geometry=geometry_trace,sequence_details=[])
        return dict(depths=depth,initial_depths=initial,member_positions=positions,
            initial_positions=(origin+initial[...,None]*direction)*valid[...,None],valid_mask=valid,
            center=(positions*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None],
            sampling_diagnostics=diagnostics,member_mask=x['member_mask'])

    def training_loss_batch(self,value,positions=None,matches=None):
        batch=self._packed(value,positions,matches)
        if batch.targets is None:
            raise ValueError('Training targets must be supplied separately when packing the batch')
        prediction=self.forward_batch(batch);targets=batch.targets
        mask=targets['valid'];weight=mask.to(prediction['depths'].dtype)
        count=weight.sum(-1).clamp_min(1)
        target=targets['depths_m']
        initial=F.huber_loss(prediction['initial_depths'],target,reduction='none',delta=self.huber_delta)
        final=F.huber_loss(prediction['depths'],target,reduction='none',delta=self.huber_delta)
        squared=(prediction['depths']-target).square()
        initial=(initial*weight).sum(-1)/count;final=(final*weight).sum(-1)/count
        squared=(squared*weight).sum(-1)/count
        position_squared=(prediction['member_positions']-targets['positions_base_m']).square().sum(-1)
        rmse=((position_squared*weight).sum(-1)/count).sqrt()
        per_frame=dict(initial_depth_huber=initial,final_depth_huber=final,final_depth_mse=squared,matched_rmse_m=rmse)
        loss=(final+self.mse_weight*squared+self.initial_weight*initial).mean()
        terms={key:value.mean() for key,value in per_frame.items()}
        matched=mask.sum()
        valid_rays=(batch.inputs['ray_valid']&batch.inputs['member_mask']).sum()
        terms.update(per_frame=per_frame,prediction=prediction,matched_training_count=mask.sum(),
            unmatched_detection_count=valid_rays-matched,
            unmatched_label_count=targets['truth_count'].sum()-matched,
            valid_training_rays=valid_rays,training_detected_count=batch.inputs['member_mask'].sum(),
            supervision_usable=mask.any(-1).all(),
            loss_units='same per-frame Huber/MSE metric-depth objective, then equal-frame batch mean')
        return loss,terms

    def predict_batch(self,value,allow_untrained_validation=False):
        if not self.trained and not allow_untrained_validation:
            raise RuntimeError('A trained evidence model is required')
        batch=self._packed(value)
        with torch.no_grad():output=self.forward_batch(batch)
        def numpy(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().numpy()
            if isinstance(value,dict):return {k:numpy(v) for k,v in value.items()}
            if isinstance(value,list):return [numpy(v) for v in value]
            return value
        output=numpy(output);scenes=[];shared=output['sampling_diagnostics']
        per_member={'global_depths'}
        lists={'local_valid_fraction','global_gate','stage_depths_m','stage_sequence_residual_logits',
            'stage_query_centers_m','stage_update_bases_m','stage_candidate_depths_m'}
        for i,n in enumerate(batch.lengths):
            scene={key:output[key][i,:n] for key in ('depths','initial_depths','member_positions','initial_positions','valid_mask')}
            scene['center']=output['center'][i]
            diagnostic={}
            for key,value in shared.items():
                if key in per_member:diagnostic[key]=value[i,:n]
                elif key in lists:diagnostic[key]=[v[i,:n] for v in value]
                elif key=='stage_candidate_geometry':
                    diagnostic[key]=[{k:(v if k=='offsets_m' else v[i,:n]) for k,v in stage.items()} for stage in value]
                else:diagnostic[key]=value
            scene.update(sampling_diagnostics=diagnostic,model_trained=bool(self.trained),
                source='trained_d_evidence_depth' if self.trained else 'untrained_d_evidence_validation',
                uncertainty_calibrated=False,hypothesis_index=0,input_frames=1,
                decoder='original two-round D continuous camera-ray depth with one isolated internal evidence change',
                mode_source='single deterministic prediction; not posterior or MAP',depth_unit='camera slant metres',
                velocities_predicted=False,d_evidence_schema=SCHEMA,d_evidence_arm=self.evidence_arm,
                d_evidence_flags={name:self.config[name] for name in FLAGS},detector_frozen=True,
                roi_features_used=self.roi_enabled)
            scenes.append(scene)
        return scenes


def from_existing(model,device=None):
    """Copy weights to the equivalent executor without consuming its RNG stream."""
    with torch.random.fork_rng(devices=[]):
        result=FastDepthLocalizer(deepcopy(model.config))
    result.load_state_dict(model.state_dict(),strict=True)
    result.to(device or next(model.parameters()).device)
    result.train(model.training);result.trained=model.trained
    if hasattr(model,'checkpoint_metadata'):result.checkpoint_metadata=deepcopy(model.checkpoint_metadata)
    return result


FastEvidenceDepthLocalizer=FastDepthLocalizer
DepthLocalizer=FastDepthLocalizer
