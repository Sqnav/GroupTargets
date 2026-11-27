"""Exact <=4-frame world BPTT with batched current-frame evidence encoding.

Current-frame evidence is independent across batch rows. The public-ID GRU,
state aging and posterior are then evaluated in physical-time order. Fixed GT
labels enter only localization_loss and metric scoring, never recurrence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

from rvloc.config import ExperimentConfig
from rvloc.models import build_model,localization_loss,save_checkpoint
from rvloc.models.world_fusion import MemberState,WorldModelMemory
from rvloc.models.checkpoint import ARCHITECTURE
from rvloc.runtime import write_json,file_sha256
from rvloc.training import _settings,_code_hashes,_association_factory,_associate

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import accelerated_training as base
import accelerated_batch_model as engine


def public_identities(values,factory):
    """Replay frozen image-only association once; no model/GT argument exists."""
    result=[];episode=None;tracker=None
    for observation,_ in values:
        if observation.episode_id!=episode:
            episode=observation.episode_id;tracker=factory()
        ids=_associate(tracker,observation)
        result.append(dict(episode_id=episode,frame_index=observation.frame_idx,
            timestamp_s=float(observation.physical_time_s),member_ids=ids,
            ray_valid=tuple(bool(v) for v in observation.features['ray_valid'])))
    return result


def forward_sequence(model,packed,identities,memory=None,*,evidence=None):
    """Unchanged causal world recurrence; only current evidence is batched."""
    if model.group!='world_fusion':raise ValueError('World-fusion model required')
    if len(identities)!=len(packed['counts']):raise ValueError('One public identity record per frame required')
    if memory is not None and not isinstance(memory,WorldModelMemory):raise TypeError('WorldModelMemory required')
    prepared=engine.prepare_evidence(model,packed) if evidence is None else evidence
    outputs=[];parameter=model.memory_bridge.weight
    for t,meta in enumerate(identities):
        n=packed['counts'][t]
        if len(meta['ray_valid'])!=n:raise ValueError('Public validity must align with detections')
        fields=dict(episode_id=meta['episode_id'],frame_index=meta['frame_index'],timestamp_s=meta['timestamp_s'],
            member_ids=meta['member_ids'],ray_directions=packed['ray_directions'][t,:n])
        episode,frame,timestamp,ids=model._identity(fields)
        previous=memory.members if memory is not None and memory.episode_id==episode else {}
        available=[];durations=[];active=[];latents=[]
        for i,member in enumerate(ids):
            past=previous.get(member) if member is not None else None
            if past is not None and (frame<=past.frame_index or timestamp<=past.timestamp_s):
                raise ValueError('World memory must be strictly causal')
            seen=past is not None and timestamp-past.timestamp_s<=model.max_memory_age_s
            dt=timestamp-past.timestamp_s if seen else 1.
            available.append(seen);durations.append(dt)
            if seen:
                latent=past.latent.to(device=parameter.device,dtype=parameter.dtype)
                if latent.shape!=(model.hidden_size,):raise ValueError('World latent shape differs')
                active.append(i);latents.append(latent)
        duration=parameter.new_tensor(durations)
        seen=torch.tensor(available,device=parameter.device,dtype=torch.bool)
        prior=parameter.new_zeros((n,model.hidden_size))
        if active:
            ix=torch.tensor(active,device=parameter.device,dtype=torch.long)
            stepped=model._step(torch.stack(latents),duration.index_select(0,ix))
            prior=prior.index_copy(0,ix,stepped)
        injection=model.memory_bridge(prior)*seen[:,None]
        injection=F.pad(injection,(0,0,0,packed['ray_valid'].shape[1]-n))[None]
        raw=engine.finish_depth(model,engine.slice_prepared(prepared,t),fusion_prior=injection)
        out={k:raw[k][0,:n] for k in ('depths','visual_depth_mean_m','visual_depth_std_m','member_positions','valid_mask')}
        out['center']=raw['center'][0];out['sampling_diagnostics']=raw['sampling_diagnostics'];out['aux']=dict(raw['aux'])
        scale=packed['range_scale_m'][t]
        observation=torch.cat((packed['detector_roi_features'][t,:n].detach(),packed['ray_origins'][t,:n]/scale,
            packed['ray_directions'][t,:n],packed['visual_scale'][t,:n],out['depths'][:,None]/scale),-1)
        posterior=model.posterior(model.observation_encoder(observation),prior)
        updated={key:value for key,value in previous.items() if 0.<=timestamp-value.timestamp_s<=model.max_memory_age_s}
        for i,member in enumerate(ids):
            if member is not None and meta['ray_valid'][i]:updated[member]=MemberState(posterior[i],timestamp,frame)
        memory=WorldModelMemory(episode,updated);out['state']=memory
        out['aux'].update(group=model.group,architecture=ARCHITECTURE,radar_used=True,depth_head_count=1,
            prior_latent=prior,posterior_latent=posterior,prior_available=seen&out['valid_mask'],
            prior_positions=model.position_decoder(prior)*scale,
            rollout_positions=model._rollout(posterior,duration,scale),rollout_horizons=model.rollout_horizons,
            memory_injection=injection[0,:n],member_ids=ids)
        outputs.append(out)
    return outputs,memory


def step(model,optimizer,batch,identities,memory,settings):
    outputs,memory=forward_sequence(model,batch['inputs'],identities,memory)
    losses=[];term_sums={};squared=torch.zeros((),device=outputs[0]['depths'].device,dtype=torch.float64)
    count=torch.zeros((),device=squared.device,dtype=torch.long)
    for i,out in enumerate(outputs):
        n=batch['inputs']['counts'][i]
        if not bool(batch['targets']['eligible'][i]):continue
        targets=dict(depths=batch['targets']['depths'][i,:n],valid_mask=batch['targets']['valid_mask'][i,:n],
            member_positions=batch['targets']['positions'][i,:n].to(dtype=out['depths'].dtype))
        loss,terms=localization_loss(out,targets,huber_delta=float(model.config.get('depth_huber_delta_m',1.)),
            mse_weight=float(model.config.get('depth_loss_mse_weight',.05)),
            visual_prior_weight=float(model.config.get('visual_prior_nll_weight',.1)))
        losses.append(loss)
        for k,v in terms.items():term_sums[k]=term_sums.get(k,0)+v
        mask=out['valid_mask']&targets['valid_mask']
        error=(out['member_positions'].detach().double()-batch['targets']['positions'][i,:n]).square().sum(-1)
        squared+=(error*mask).sum();count+=mask.sum()
    if losses:
        loss=torch.stack(losses).mean();optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True);optimizer.step()
        terms={k:v/len(losses) for k,v in term_sums.items()}
    else:
        loss=outputs[0]['depths'].sum()*0;terms={};norm=loss.detach()
    return loss,terms,squared,count,norm,memory.detached()


def train(config_path,index_path,output_dir,*,device='cpu',batch_size=4,prepared_data=None,association_factory=None):
    config=ExperimentConfig.load(config_path);settings=_settings(config);settings['device']=str(device)
    base.require(batch_size==4 and settings['bptt_steps']==4,'World exact-update training requires batch_size=4')
    torch.manual_seed(settings['seed']);random.seed(settings['seed']);np.random.seed(settings['seed'])
    if str(device).startswith('cuda'):torch.cuda.set_device(device)
    started=time.monotonic()
    values,data_identity=prepared_data if prepared_data is not None else base.load_training_data(config,index_path,7)
    base.require(data_identity['full_train'],'Full TRAIN required')
    if association_factory is None:
        association_factory=_association_factory(config,base.load_manifest(config.manifest))
    identities=public_identities(values,association_factory)
    options=dict(config.raw.get('model',{}));options['newworld']=dict(config.raw.get('world',{}))
    model=build_model(7,options).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    destination,weights=base._paths(config,output_dir)
    batches=base.prepare_batches(values,4,device,False)
    groups=base.group_indices([o.episode_id for o,_ in values],4)
    for batch,indices in zip(batches,groups):
        batch['public_identities']=[identities[i] for i in indices]
        batch['episode_id']=values[indices[0]][0].episode_id
    identity=dict(config_sha256=config.sha256,manifest_sha256=data_identity['manifest_sha256'],
        architecture=ARCHITECTURE,method_id=7,center_contract=config.raw['center_contract'],
        observer_position_reference='camera_optical_center',body_navigation_is_model_input=False,
        code_sha256=_code_hashes(),experiment_config=config.raw,training_split='train',
        run_id=destination.relative_to(config.output_root).as_posix(),output_dir=str(destination),weights_dir=str(weights),
        calibration_sha256=getattr(association_factory,'calibration_sha256',None),
        association='TRAIN-frozen normal visual tracker; never GT IDs',
        engine=dict(name='batched_evidence_causal_world_v1',optimizer_frames=4,
            original_update_schedule_preserved=True,episode_tail_flush_preserved=True,
            frame_equal_loss=True,fp32=True,trainer_sha256=file_sha256(__file__),
            batch_model_sha256=file_sha256(engine.__file__),utility_sha256=file_sha256(base.__file__)),
        fixed_label_identity=data_identity,inference_uses_gt_ids=False,future_rollout_supervision=False,
        synthetic_only=bool(data_identity.get('synthetic_only',False)))
    write_json(destination/'identity.json',identity)
    history=[];updates=0;train_started=time.monotonic();total_frames=len(values)
    print(f'TRAIN_WORLD_ACCEL START epochs={settings["epochs"]} frames={total_frames} updates_per_epoch={len(batches)} '
          f'preload_s={train_started-started:.3f} device={device}',flush=True)
    for epoch in range(settings['epochs']):
        epoch_started=time.monotonic();warmup=min(1.,(epoch+1)/max(1,settings['warmup_epochs']))
        for g in optimizer.param_groups:g['lr']=settings['learning_rate']*warmup
        totals={};sse=torch.zeros((),device=device,dtype=torch.float64);members=torch.zeros((),device=device,dtype=torch.long)
        frames=matched_frames=matched_members=0;memory=None;episode=None
        for batch in batches:
            if batch['episode_id']!=episode:memory=None;episode=batch['episode_id']
            loss,terms,error,count,norm,memory=step(model,optimizer,batch,batch['public_identities'],memory,settings)
            if batch['matched_frames']:updates+=1
            frames+=batch['frames'];matched_frames+=batch['matched_frames'];matched_members+=batch['matched_members']
            sse+=error;members+=count
            for k,v in terms.items():totals[k]=totals.get(k,0)+v.detach().double()*batch['matched_frames']
            if updates%settings['log_every_updates']==0:
                elapsed=time.monotonic()-train_started;done=epoch*total_frames+frames
                eta=((settings['epochs']-epoch)*total_frames-frames)*elapsed/max(1,done)
                rmse=float(torch.sqrt(sse/members.clamp_min(1)));now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
                print(f'[{now}] [PROGRESS] TRAIN_WORLD_ACCEL G7 epoch={epoch+1}/{settings["epochs"]} frame={frames}/{total_frames} '
                    f'update={updates} loss={float(loss.detach()):.6f} TRAIN_ONLINE_FIXED_RMSE={rmse:.4f}m '
                    f'已用={base.hms(elapsed)} 预计剩余={base.hms(eta)}',flush=True)
                write_json(destination/'progress.json',dict(schema='rvloc.training_progress.v1',status='training',method_id=7,
                    epoch=epoch+1,epochs=settings['epochs'],frames=frames,optimizer_updates=updates,last_loss=float(loss.detach()),
                    online_fixed_train_rmse_m=rmse,rmse_scope='online_epoch_so_far_fixed_members',elapsed_s=elapsed,eta_s=eta,device=str(device)))
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        base.require(matched_frames>0,'No world TRAIN supervision')
        row=dict(epoch=epoch+1,frames=frames,matched_frames=matched_frames,matched_members=matched_members,
            unassigned_visual_members=sum(member is None for item in identities for member in item['member_ids']),
            optimizer_updates=updates,learning_rate=optimizer.param_groups[0]['lr'],
            loss={k:float(v)/matched_frames for k,v in totals.items()},online_fixed_train_rmse_m=float(torch.sqrt(sse/members.clamp_min(1))),
            epoch_elapsed_s=time.monotonic()-epoch_started,engine=identity['engine']['name'])
        history.append(row);model.core.trained=True
        metadata=dict(identity,completed_epochs=epoch+1,optimizer_updates=updates,history=history,training_settings=settings,
            observation_identity_sha256=data_identity['observation_identity_sha256'],rng_state=base._rng_state(device))
        save_checkpoint(model,weights/f'epoch_{epoch+1:04d}.pt',optimizer=optimizer,metadata=metadata)
        write_json(destination/'history.json',history)
        print(f'TRAIN_WORLD_ACCEL G7 EPOCH {epoch+1}/{settings["epochs"]} COMPLETE loss={row["loss"]["total"]:.6f} '
              f'TRAIN_ONLINE_FIXED_RMSE={row["online_fixed_train_rmse_m"]:.4f}m epoch_s={row["epoch_elapsed_s"]:.3f}',flush=True)
    save_checkpoint(model,weights/'model.pt',optimizer=optimizer,metadata=metadata)
    receipt=dict(schema='rvloc.accelerated_training.v1',status='trained',method_id=7,group=model.group,
        architecture=ARCHITECTURE,checkpoint=str(weights/'model.pt'),output_dir=str(destination),weights_dir=str(weights),
        epochs=settings['epochs'],completed_epochs=settings['epochs'],new_epochs=settings['epochs'],optimizer_updates=updates,
        history=history,training_identity=identity,elapsed_s=time.monotonic()-started,
        normal_inference_uses_gt_ids=False,future_rollout_supervision=False)
    write_json(destination/'receipt.json',receipt)
    write_json(destination/'progress.json',dict(schema='rvloc.training_progress.v1',status='completed',method_id=7,
        epoch=settings['epochs'],epochs=settings['epochs'],frames=total_frames,optimizer_updates=updates,
        mean_loss=history[-1]['loss']['total'],online_fixed_train_rmse_m=history[-1]['online_fixed_train_rmse_m'],
        elapsed_s=time.monotonic()-train_started,eta_s=0.))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--index',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--method',default='7',choices=['7']);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();print(json.dumps(train(a.config,a.index,a.output_dir,device=a.device,batch_size=a.batch_size),indent=2),flush=True)


if __name__=='__main__':main()
