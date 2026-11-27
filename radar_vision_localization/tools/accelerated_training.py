"""Fixed-label, true frame-batched G2/G3 training and epoch-boundary resume.

The frozen src implementation and prior result/weight runs are never modified.
Default batches contain the same <=4 sequential frames, including episode-tail
flushes, as the original optimizer schedule. Larger batches are explicitly a
changed optimization protocol. No GT identity enters model inputs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from datetime import datetime
import hashlib
from io import BytesIO
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

from rvloc.config import ExperimentConfig, external_path, resolve_method
from rvloc.data import load_manifest
from rvloc.models import build_model, save_checkpoint
from rvloc.models.checkpoint import SCHEMA, ARCHITECTURE
from rvloc.runtime import file_sha256, write_json, source_identity
from rvloc.supervision import match_direct_ray_labels
from rvloc.training import _settings, _code_hashes


@lru_cache(maxsize=None)
def sibling(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('accelerated_training_'+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def require(condition,message):
    if not condition:raise ValueError(message)


def hms(value):
    if value is None:return 'pending'
    n=max(0,int(round(value)));return f'{n//3600:02d}:{n//60%60:02d}:{n%60:02d}'


def group_indices(episodes,batch_size=4):
    """Original <=4-frame update boundaries; larger mode is a new protocol."""
    require(type(batch_size) is int and batch_size>0,'batch_size must be positive integer')
    groups=[];current=[];previous=None
    for i,episode in enumerate(episodes):
        if batch_size==4 and previous is not None and episode!=previous and current:
            groups.append(current);current=[]
        current.append(i);previous=episode
        if len(current)==batch_size:groups.append(current);current=[]
    if current:groups.append(current)
    return groups


def fixed_frame_loss(output, targets, *, huber_delta=1., mse_weight=.05, visual_prior_weight=.1):
    """Frame-equal loss, exactly matching mean(original per-frame losses)."""
    valid=output['valid_mask']&targets['valid_mask']
    error=output['depths']-targets['depths']
    absolute=error.abs()
    huber=torch.where(absolute<huber_delta,.5*error.square(),huber_delta*(absolute-.5*huber_delta))
    variance=output['visual_depth_std_m'].square()
    # Match GaussianNLL's forward clamp without zeroing its variance gradient.
    variance=variance+(variance.clamp_min(1e-6)-variance).detach()
    nll=.5*((output['visual_depth_mean_m']-targets['depths']).square()/variance+variance.log()+math.log(2*math.pi))
    den=valid.sum(1).clamp_min(1)
    eligible=targets['eligible']
    count=targets['matched_frames']
    def frame_mean(values):
        return ((values*valid).sum(1)/den*eligible).sum()/max(1,count)
    terms={'depth_huber':frame_mean(huber),'depth_mse':frame_mean(error.square()),
           'visual_prior_nll':frame_mean(nll)}
    loss=terms['depth_huber']+mse_weight*terms['depth_mse']+visual_prior_weight*terms['visual_prior_nll']
    terms['total']=loss
    return loss,terms


def _rng_state(device='cpu'):
    value=dict(torch_cpu=torch.get_rng_state(),python=random.getstate())
    state=np.random.get_state()
    value['numpy']=dict(name=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached=state[4])
    if str(device).startswith('cuda'):value['torch_cuda_current']=torch.cuda.get_rng_state(device)
    return value


def _restore_rng(value,device='cpu'):
    torch.set_rng_state(value['torch_cpu']);random.setstate(value['python'])
    n=value['numpy'];np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached']))
    if 'torch_cuda_current' in value and str(device).startswith('cuda'):
        torch.cuda.set_rng_state(value['torch_cuda_current'],device)


def resume_model(checkpoint_path,config,manifest_sha,method,device,settings):
    raw=Path(checkpoint_path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    payload=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    metadata=payload.get('metadata',{})
    require(payload.get('schema')==SCHEMA and payload.get('architecture')==ARCHITECTURE,'Resume architecture/schema differs')
    require(metadata.get('method_id')==method and metadata.get('manifest_sha256')==manifest_sha,'Resume method/data differs')
    require(metadata.get('center_contract')=='observer_scatter_at_camera_optical_center'
            and metadata.get('observer_position_reference')=='camera_optical_center'
            and metadata.get('body_navigation_is_model_input') is False,'Resume optical-centre contract differs')
    require('optimizer_state_dict' in payload,'Resume requires the original AdamW state')
    old=metadata.get('training_settings',{})
    for name in ('learning_rate','weight_decay','seed','warmup_epochs','bptt_steps','grad_clip'):
        if name in old:require(old[name]==settings[name],'Resume training setting differs: '+name)
    model=build_model(method,payload['config']).to(device)
    model.load_state_dict(payload['state_dict'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    restored='rng_state' in metadata
    if restored:_restore_rng(metadata['rng_state'],device)
    epoch=metadata.get('completed_epochs',0)
    require(type(epoch) is int and 0<epoch<=settings['epochs'],'Invalid completed resume epoch')
    return model,optimizer,metadata,dict(path=str(Path(checkpoint_path).resolve()),sha256=digest,
        completed_epochs=epoch,optimizer_restored=True,rng_restored=restored,
        continuity='epoch-boundary optimizer/model restoration; batched floating-point kernels need not be bitwise identical',
        legacy_rng_note=None if restored else 'Legacy checkpoint lacks RNG state; inputs remain ordered and network dropout is zero; bitwise continuation is not certified')


def load_training_data(config,index_path,method,*,limit_frames=None,verify_equivalence=True):
    """Read/authenticate TRAIN only; save no trainable encoded radar maps."""
    backend=sibling('checkpoint_train_rmse')
    manifest=load_manifest(config.manifest)
    index_path,index_sha,refs,expected=backend._index(index_path,manifest)
    options={'modality':'rgb' if method==2 else 'rgb_radar'}
    if method in (3,7):options['prepared_only']=True
    values=[];source_hash=hashlib.sha256();matched=0
    for observation in manifest.observations('train',**options):
        key=observation.episode_id,observation.frame_idx
        ref=refs[key];path=(index_path.parent/ref['path']).resolve()
        require(path.is_relative_to(index_path.parent),'Fixed frame escaped index root')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Fixed frame SHA changed')
        frozen=json.loads(raw);fixed=backend._bind(observation,frozen,expected[key])
        require(observation.iq is None,'Raw IQ must not be retained by neural training')
        if verify_equivalence:
            old=match_direct_ray_labels(observation.features,fixed['truth'],config.raw.get('model',{}))
            require(np.array_equal(old['rows'],fixed['rows']) and np.array_equal(old['cols'],fixed['columns']),
                    'Frozen labels differ from the original training assignment')
        selected=fixed['truth'][fixed['columns']]
        origins=observation.features['ray_origins'][fixed['rows']]
        dirs=observation.features['ray_directions'][fixed['rows']]
        depths=((selected-origins)*dirs).sum(-1)
        for pair,depth in zip(frozen['training_equivalent_assignment'],depths):
            require(abs(pair['target_depth_m']-depth)<1e-8,'Frozen depth target differs from original projection')
        fixed['depths']=depths
        values.append((observation,fixed));matched+=len(fixed['rows'])
        source_hash.update(json.dumps({'episode':observation.episode_id,'frame':observation.frame_idx,
            'source_hashes':observation.source_hashes},sort_keys=True,allow_nan=False).encode())
        if len(values)%1000==0:print(f'PRELOAD method={method} frames={len(values)} matched={matched}',flush=True)
        if limit_frames is not None and len(values)>=limit_frames:break
    full=limit_frames is None or len(values)==len(expected)
    if full:require(len(values)==len(expected),'TRAIN coverage differs')
    return values,dict(manifest_sha256=manifest.sha256,fixed_index_sha256=index_sha,
        observation_identity_sha256=source_hash.hexdigest(),frames=len(values),matched_members=matched,
        full_train=full,full_assignment_equivalence_verified=bool(verify_equivalence and full),
        native_iq_read=False,dev_labels_read=False,test_read=False)


def prepare_batches(values,batch_size,device,visual):
    engine=sibling('accelerated_batch_model')
    result=[]
    for indices in group_indices([o.episode_id for o,_ in values],batch_size):
        selected=[values[i] for i in indices]
        packed=engine.pack_features([o.features for o,_ in selected],device=device,visual=visual,validate=True)
        b=len(selected);n=max(1,max(f['detection_count'] for _,f in selected))
        depths=torch.zeros((b,n),device=device);positions=torch.zeros((b,n,3),device=device,dtype=torch.float64)
        mask=torch.zeros((b,n),device=device,dtype=torch.bool)
        matched_frames=matched_members=0
        eligible=[]
        for j,(_,fixed) in enumerate(selected):
            rows=torch.as_tensor(fixed['rows'],device=device)
            if len(rows):
                depths[j,rows]=torch.as_tensor(fixed['depths'],device=device,dtype=depths.dtype)
                positions[j,rows]=torch.as_tensor(fixed['truth'][fixed['columns']],device=device,dtype=torch.float64)
                mask[j,rows]=True;matched_frames+=1;matched_members+=len(rows)
            eligible.append(bool(len(rows)))
        result.append(dict(inputs=packed,targets=dict(depths=depths,positions=positions,valid_mask=mask,
            eligible=torch.tensor(eligible,device=device),matched_frames=matched_frames),
            frames=b,matched_frames=matched_frames,matched_members=matched_members,
            detection_count=sum(f['detection_count'] for _,f in selected)))
    return result


def step(model,optimizer,batch,settings):
    engine=sibling('accelerated_batch_model')
    output=engine.forward_batch(model,batch['inputs'])
    loss,terms=fixed_frame_loss(output,batch['targets'],
        huber_delta=float(model.config.get('depth_huber_delta_m',1.)),
        mse_weight=float(model.config.get('depth_loss_mse_weight',.05)),
        visual_prior_weight=float(model.config.get('visual_prior_nll_weight',.1)))
    if batch['matched_frames']:
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True)
        optimizer.step()
    else:norm=loss.detach()*0
    mask=output['valid_mask']&batch['targets']['valid_mask']
    squared=(output['member_positions'].detach().double()-batch['targets']['positions']).square().sum(-1)
    return loss,terms,(squared*mask).sum(),mask.sum(),norm


def _paths(config,output_dir):
    if output_dir is None:raise ValueError('An explicit independent output-dir is required')
    destination=external_path(output_dir,field='accelerated output',base=config.path.parent)
    require(destination!=config.output_root and destination.is_relative_to(config.output_root),'Output must be an independent child of configured output_root')
    weights=config.weights_root/destination.relative_to(config.output_root)
    if destination.exists() or weights.exists():raise FileExistsError('Accelerated results or weights already exist')
    destination.mkdir(parents=True);weights.mkdir(parents=True)
    return destination,weights


def train(config_path,method,index_path,output_dir,*,device='cpu',batch_size=4,resume_checkpoint=None,
          prepared_data=None):
    config=ExperimentConfig.load(config_path);method=int(method)
    require(method in (2,3),'Accelerated independent training supports G2/G3 only')
    require(type(batch_size) is int and batch_size>0,'batch_size must be positive integer')
    settings=_settings(config);settings['device']=str(device)
    require(batch_size!=4 or settings['bptt_steps']==4,'Exact-update4 requires original truncation_frames=4')
    torch.manual_seed(settings['seed']);random.seed(settings['seed']);np.random.seed(settings['seed'])
    if str(device).startswith('cuda'):torch.cuda.set_device(device)
    started=time.monotonic()
    values,data_identity=prepared_data if prepared_data is not None else load_training_data(config,index_path,method)
    require(data_identity['full_train'],'Formal training requires full TRAIN coverage')
    if resume_checkpoint:
        model,optimizer,old,resume=resume_model(resume_checkpoint,config,data_identity['manifest_sha256'],method,device,settings)
        if old.get('observation_identity_sha256'):
            require(old['observation_identity_sha256']==data_identity['observation_identity_sha256'],'Resume source ordering/identities changed')
        first_epoch=resume['completed_epochs'];updates=old['optimizer_updates'];history=list(old.get('history',[]))
    else:
        model_options=dict(config.raw.get('model',{}));model_options['newworld']=dict(config.raw.get('world',{}))
        model=build_model(method,model_options).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        old={};resume=None;first_epoch=updates=0;history=[]
    model.train()
    destination,weights=_paths(config,output_dir)
    engine_path=Path(__file__).with_name('accelerated_batch_model.py')
    identity=dict(config_sha256=config.sha256,manifest_sha256=data_identity['manifest_sha256'],
        architecture=ARCHITECTURE,method_id=method,center_contract=config.raw['center_contract'],
        observer_position_reference='camera_optical_center',body_navigation_is_model_input=False,
        code_sha256=_code_hashes(),experiment_config=config.raw,training_split='train',
        run_id=destination.relative_to(config.output_root).as_posix(),output_dir=str(destination),weights_dir=str(weights),
        engine=dict(name='true_frame_batch_v1',trainer_sha256=file_sha256(__file__),batch_model_sha256=file_sha256(engine_path),
            optimizer_frames=batch_size,original_update_schedule_preserved=batch_size==4,
            episode_tail_flush_preserved=batch_size==4,frame_equal_loss=True,fp32=True,
            protocol_change=None if batch_size==4 else 'Larger optimizer batch changes update count and crosses episode boundaries'),
        fixed_label_identity=data_identity,resume=resume,
        synthetic_only=bool(data_identity.get('synthetic_only',False)))
    write_json(destination/'identity.json',identity)
    pack_started=time.monotonic();batches=prepare_batches(values,batch_size,device,method==2)
    packing_s=time.monotonic()-pack_started
    total_frames=len(values);historical_elapsed=sum(float(r.get('epoch_elapsed_s',0.)) for r in history)
    train_started=time.monotonic()
    print(f'TRAIN_ACCEL G{method} START epoch={first_epoch}/{settings["epochs"]} true_batch={batch_size} '
          f'batches_per_epoch={len(batches)} device={device} preload_s={train_started-started:.3f} packing_s={packing_s:.3f}',flush=True)
    for epoch in range(first_epoch,settings['epochs']):
        epoch_started=time.monotonic()
        warmup=min(1.,(epoch+1)/max(1,settings['warmup_epochs']))
        for g in optimizer.param_groups:g['lr']=settings['learning_rate']*warmup
        totals={k:torch.zeros((),device=device,dtype=torch.float64) for k in ('depth_huber','depth_mse','visual_prior_nll','total')}
        squared_sum=torch.zeros((),device=device,dtype=torch.float64);metric_count=torch.zeros((),device=device,dtype=torch.long)
        frames=matched_frames=matched_members=0
        for batch in batches:
            loss,terms,squared,count,norm=step(model,optimizer,batch,settings)
            if batch['matched_frames']:updates+=1
            frames+=batch['frames'];matched_frames+=batch['matched_frames'];matched_members+=batch['matched_members']
            squared_sum+=squared;metric_count+=count
            for k in totals:totals[k]+=terms[k].detach().double()*batch['matched_frames']
            if updates%settings['log_every_updates']==0:
                if str(device).startswith('cuda'):torch.cuda.synchronize(device)
                elapsed=time.monotonic()-train_started
                processed=(epoch-first_epoch)*total_frames+frames
                remaining=(settings['epochs']-epoch)*total_frames-frames
                eta=remaining*elapsed/max(1,processed)
                rmse=float(torch.sqrt(squared_sum/metric_count.clamp_min(1)))
                now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
                print(f'[{now}] [PROGRESS] TRAIN_ACCEL G{method} epoch={epoch+1}/{settings["epochs"]} '
                    f'frame={frames}/{total_frames} update={updates} loss={float(loss.detach()):.6f} '
                    f'TRAIN_ONLINE_FIXED_RMSE={rmse:.4f}m 已用={hms(elapsed)} 预计剩余={hms(eta)}',flush=True)
                write_json(destination/'progress.json',dict(schema='rvloc.training_progress.v1',status='training',
                    method_id=method,epoch=epoch+1,epochs=settings['epochs'],frames=frames,optimizer_updates=updates,
                    last_loss=float(loss.detach()),online_fixed_train_rmse_m=rmse,rmse_scope='online_epoch_so_far_fixed_members',
                    elapsed_s=elapsed,eta_s=eta,historical_elapsed_s=historical_elapsed,device=str(device)))
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        require(matched_frames>0,'No TRAIN supervision')
        row=dict(epoch=epoch+1,frames=frames,matched_frames=matched_frames,matched_members=matched_members,
            optimizer_updates=updates,learning_rate=optimizer.param_groups[0]['lr'],
            loss={k:float(v)/matched_frames for k,v in totals.items()},
            online_fixed_train_rmse_m=float(torch.sqrt(squared_sum/metric_count.clamp_min(1))),
            epoch_elapsed_s=time.monotonic()-epoch_started,engine='true_frame_batch_v1')
        history.append(row);model.core.trained=True
        metadata=dict(identity,completed_epochs=epoch+1,optimizer_updates=updates,history=history,
            training_settings=settings,observation_identity_sha256=data_identity['observation_identity_sha256'],
            inference_uses_gt_ids=False,rng_state=_rng_state(device))
        save_checkpoint(model,weights/f'epoch_{epoch+1:04d}.pt',optimizer=optimizer,metadata=metadata)
        write_json(destination/'history.json',history)
        print(f'TRAIN_ACCEL G{method} EPOCH {epoch+1}/{settings["epochs"]} COMPLETE '
            f'loss={row["loss"]["total"]:.6f} TRAIN_ONLINE_FIXED_RMSE={row["online_fixed_train_rmse_m"]:.4f}m '
            f'epoch_s={row["epoch_elapsed_s"]:.3f} checkpoint={weights/f"epoch_{epoch+1:04d}.pt"}',flush=True)
    require(bool(history),'No completed epoch available')
    if first_epoch==settings['epochs']:
        model.core.trained=True
        metadata=dict(old,**identity,rng_state=_rng_state(device))
    save_checkpoint(model,weights/'model.pt',optimizer=optimizer,metadata=metadata)
    receipt=dict(schema='rvloc.accelerated_training.v1',status='trained',method_id=method,group=model.group,
        architecture=ARCHITECTURE,checkpoint=str(weights/'model.pt'),output_dir=str(destination),weights_dir=str(weights),
        epochs=settings['epochs'],completed_epochs=settings['epochs'],new_epochs=settings['epochs']-first_epoch,
        optimizer_updates=updates,history=history,resume=resume,training_identity=identity,
        elapsed_s=time.monotonic()-started,packing_s=packing_s,training_split='train',normal_inference_uses_gt_ids=False)
    write_json(destination/'receipt.json',receipt)
    write_json(destination/'progress.json',dict(schema='rvloc.training_progress.v1',status='completed',
        method_id=method,epoch=settings['epochs'],epochs=settings['epochs'],frames=total_frames,optimizer_updates=updates,
        mean_loss=history[-1]['loss']['total'],online_fixed_train_rmse_m=history[-1].get('online_fixed_train_rmse_m'),
        elapsed_s=time.monotonic()-train_started,eta_s=0.))
    return receipt


def benchmark(config_path,method,index_path,*,device='cpu',batch_size=4,updates=40,warmup_updates=5,
              resume_checkpoint=None):
    require(type(updates) is int and updates>0 and type(warmup_updates) is int and warmup_updates>=0,'Invalid benchmark budget')
    config=ExperimentConfig.load(config_path);method=int(method);settings=_settings(config)
    torch.manual_seed(settings['seed'])
    limit=(updates+warmup_updates)*batch_size
    values,data=load_training_data(config,index_path,method,limit_frames=limit,verify_equivalence=True)
    if resume_checkpoint:model,optimizer,_,_=resume_model(resume_checkpoint,config,data['manifest_sha256'],method,device,settings)
    else:
        model=build_model(method,config.raw.get('model',{})).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    model.train();batches=prepare_batches(values,batch_size,device,method==2)
    for i in range(warmup_updates):step(model,optimizer,batches[i%len(batches)],settings)
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    started=time.perf_counter();processed=0
    for i in range(updates):
        batch=batches[(i+warmup_updates)%len(batches)];step(model,optimizer,batch,settings);processed+=batch['frames']
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    return dict(schema='rvloc.accelerated_training_benchmark.v1',benchmark_only=True,formal_checkpoint_written=False,
        method_id=method,device=str(device),batch_size=batch_size,updates=updates,frames=processed,
        elapsed_s=elapsed,frames_per_s=processed/elapsed,updates_per_s=updates/elapsed,
        full_epoch_evaluated=False,data=data,source=source_identity())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--method',type=int,choices=(2,3),required=True)
    p.add_argument('--index',required=True);p.add_argument('--output-dir');p.add_argument('--resume-checkpoint')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--benchmark-updates',type=int,default=0);p.add_argument('--warmup-updates',type=int,default=5)
    a=p.parse_args(argv)
    if a.benchmark_updates:
        result=benchmark(a.config,a.method,a.index,device=a.device,batch_size=a.batch_size,
            updates=a.benchmark_updates,warmup_updates=a.warmup_updates,resume_checkpoint=a.resume_checkpoint)
        if a.output_dir:
            output=external_path(a.output_dir,field='benchmark output');output.mkdir(parents=True,exist_ok=False)
            write_json(output/'benchmark.json',result)
    else:result=train(a.config,a.method,a.index,a.output_dir,device=a.device,batch_size=a.batch_size,resume_checkpoint=a.resume_checkpoint)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
