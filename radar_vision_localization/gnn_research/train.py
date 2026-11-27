"""Controlled, fixed-label TRAIN-only MCE-GNN experiments.

The active seven-method sources and tools are imported read-only. All graph
variants use the same loss/update schedule; labels never determine graph edges.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'tools') not in sys.path:sys.path.insert(0,str(PROJECT/'tools'))
import accelerated_training as base
from rvloc.config import ExperimentConfig
from rvloc.data import load_manifest
from rvloc.runtime import file_sha256,write_json,source_identity,jsonable
from rvloc.training import _settings

SCHEMA='rvloc.mce_gnn_checkpoint.v1'
ARCHITECTURE='vision_conditioned_shared_radar_gnn_v1'
VARIANTS=('shared','isolated','shuffled')
CORE_FILES=('model.py','graph.py','train.py')
UTILITY_FILES=('accelerated_training.py','accelerated_batch_model.py','checkpoint_train_rmse.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def training_sources():
    folder=Path(__file__).parent
    return dict(graph={name:file_sha256(folder/name) for name in CORE_FILES},
        utilities={name:file_sha256(PROJECT/'tools'/name) for name in UTILITY_FILES},
        frozen_base=source_identity())


def build(options,variant,rf):
    from gnn_research.model import build_graph_model
    return build_graph_model(options,variant=variant,rf=rf)


def forward(model,packed):
    from gnn_research.model import forward_batch
    return forward_batch(model,packed)


def _checkpoint_safe(value):
    """Keep torch/RNG state exact while removing NumPy pickle dependencies."""
    if isinstance(value,np.ndarray):return _checkpoint_safe(value.tolist())
    if isinstance(value,np.generic):return _checkpoint_safe(value.item())
    if isinstance(value,dict):return {key:_checkpoint_safe(item) for key,item in value.items()}
    if isinstance(value,tuple):return tuple(_checkpoint_safe(item) for item in value)
    if isinstance(value,list):return [_checkpoint_safe(item) for item in value]
    if isinstance(value,Path):return str(value)
    return value


def save_graph_checkpoint(model,path,*,variant,optimizer=None,metadata=None):
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    value=dict(schema=SCHEMA,architecture=ARCHITECTURE,variant=variant,
        config=_checkpoint_safe(model.config),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        metadata=_checkpoint_safe(dict(metadata or {})))
    if optimizer is not None:value['optimizer_state_dict']=optimizer.state_dict()
    fd,temporary=tempfile.mkstemp(prefix='.checkpoint-',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as stream:torch.save(value,stream)
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_graph_checkpoint(path,*,device='cpu',expected_variant=None):
    raw=Path(path).read_bytes();payload=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    require(payload.get('schema')==SCHEMA and payload.get('architecture')==ARCHITECTURE,'Not an MCE-GNN checkpoint')
    variant=payload['variant'];require(variant in VARIANTS,'Unknown graph variant')
    if expected_variant is not None:require(variant==expected_variant,'Graph variant differs')
    options=dict(payload['config']);rf=options.pop('graph_rf',options.pop('rf',None))
    # Graph wrapper may store graph_variant alongside the baseline architecture.
    options.pop('graph_variant',None)
    model=build(options,variant,rf);model.load_state_dict(payload['state_dict'],strict=True)
    model.core.trained=payload.get('metadata',{}).get('optimizer_updates',0)>0
    model.to(device).eval()
    return model,payload,hashlib.sha256(raw).hexdigest()


def step(model,optimizer,batch,settings):
    output=forward(model,batch['inputs'])
    loss,terms=base.fixed_frame_loss(output,batch['targets'],
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
    return loss,terms,(squared*mask).sum(),mask.sum(),norm,output.get('graph_diagnostics',{})


def train(config_path,variant,index_path,output_dir,*,device='cpu',batch_size=4,resume_checkpoint=None,
          prepared_data=None,model=None):
    config=ExperimentConfig.load(config_path);require(variant in VARIANTS,'Unknown graph variant')
    settings=_settings(config);settings['device']=str(device)
    require(batch_size==4 and settings['bptt_steps']==4,'Controlled graph experiment preserves four-frame optimizer updates')
    require(config.raw.get('gnn',{}).get('candidate_loss_weight',0.)==0.,'First controlled graph experiment keeps candidate auxiliary loss disabled')
    torch.manual_seed(settings['seed']);random.seed(settings['seed']);np.random.seed(settings['seed'])
    if str(device).startswith('cuda'):torch.cuda.set_device(device)
    started=time.monotonic()
    sources=training_sources()
    values,data_identity=prepared_data if prepared_data is not None else base.load_training_data(config,index_path,3)
    require(data_identity['full_train'],'Formal graph training requires the full TRAIN split')
    manifest=load_manifest(config.manifest) if prepared_data is None else None
    rf=manifest.rf.as_dict() if manifest is not None else config.raw.get('gnn',{}).get('rf')
    options=dict(config.raw.get('model',{}));options['gnn']=dict(config.raw.get('gnn',{}))
    if model is None:model=build(options,variant,rf)
    model.to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    history=[];updates=first_epoch=0;continuation=None
    if resume_checkpoint is not None:
        raw=Path(resume_checkpoint).read_bytes();payload=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
        old=payload.get('metadata',{})
        require(payload.get('schema')==SCHEMA and payload.get('architecture')==ARCHITECTURE and payload.get('variant')==variant,'Resume graph architecture/variant differs')
        require(payload.get('config')==model.config,'Resume graph/model configuration differs')
        require(old.get('manifest_sha256')==data_identity['manifest_sha256'],'Resume data identity differs')
        require(old.get('fixed_label_identity',{}).get('fixed_index_sha256')==data_identity['fixed_index_sha256'],'Resume fixed labels differ')
        require(old.get('training_sources')==sources,'Resume graph sources differ')
        for name in ('learning_rate','weight_decay','seed','warmup_epochs','bptt_steps','grad_clip'):
            require(old['training_settings'][name]==settings[name],'Resume optimizer setting differs: '+name)
        model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer_state_dict'])
        base._restore_rng(old['rng_state'],device)
        first_epoch=old['completed_epochs'];updates=old['optimizer_updates'];history=list(old['history'])
        continuation=dict(path=str(Path(resume_checkpoint).resolve()),sha256=hashlib.sha256(raw).hexdigest(),completed_epochs=first_epoch,optimizer_and_rng_restored=True)
    require(first_epoch<=settings['epochs'],'Resume checkpoint exceeds frozen epoch budget')
    destination,weights=base._paths(config,output_dir)
    parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    common={name:t.detach().cpu() for name,t in model.state_dict().items() if not any(key in name for key in ('local_query','graph_'))}
    common_hash=hashlib.sha256()
    for name,value in sorted(common.items()):common_hash.update(name.encode());common_hash.update(value.numpy().tobytes())
    identity=dict(schema='rvloc.mce_gnn_training_identity.v1',config_sha256=config.sha256,
        manifest_sha256=data_identity['manifest_sha256'],fixed_label_identity=data_identity,
        architecture=ARCHITECTURE,variant=variant,training_sources=sources,
        experiment_config=config.raw,training_settings=settings,parameter_count=parameter_count,
        common_initial_state_sha256=common_hash.hexdigest() if not resume_checkpoint else None,
        initialization='same seed constructs original G3 backbone first, then graph parameters; common parameter hash recorded',
        comparison_to_baseline='same TRAIN/seed/80 epochs/loss/update4; graph parameter addition changes architecture and total initialization RNG consumption',
        center_contract=config.raw['center_contract'],observer_position_reference='camera_optical_center',
        body_navigation_is_model_input=False,labels_only_in_loss=True,inference_uses_gt_ids=False,
        candidate_loss_weight=0.,candidate_generation='unchanged visual mean/std multipliers, no GT or discrete final depth',
        training_split='train',run_id=destination.relative_to(config.output_root).as_posix(),
        output_dir=str(destination),weights_dir=str(weights),resume=continuation,
        synthetic_only=bool(data_identity.get('synthetic_only',False)))
    write_json(destination/'identity.json',identity)
    packing_started=time.monotonic();batches=base.prepare_batches(values,4,device,False);packing_s=time.monotonic()-packing_started
    total_frames=len(values);train_started=time.monotonic()
    print(f'MCE_TRAIN {variant} START epoch={first_epoch}/{settings["epochs"]} batch=4 frames={total_frames} parameters={parameter_count} device={device} preload_s={train_started-started:.3f}',flush=True)
    for epoch in range(first_epoch,settings['epochs']):
        epoch_started=time.monotonic();warmup=min(1.,(epoch+1)/max(1,settings['warmup_epochs']))
        for group in optimizer.param_groups:group['lr']=settings['learning_rate']*warmup
        totals={key:torch.zeros((),device=device,dtype=torch.float64) for key in ('depth_huber','depth_mse','visual_prior_nll','total')}
        sse=torch.zeros((),device=device,dtype=torch.float64);members=torch.zeros((),device=device,dtype=torch.long)
        frames=matched_frames=matched_members=0
        for batch in batches:
            loss,terms,error,count,norm,diagnostic=step(model,optimizer,batch,settings)
            if batch['matched_frames']:updates+=1
            frames+=batch['frames'];matched_frames+=batch['matched_frames'];matched_members+=batch['matched_members']
            sse+=error;members+=count
            for key in totals:totals[key]+=terms[key].detach().double()*batch['matched_frames']
            if updates%settings['log_every_updates']==0:
                elapsed=time.monotonic()-train_started;done=(epoch-first_epoch)*total_frames+frames
                eta=((settings['epochs']-epoch)*total_frames-frames)*elapsed/max(1,done)
                rmse=float(torch.sqrt(sse/members.clamp_min(1)));stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
                print(f'[{stamp}] [PROGRESS] MCE_{variant} epoch={epoch+1}/{settings["epochs"]} frame={frames}/{total_frames} update={updates} loss={float(loss.detach()):.6f} TRAIN_ONLINE_FIXED_RMSE={rmse:.4f}m 已用={base.hms(elapsed)} 预计剩余={base.hms(eta)}',flush=True)
                write_json(destination/'progress.json',dict(status='training',variant=variant,epoch=epoch+1,epochs=settings['epochs'],frames=frames,optimizer_updates=updates,last_loss=float(loss.detach()),online_fixed_train_rmse_m=rmse,rmse_scope='online_epoch_so_far_fixed_members',elapsed_s=elapsed,eta_s=eta,device=str(device)))
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        require(matched_frames>0,'No matched TRAIN supervision')
        row=dict(epoch=epoch+1,frames=frames,matched_frames=matched_frames,matched_members=matched_members,
            optimizer_updates=updates,learning_rate=optimizer.param_groups[0]['lr'],loss={k:float(v)/matched_frames for k,v in totals.items()},
            online_fixed_train_rmse_m=float(torch.sqrt(sse/members.clamp_min(1))),epoch_elapsed_s=time.monotonic()-epoch_started)
        history.append(row);model.core.trained=True
        require(training_sources()==sources,'Training core sources changed during this experiment')
        metadata=dict(identity,completed_epochs=epoch+1,optimizer_updates=updates,history=history,rng_state=base._rng_state(device))
        save_graph_checkpoint(model,weights/f'epoch_{epoch+1:04d}.pt',variant=variant,optimizer=optimizer,metadata=metadata)
        write_json(destination/'history.json',history)
        print(f'MCE_TRAIN {variant} EPOCH {epoch+1}/{settings["epochs"]} COMPLETE loss={row["loss"]["total"]:.6f} TRAIN_ONLINE_FIXED_RMSE={row["online_fixed_train_rmse_m"]:.4f}m epoch_s={row["epoch_elapsed_s"]:.3f}',flush=True)
    require(bool(history),'No completed training epoch')
    if first_epoch==settings['epochs']:
        metadata=dict(identity,completed_epochs=first_epoch,optimizer_updates=updates,history=history,rng_state=base._rng_state(device))
    save_graph_checkpoint(model,weights/'model.pt',variant=variant,optimizer=optimizer,metadata=metadata)
    receipt=dict(schema='rvloc.mce_gnn_training.v1',status='trained',architecture=ARCHITECTURE,variant=variant,
        checkpoint=str(weights/'model.pt'),output_dir=str(destination),weights_dir=str(weights),epochs=settings['epochs'],
        optimizer_updates=updates,parameter_count=parameter_count,elapsed_s=time.monotonic()-started,packing_s=packing_s,
        history=history,training_identity=identity,normal_inference_uses_gt_ids=False,read_TEST=False)
    write_json(destination/'receipt.json',receipt)
    write_json(destination/'progress.json',dict(status='completed',variant=variant,epoch=settings['epochs'],epochs=settings['epochs'],optimizer_updates=updates,mean_loss=history[-1]['loss']['total'],online_fixed_train_rmse_m=history[-1]['online_fixed_train_rmse_m'],elapsed_s=time.monotonic()-train_started,eta_s=0.))
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--index',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--resume-checkpoint')
    args=parser.parse_args(argv)
    print(json.dumps(train(args.config,args.variant,args.index,args.output_dir,device=args.device,batch_size=args.batch_size,resume_checkpoint=args.resume_checkpoint),indent=2,ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
