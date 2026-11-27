"""Real TRAIN-only microbenchmark; smoke updates are not formal training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
for folder in (PROJECT,PROJECT/'src',PROJECT/'tools'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
from gnn_research import train as graph_train
from rvloc.config import ExperimentConfig
from rvloc.data import load_manifest
from rvloc.runtime import file_sha256,write_json
from rvloc.training import _settings

FRAME_FIELDS=('shared_candidate_count','valid_candidate_count','shared_member_count','valid_member_count',
    'unique_evidence_nodes','edge_count','shuffled_changed_edge_count','geometry_valid_candidate_count','connected_member_count')


def collect_diagnostic(value):
    result={key:value[key].detach().clone() for key in FRAME_FIELDS if key in value}
    for name in ('evidence_candidate_degree','evidence_distinct_member_degree','candidate_degree'):
        if name in value:result[name]=value[name].detach().flatten().clone()
    if 'candidate_offsets_m' in value:
        mask=value['geometry_valid'].bool()
        result['candidate_offsets_m']=value['candidate_offsets_m'].detach()[mask].flatten().clone()
    return result


def summarize_diagnostics(values):
    result={name:int(torch.cat([row[name].flatten() for row in values if name in row]).sum()) for name in FRAME_FIELDS}
    result['shared_candidate_coverage']=result['shared_candidate_count']/max(1,result['valid_candidate_count'])
    result['shared_member_coverage']=result['shared_member_count']/max(1,result['valid_member_count'])
    result['shuffled_changed_edge_fraction']=result['shuffled_changed_edge_count']/max(1,result['edge_count'])
    for name in ('evidence_candidate_degree','evidence_distinct_member_degree','candidate_degree','candidate_offsets_m'):
        flat=torch.cat([row[name].flatten() for row in values if name in row]).float()
        result[name]=dict(count=flat.numel(),minimum=float(flat.min()) if flat.numel() else None,
            mean=float(flat.mean()) if flat.numel() else None,maximum=float(flat.max()) if flat.numel() else None)
    result['aggregation_scope']='20 timed optimizer updates, cycling 32 frozen TRAIN frames; repeated frames counted repeatedly'
    return result


def benchmark(config_path,index_path,output_dir,*,device='cuda:0',warmup_updates=3,updates=20,limit_frames=32):
    started=time.monotonic();config=ExperimentConfig.load(config_path);settings=_settings(config)
    require=graph_train.require
    require(str(device).startswith('cuda'),'This receipt requires actual CUDA hardware')
    require(limit_frames==32 and warmup_updates==3 and updates==20,'Frozen smoke budget differs')
    torch.cuda.set_device(device);torch.set_num_threads(1)
    destination=Path(output_dir).resolve();destination.mkdir(parents=True,exist_ok=True)
    sources=graph_train.training_sources()
    manifest=load_manifest(config.manifest);rf=manifest.rf.as_dict()
    load_started=time.monotonic()
    values,data_identity=graph_train.base.load_training_data(config,index_path,3,limit_frames=limit_frames)
    require(not data_identity['full_train'],'Smoke must remain explicitly partial TRAIN')
    selected=[dict(episode_id=observation.episode_id,frame_idx=observation.frame_idx,
        physical_time_s=observation.physical_time_s,source_hashes=observation.source_hashes) for observation,_ in values]
    batch_started=time.monotonic();batches=graph_train.base.prepare_batches(values,4,device,False)
    torch.cuda.synchronize(device)
    preload_s=batch_started-load_started;packing_s=time.monotonic()-batch_started
    weights=config.weights_root/destination.relative_to(config.output_root)
    weights.mkdir(parents=True,exist_ok=True)
    options=dict(config.raw.get('model',{}));options['gnn']=dict(config.raw.get('gnn',{}))
    identity=dict(schema='rvloc.mce_gnn_real_train_smoke.v1',smoke_only=True,formal_training=False,
        training_split='train',DEV_read=False,TEST_read=False,config_path=str(config.path),config_sha256=config.sha256,
        manifest_path=str(config.manifest),data_identity=data_identity,rf=rf,
        sources=sources,smoke_source_sha256=file_sha256(Path(__file__)),selected_frames=selected,
        frozen_budget=dict(limit_frames=limit_frames,batch_size=4,warmup_updates=warmup_updates,timed_updates=updates),
        reproduction_command=[sys.executable,*sys.argv],environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')},
        optimizer_settings=dict(learning_rate=settings['learning_rate'],weight_decay=settings['weight_decay'],gradient_clip=settings['grad_clip']),
        gpu=dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(device),name=torch.cuda.get_device_name(device),
            total_memory_bytes=torch.cuda.get_device_properties(device).total_memory),
        data_preload_s=preload_s,gpu_packing_s=packing_s,batch_frame_counts=[batch['frames'] for batch in batches],
        normal_inference_uses_GT=False,labels_only_in_loss=True,default_deployment_modified=False,
        cost_scope='prepacked cached RGB ROI and native RA/RE; excludes DINO extraction, FFT, data IO and evaluation')
    write_json(destination/'identity.json',identity)
    results=[]
    for variant in graph_train.VARIANTS:
        torch.manual_seed(settings['seed']);np.random.seed(settings['seed']);random.seed(settings['seed'])
        torch.cuda.empty_cache()
        model=graph_train.build(options,variant,rf).to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        print(f'SMOKE {variant} warmup={warmup_updates} timed={updates} frames={limit_frames} device={device}',flush=True)
        for i in range(warmup_updates):graph_train.step(model,optimizer,batches[i%len(batches)],settings)
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        timer=time.monotonic();diagnostics=[];losses=[];norms=[];frame_count=matched_count=0
        for i in range(updates):
            batch=batches[(warmup_updates+i)%len(batches)]
            loss,terms,sse,count,norm,diag=graph_train.step(model,optimizer,batch,settings)
            losses.append(loss.detach());norms.append(norm.detach());diagnostics.append(collect_diagnostic(diag))
            frame_count+=batch['frames'];matched_count+=batch['matched_members']
        torch.cuda.synchronize(device);duration=time.monotonic()-timer
        peak_allocated=torch.cuda.max_memory_allocated(device);peak_reserved=torch.cuda.max_memory_reserved(device)
        require(bool(torch.isfinite(torch.stack(losses)).all()),'Nonfinite real TRAIN smoke loss')
        require(bool(torch.isfinite(torch.stack(norms)).all()),'Nonfinite real TRAIN smoke gradient norm')
        grad_count=sum(p.grad is not None for p in model.parameters())
        require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'Nonfinite final gradient')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite parameter')
        summary=summarize_diagnostics(diagnostics)
        if variant=='isolated':require(summary['shared_member_count']==0,'Isolated graph leaked cross-member sharing')
        if variant=='shared':require(summary['shared_member_count']>0,'Real TRAIN frames did not exercise evidence sharing')
        if variant=='shuffled':require(summary['shuffled_changed_edge_count']>0,'Shuffled graph did not alter associations')
        write_json(destination/f'{variant}_timing_before_checkpoint.json',dict(variant=variant,timed_elapsed_s=duration,timed_frame_presentations=frame_count,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,graph_diagnostics=summary,smoke_only=True))
        checkpoint=weights/f'{variant}_SMOKE_ONLY.pt'
        graph_train.save_graph_checkpoint(model,checkpoint,variant=variant,
            metadata=dict(smoke_only=True,formal_training=False,optimizer_updates=warmup_updates+updates,
                rf=rf,config_sha256=config.sha256,data_identity=data_identity,training_sources=sources,rng_state=graph_train.base._rng_state(device)))
        model.eval()
        with torch.inference_mode():reference=graph_train.forward(model,batches[0]['inputs'])['member_positions'].detach()
        # Reload with the experiment's actual RF dictionary, never the defaults.
        restored,payload,checkpoint_sha=graph_train.load_graph_checkpoint(checkpoint,device=device,expected_variant=variant)
        require(restored.config['graph_rf']==rf,'Checkpoint RF roundtrip differs')
        graph_train.base._restore_rng(payload['metadata']['rng_state'],device)
        require(torch.equal(torch.get_rng_state(),payload['metadata']['rng_state']['torch_cpu']),'CPU RNG checkpoint roundtrip differs')
        require(torch.equal(torch.cuda.get_rng_state(device),payload['metadata']['rng_state']['torch_cuda_current']),'CUDA RNG checkpoint roundtrip differs')
        with torch.inference_mode():
            roundtrip=graph_train.forward(restored,batches[0]['inputs'])['member_positions']
            repeat=graph_train.forward(model,batches[0]['inputs'])['member_positions']
        delta=float((reference-roundtrip).abs().max())
        repeat_delta=float((reference-repeat).abs().max())
        exact_state=all(torch.equal(value.detach().cpu(),restored.state_dict()[key].detach().cpu()) for key,value in model.state_dict().items())
        require(exact_state,'Checkpoint parameter or buffer state differs')
        require(torch.allclose(reference,roundtrip,atol=1e-4,rtol=1e-5),'Checkpoint forward roundtrip exceeds measured CUDA sparse-sum tolerance')
        row=dict(variant=variant,smoke_only=True,formal_training=False,warmup_updates=warmup_updates,timed_updates=updates,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            timed_elapsed_s=duration,timed_frame_presentations=frame_count,timed_matched_member_presentations=matched_count,
            updates_per_s=updates/duration,frames_per_s=frame_count/duration,mean_update_ms=duration*1000/updates,
            amortized_frame_ms=duration*1000/frame_count,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
            final_loss=float(losses[-1]),all_timed_losses_finite=True,all_timed_gradient_norms_finite=True,
            final_gradient_tensor_count=grad_count,final_gradients_finite=True,parameters_finite=True,
            graph_diagnostics=summary,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            checkpoint_actual_rf_roundtrip=True,checkpoint_rng_roundtrip=True,checkpoint_state_exact=exact_state,
            checkpoint_forward_max_abs_error_m=delta,same_model_repeat_max_abs_error_m=repeat_delta,
            checkpoint_forward_tolerance=dict(absolute_m=1e-4,relative=1e-5,reason='CUDA sparse index_add atomic accumulation has measured float32 repeat variation'),
            training_sources_unchanged=graph_train.training_sources()==sources)
        require(row['training_sources_unchanged'],'Frozen core changed during smoke')
        results.append(row);write_json(destination/f'{variant}_result.json',row)
        print(f'SMOKE {variant} DONE update_ms={row["mean_update_ms"]:.3f} fps={row["frames_per_s"]:.2f} peak_GiB={peak_allocated/2**30:.3f} shared_member={summary["shared_member_coverage"]:.4f} changed_edge={summary["shuffled_changed_edge_fraction"]:.4f} RF_roundtrip={delta:.1e}',flush=True)
        del restored,payload,roundtrip,repeat,reference,model,optimizer,diagnostics,losses,norms,loss,terms,sse,count,norm,diag
        gc.collect();torch.cuda.empty_cache()
    receipt=dict(identity,status='completed',started_at_utc=datetime.fromtimestamp(time.time()-(time.monotonic()-started),timezone.utc).isoformat(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,variants=results)
    write_json(destination/'receipt.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--index',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    try:benchmark(args.config,args.index,args.output_dir,device=args.device)
    except Exception as exc:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
        write_json(Path(args.output_dir)/'failure_receipt.json',dict(status='failed',smoke_only=True,error=str(exc),traceback=traceback.format_exc(),DEV_read=False,TEST_read=False))
        raise
    return 0

if __name__=='__main__':raise SystemExit(main())
