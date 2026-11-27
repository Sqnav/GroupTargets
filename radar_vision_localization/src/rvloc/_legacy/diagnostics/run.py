"""Frozen-model error decomposition and paired radar intervention experiments."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from rvwm.data import sha256
from rvwm.group_model import load_group_model
from rvwm.prior_validation import frame_seed
from rvwm.ray_scoring import select_scene_modes,observation_rejection_reason
from rvwm.source_compatibility import verify_inference_sources
from rvwm.validation import write_json
from .data import ROOT,load_cached_samples,original_root,verify_original_sources
from .geometry import freeze_correspondence,decompose_frame,aggregate_frames,oracle_bearing_features
from .interventions import INTERVENTIONS,apply_intervention,choose_joint_donor_indices,paired_response,normalization_audit


ARMS=(*INTERVENTIONS,'oracle_true_bearing')


def source_receipt():
    directory=Path(__file__).parent
    return {name:sha256(directory/name) for name in ('run.py','data.py','geometry.py','interventions.py')}


def primary(model,features,config,episode,index):
    scenes=model.propose(features,int(config.get('group_hypotheses',32)),
        np.random.default_rng(frame_seed(int(config['seed']),episode,int(index))))
    kept=select_scene_modes(features,scenes,config)['modes']
    return kept[0] if kept else None


def _response_summary(rows):
    responses=[row['response'] for row in rows if row.get('response',{}).get('paired_members',0)]
    distance=np.asarray([v for r in responses for v in r['member_displacement_m']],float)
    radial=np.asarray([v for r in responses for v in r['member_base_radial_shift_m']],float)
    def stats(v):
        return dict(count=len(v),mean=float(v.mean()),rms=float(np.sqrt(np.mean(v*v))),p50=float(np.quantile(v,.5)),p95=float(np.quantile(v,.95)),maximum_signed=float(v.max()),maximum_absolute=float(np.abs(v).max())) if len(v) else dict(count=0)
    result=dict(displacement_m=stats(distance),base_radial_shift_m=stats(radial),matching='same detection slots; labels not used for response')
    gains=[r['mean_base_radial_response_gain'] for r in responses if r.get('mean_base_radial_response_gain') is not None]
    if gains:result['mean_frame_radial_response_gain']=float(np.mean(gains))
    return result


def summarize(rows):
    return dict(raw_model=aggregate_frames(rows),
        normal_pipeline_gate_counts=dict(Counter(r['normal_pipeline_rejection'] or 'accepted' for r in rows)),
        raw_primary_count=sum(r['raw_primary_available'] for r in rows),
        normally_publishable_primary_count=sum(r['raw_primary_available'] and r['normal_pipeline_rejection'] is None for r in rows),
        paired_response=_response_summary(rows))


def run_shard(args):
    config=json.loads(Path(args.config).read_text());run=verify_original_sources(config)
    output=Path(args.output).resolve();directory=output/f'part_{args.rank}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise FileExistsError('Existing diagnostic shard is protected')
    torch.set_num_threads(args.threads)
    checkpoint=Path(args.checkpoint or config['world_checkpoint']).resolve()
    model=load_group_model(checkpoint,config,args.device)
    source=verify_inference_sources(checkpoint,model.checkpoint_metadata)
    ids=run['validation_episodes'] if args.split=='val' else run['train_episodes']
    started=time.perf_counter();samples,manifest,frontend=load_cached_samples(config,ids,args.split)
    # Donors are drawn from the full frozen split, even for a small smoke run.
    # A prefix can contain just one episode and cannot supply its own donor.
    donors=choose_joint_donor_indices(samples,seed=args.donor_seed) if 'joint_shuffle' in args.arms else None
    selected_count=min(args.limit,len(samples)) if args.limit else len(samples)
    local=[i for i in range(selected_count) if i%args.world_size==args.rank]
    if not local:raise ValueError('Empty diagnostic shard')
    receipt=dict(status='running',schema='frozen_fusion_radar_diagnostics_v1',rank=args.rank,world_size=args.world_size,
        configuration=config,checkpoint=dict(path=str(checkpoint),sha256=sha256(checkpoint),steps=model.checkpoint_metadata['steps']),
        original_training_manifest_sha256=sha256(original_root(config)/'run_manifest.json'),source_compatibility=source,
        diagnostic_source_sha256=source_receipt(),split=args.split,arms=args.arms,
        donor_seed=args.donor_seed,selection=[dict(episode_id=samples[i]['episode_id'],frame_idx=samples[i]['frame_idx']) for i in local],
        full_selection=[dict(episode_id=s['episode_id'],frame_idx=s['frame_idx']) for s in samples[:selected_count]],
        expected_total_frames=selected_count,donor_pool_frames=len(samples),selected_frames=len(local),completed_frames=0,
        sensor_payloads_reread=False,input_source='authenticated original training feature cache plus frozen DINO export',
        gt_use='separate scoring, ray decomposition and explicitly named oracle bearing only',
        raw_vs_published='zero or invalid input retains gate rejection; raw model diagnostics are not normal deployment outputs',
        normalization_audit=normalization_audit(),frontend=frontend)
    write_json(directory/'manifest.json',receipt)
    collected={arm:[] for arm in args.arms};handles={arm:(directory/(arm+'.jsonl')).open('x') for arm in args.arms}
    try:
        for number,i in enumerate(local):
            sample=samples[i];episode=sample['episode_id'];index=sample['frame_idx'];features=sample['features'];truth=sample['positions']
            correspondence=freeze_correspondence(features,truth,config)
            reference=primary(model,features,config,episode,index)
            for arm in args.arms:
                if arm=='baseline':transformed=features;scene=reference
                elif arm=='oracle_true_bearing':
                    transformed=oracle_bearing_features(features,truth,correspondence=correspondence,config=config)
                    scene=primary(model,transformed,config,episode,index)
                else:
                    donor=samples[donors[i]] if arm=='joint_shuffle' else None
                    transformed=apply_intervention(features,arm,donor=None if donor is None else donor['features'],
                        donor_identity=None if donor is None else dict(episode_id=donor['episode_id'],frame_idx=donor['frame_idx']))
                    scene=primary(model,transformed,config,episode,index)
                row=decompose_frame(transformed,truth,scene,correspondence=correspondence,config=config)
                rejection=observation_rejection_reason(transformed,config)
                probe=transformed['metadata'].get('diagnostic_intervention',{})
                shift=probe.get('commanded_range_shift_m',probe.get('range_shift_m'))
                if arm.startswith('range_shift_') and shift is None:
                    bins=-2 if arm.endswith('minus2') else -1 if arm.endswith('minus1') else 1 if arm.endswith('plus1') else 2
                    shift=float(np.diff(features['range_m'])[0])*bins
                valid=None if reference is None or scene is None else np.asarray(reference['valid_mask'])&np.asarray(scene['valid_mask'])
                response=paired_response(None if reference is None else reference['member_positions'],None if scene is None else scene['member_positions'],
                    valid,commanded_range_shift_m=shift,ray_directions=features['ray_directions'])
                row.update(episode_id=episode,frame_idx=index,intervention=arm,normal_pipeline_rejection=rejection,
                    raw_primary_available=scene is not None,raw_only=rejection is not None,
                    oracle_label_assisted=arm=='oracle_true_bearing',response=response,
                    intervention_receipt=probe,primary_hypothesis=None if scene is None else scene['hypothesis_index'])
                collected[arm].append(row)
                handles[arm].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if (number+1)%20==0 or number+1==len(local):
                receipt.update(completed_frames=number+1,elapsed_s=time.perf_counter()-started)
                write_json(directory/'manifest.json',receipt)
                print(json.dumps(dict(rank=args.rank,completed_frames=number+1,total_frames=len(local),elapsed_s=receipt['elapsed_s'])),flush=True)
        if source_receipt()!=receipt['diagnostic_source_sha256']:raise ValueError('Diagnostic inference source changed during execution')
        summary={arm:summarize(rows) for arm,rows in collected.items()}
        write_json(directory/'summary.json',summary)
        receipt.update(status='complete',completed_frames=len(local),elapsed_s=time.perf_counter()-started)
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        for stream in handles.values():stream.close()
        receipt['prediction_sha256']={arm:sha256(directory/(arm+'.jsonl')) for arm in args.arms}
        write_json(directory/'manifest.json',receipt)


def merge(output):
    output=Path(output).resolve();manifests=[];by_arm={};seen=set();ranks=set()
    for directory in sorted(output.glob('part_*')):
        m=json.loads((directory/'manifest.json').read_text())
        if m['status']!='complete':raise ValueError('Diagnostic shard is incomplete: '+str(directory))
        rank=m['rank'];world_size=m['world_size']
        if not isinstance(rank,int) or not 0<=rank<world_size or rank in ranks or directory.name!=f'part_{rank}':
            raise ValueError('Invalid or duplicate diagnostic rank')
        ranks.add(rank)
        if m['diagnostic_source_sha256']!=source_receipt():raise ValueError('Diagnostic inference source changed')
        if sha256(m['checkpoint']['path'])!=m['checkpoint']['sha256']:raise ValueError('Diagnostic checkpoint bytes changed')
        if sha256(original_root(m['configuration'])/'run_manifest.json')!=m['original_training_manifest_sha256']:
            raise ValueError('Original training manifest changed')
        plan=m['full_selection']
        if len(plan)!=m['expected_total_frames'] or m['selection']!=plan[rank::world_size]:
            raise ValueError('Diagnostic selection differs from declared global plan')
        if manifests:
            for key in ('world_size','configuration','checkpoint','split','arms','donor_seed','expected_total_frames','donor_pool_frames','full_selection','diagnostic_source_sha256','original_training_manifest_sha256'):
                if m[key]!=manifests[0][key]:raise ValueError('Inconsistent diagnostic shards: '+key)
        keys={(r['episode_id'],r['frame_idx']) for r in m['selection']}
        if len(keys)!=len(m['selection']) or seen&keys:raise ValueError('Duplicate diagnostic frame selection')
        seen|=keys;manifests.append(m)
        for arm in m['arms']:
            p=directory/(arm+'.jsonl')
            if sha256(p)!=m['prediction_sha256'][arm]:raise ValueError('Diagnostic rows changed')
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            if {(r['episode_id'],r['frame_idx']) for r in rows}!=keys or len(rows)!=len(keys):raise ValueError('Diagnostic shard row coverage differs')
            if any(r.get('intervention')!=arm for r in rows):raise ValueError('Diagnostic row intervention differs from arm')
            by_arm.setdefault(arm,[]).extend(rows)
    if not manifests or ranks!=set(range(manifests[0]['world_size'])) or len(seen)!=manifests[0]['expected_total_frames']:
        raise ValueError('Full declared diagnostic population is not covered')
    summary={arm:summarize(rows) for arm,rows in by_arm.items()}
    result=dict(status='complete',schema='frozen_fusion_radar_diagnostics_v1',frames=len(seen),split=manifests[0]['split'],
        checkpoint=manifests[0]['checkpoint'],arms=summary,normalization_audit=normalization_audit(),
        source_sha256=manifests[0]['diagnostic_source_sha256'],
        interpretation='Inference interventions are OOD sensitivity probes, not a retrained visual-only baseline. Oracle bearings use GT labels only for diagnosis.')
    write_json(output/'summary.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--donor-seed',type=int,default=20261121);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));p.add_argument('--merge',action='store_true')
    args=p.parse_args(argv)
    if args.merge:print(json.dumps(merge(args.output),ensure_ascii=False));return 0
    if not 0<=args.rank<args.world_size or args.threads<1 or args.limit<0:p.error('Invalid shard or diagnostic scope')
    if len(set(args.arms))!=len(args.arms):p.error('Duplicate intervention arms')
    run_shard(args);return 0


if __name__=='__main__':raise SystemExit(main())
