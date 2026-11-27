"""Reproduce frozen v3 and v2 on exactly the same declared validation frames."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import torch

from depth_localization.checkpoint import load_checkpoint, source_hashes
from depth_localization.data import load_samples
from depth_localization.features import feature_digest
from .evaluation import evaluate_samples, serializable

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_v2_loader(snapshot):
    """Import the immutable v2 package under its own namespace, strictly."""
    path = Path(snapshot)/'depth_localization'
    name = '_precision_frozen_depth_v2'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,path/'__init__.py',
                                                       submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name+'.checkpoint')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'depth_config.json')
    parser.add_argument('--v3-checkpoint',type=Path,default=ROOT/'results/training/depth_v3/depth.rmse.best.pt')
    parser.add_argument('--v2-checkpoint',type=Path,default=ROOT/'results/training/depth_v2/depth.best.pt')
    parser.add_argument('--v2-snapshot',type=Path,default=ROOT/'results/training/depth_v2/source_snapshot_before_restart')
    parser.add_argument('--output',type=Path,default=ROOT/'results/precision_study_20261122/baseline')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--max-episodes',type=int,default=0)
    parser.add_argument('--frames-per-episode',type=int,default=0)
    parser.add_argument('--models',default='v3,v2')
    args=parser.parse_args(argv)
    if args.threads<1 or min(args.max_episodes,args.frames_per_episode)<0:
        parser.error('Invalid thread/data limit')
    names=args.models.split(',')
    if not names or any(x not in ('v3','v2') for x in names) or len(set(names))!=len(names):
        parser.error('--models must contain v3 and/or v2 without duplicates')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Existing baseline reproduction is protected: '+str(out))
    torch.set_num_threads(args.threads)
    cfg=json.loads(args.config.read_text())
    reference=load_checkpoint(args.v3_checkpoint,cfg,args.device)
    metadata=reference.checkpoint_metadata
    split=json.loads(Path(cfg['split_manifest']).read_text())
    ids=list(split['splits']['val']['episode_ids'])
    if ids != metadata['validation_episodes'] or set(ids)&set(metadata['train_episodes']):
        raise ValueError('Checkpoint validation episodes differ from frozen episode split')
    if args.max_episodes:
        ids=ids[:args.max_episodes]
    started=time.perf_counter()
    progress=dict(stage='loading_authenticated_validation',episodes=len(ids),models=names,
                  labels_used_for_prediction=False,test_set_used=False)
    (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    samples,manifest,frontend=load_samples(cfg,ids,'val',count=args.frames_per_episode,include_labels=True)
    identity=[dict(episode_id=s['episode_id'],frame_idx=s['frame_idx'],
                   feature_digest=feature_digest(s['features'])) for s in samples]
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    info=dict(started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        command=[sys.executable,*sys.argv],configuration=cfg,configuration_path=str(args.config.resolve()),
        configuration_sha256=sha256(args.config),sensor_input_inventory_sha256=identity_sha,
        input_frames=len(samples),declared_frames=len(manifest),validation_episodes=ids,
        diagnostic_subset=bool(args.max_episodes or args.frames_per_episode),split='val',
        labels_used_for_prediction=False,test_set_used=False,frontend=frontend,
        current_source_sha256=source_hashes(),torch_version=torch.__version__,python=platform.python_version(),
        device=args.device,threads=args.threads,
        evaluation_source_sha256=sha256(Path(__file__).with_name('evaluation.py')))
    (out/'input_inventory.json').write_text(json.dumps(dict(frames=identity,manifest=manifest),ensure_ascii=False,indent=2)+'\n')
    (out/'configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    results={}
    for name in names:
        model=reference if name=='v3' else frozen_v2_loader(args.v2_snapshot).load_checkpoint(args.v2_checkpoint,device=args.device)
        checkpoint=args.v3_checkpoint if name=='v3' else args.v2_checkpoint
        metadata=model.checkpoint_metadata
        if metadata['validation_episodes'] != split['splits']['val']['episode_ids']:
            raise ValueError('Compared model was trained with another validation inventory')
        run_info=dict(info,model_name=name,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint),
            checkpoint_step=metadata['steps'],model_configuration=model.config,
            checkpoint_source_sha256=metadata['source_sha256'],checkpoint_training_scope=metadata['training_scope'],
            checkpoint_train_inventory_sha256=metadata['inventory_sha256'],
            checkpoint_val_inventory_sha256=metadata['validation_inventory_sha256'])
        def pulse(row):
            value=dict(stage='evaluating',model=name,**row,labels_used_for_prediction=False,test_set_used=False)
            (out/'progress.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(value),flush=True)
        result=evaluate_samples(model,samples,manifest,cfg,out/name,
                                capture_stages=True,progress_callback=pulse,run_info=run_info)
        results[name]=result['summary']
        print(json.dumps(dict(event='model_complete',name=name,rmse_m=result['summary']['member_position_error_m']['rmse'])),flush=True)
    report=dict(status='complete',summaries=results,sensor_input_inventory_sha256=identity_sha,
                elapsed_s=time.perf_counter()-started,run_info=info)
    (out/'comparison.json').write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'progress.json').write_text(json.dumps(dict(stage='complete',frames=len(manifest),models=names,
        elapsed_s=time.perf_counter()-started),indent=2)+'\n')
    print(json.dumps(dict(status='complete',output=str(out),frames=len(manifest))),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
