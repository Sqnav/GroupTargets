"""Fresh training using the same fixed model/config identity as deployment.

The deployed checkpoint authenticates the selected architecture and sources;
its tensors are never used for initialization. --dry-run only prints the
reviewable command and does not create output or run a trainer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys

from . import deployed_infer

ROOT=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--output',type=Path);p.add_argument('--ranks',type=int,default=6)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--validate-every',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--detection-dropout',type=float,default=.2)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    for name in ('max-episodes','frames-per-episode','validation-episodes','validation-frames'):
        p.add_argument('--'+name,type=int,default=0)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--dry-run',action='store_true')
    return p


def source_identity(description):
    """Check frozen training sources against the authenticated clean receipt."""
    if description['kind']=='study':
        from .versions import frozen_study_package
        name,package=frozen_study_package(description['source_root'])
        if not (package/'training.py').is_file():raise FileNotFoundError('Frozen study trainer is missing')
        actual=importlib.import_module(name+'.checkpoint').source_hashes()
        cwd=package.parent;module='precision_study.training'
    else:
        from depth_localization.checkpoint import source_hashes
        actual=source_hashes();cwd=ROOT;module='depth_localization.training'
    receipt=json.loads((description['checkpoint'].parent/'deployment_receipt.json').read_text())
    if receipt.get('source_sha256')!=actual:raise ValueError('Fixed deployment and actual training sources differ')
    return cwd,module


def build_command(args,*,descriptor_path=None):
    if min(args.ranks,args.steps,args.validate_every,args.batch_size,args.threads)<1 or args.seed<0:
        raise ValueError('Positive ranks, steps, intervals, batch size, threads and a nonnegative seed are required')
    if min(args.max_episodes,args.frames_per_episode,args.validation_episodes,args.validation_frames)<0:
        raise ValueError('Training subset limits cannot be negative')
    if (not math.isfinite(args.learning_rate) or args.learning_rate<=0 or
            not math.isfinite(args.detection_dropout) or not 0<=args.detection_dropout<=1):
        raise ValueError('Require finite positive learning rate and dropout in [0,1]')
    fixed=Path(descriptor_path) if descriptor_path is not None else deployed_infer.DEFAULT_DESCRIPTOR
    description,identity=deployed_infer.load_descriptor(fixed)
    cwd,module=source_identity(description)
    output=(args.output or ROOT/'results/training/deployed'/description['model_name']/datetime.now(
        timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')).expanduser().resolve()
    if output.exists():raise FileExistsError('Any existing training output is protected: '+str(output))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node='+str(args.ranks),'-m',module,'--config',str(description['config']),
        '--output',str(output),'--device',args.device]
    for name in ('steps','validate_every','batch_size','seed','learning_rate','detection_dropout','threads',
                 'max_episodes','frames_per_episode','validation_episodes','validation_frames'):
        command+=['--'+name.replace('_','-'),str(getattr(args,name))]
    if args.plan_only:command+=['--plan-only']
    overrides=dict(PYTHONPATH=str(cwd)+(os.pathsep+str(ROOT) if cwd!=ROOT else ''),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
    if args.device=='cpu':overrides['CUDA_VISIBLE_DEVICES']=''
    return dict(command=command,cwd=str(cwd),environment_overrides=overrides,output=str(output),
        descriptor=identity,descriptor_path=str(fixed.resolve()),training_from_scratch=True,
        initial_fusion_checkpoint=None,deployment_weights_used_for_initialization=False,
        world_size=args.ranks,global_batch_size=args.ranks*args.batch_size,
        plan_only=args.plan_only,dry_run=args.dry_run)


def main(argv=None,*,descriptor_path=None):
    args=parser().parse_args(argv);plan=build_command(args,descriptor_path=descriptor_path)
    if args.dry_run:
        print(json.dumps(plan,indent=2,allow_nan=False));return 0
    _,identity=deployed_infer.load_descriptor(plan['descriptor_path'])
    if identity!=plan['descriptor']:raise ValueError('Fixed deployment identity changed before training')
    environment=os.environ.copy()
    for name in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):
        environment.pop(name,None)
    environment.update(plan['environment_overrides'])
    if not args.plan_only:
        output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
        with (output/'training_route_receipt.json').open('x') as stream:
            json.dump(plan,stream,indent=2,allow_nan=False);stream.write('\n')
    os.chdir(plan['cwd'])
    os.execve(sys.executable,plan['command'],environment)


if __name__=='__main__':raise SystemExit(main())
