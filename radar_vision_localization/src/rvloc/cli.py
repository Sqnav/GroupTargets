"""Explicit experiment commands. The default action is never training."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from .config import ExperimentConfig, external_path
from .runtime import jsonable, write_json


def parser():
    root=argparse.ArgumentParser(prog='rvloc',description='32 MHz co-centered seven-method controlled experiment')
    commands=root.add_subparsers(dest='command',required=True)
    plan=commands.add_parser('plan',help='Read configuration only; never read data or checkpoints')
    plan.add_argument('--config',default='configs/seven_methods.json')
    validate=commands.add_parser('validate-data',help='Validate ready sensor assets, without opening labels')
    validate.add_argument('--manifest',required=True)
    validate.add_argument('--metadata-only',action='store_true')
    validate.add_argument('--modality',choices=('rgb','radar','rgb_radar'),default='rgb_radar')
    experiment=commands.add_parser('run-experiment',help='Train, calibrate and evaluate all seven methods with a realtime run.log')
    experiment.add_argument('--config',required=True)
    experiment.add_argument('--run-dir',required=True)
    experiment.add_argument('--devices',default='cpu',help='Comma-separated visible CUDA indices, cuda:N, or cpu')
    experiment.add_argument('--resume',action='store_true',help='Reuse verified completed stages; interrupted training restarts its stage')
    for name in ('train','infer'):
        command=commands.add_parser(name)
        command.add_argument('--config',required=True)
        command.add_argument('--method',required=True,choices=tuple(str(i) for i in range(1,8)))
        command.add_argument('--output-dir')
        if name=='infer':
            command.add_argument('--split',choices=('train','dev'),default='dev')
            command.add_argument('--device',default='cpu')
    calibration=commands.add_parser('calibrate',help='Fit and freeze temporal parameters using TRAIN only')
    calibration.add_argument('--config',required=True)
    calibration.add_argument('--predictions',nargs='+',required=True,help='Persisted TRAIN predictions for methods 1, 2 and 3')
    calibration.add_argument('--output',required=True)
    evaluation=commands.add_parser('evaluate',help='Offline GT scoring of already committed predictions')
    evaluation.add_argument('--config',required=True)
    evaluation.add_argument('--predictions',required=True)
    evaluation.add_argument('--output-dir')
    compare=commands.add_parser('compare',help='Offline paired episode bootstrap, second minus first')
    compare.add_argument('--first',required=True)
    compare.add_argument('--second',required=True)
    compare.add_argument('--output',required=True)
    compare.add_argument('--draws',type=int,default=2000)
    compare.add_argument('--seed',type=int,default=20261126)
    visual=commands.add_parser('prepare-visual',help='Export fixed detector boxes/ROI from RGB; never train a detector')
    visual.add_argument('--rgb',required=True)
    visual.add_argument('--output',required=True)
    visual.add_argument('--detector',required=True,help='Explicit module:function callback accepting an RGB image')
    visual.add_argument('--detector-sha256',required=True)
    radar=commands.add_parser('prepare-radar',help='Audited FFT RA/RE feature preparation from one native IQ CPI')
    radar.add_argument('--manifest',required=True)
    radar.add_argument('--iq',required=True)
    radar.add_argument('--output',required=True)
    return root


def execute(args):
    if args.command=='run-experiment':
        from .experiment import run_experiment
        return run_experiment(args.config,args.run_dir,devices=args.devices,resume=args.resume)
    if args.command=='plan':
        result=ExperimentConfig.load(args.config).plan()
        if result['manifest']:
            from .data import load_manifest
            result['data']=load_manifest(result['manifest'],allow_pending=True).metadata_report()
            if result['data']['status']!='ready':result['status']='code_ready_data_pending'
        return result
    if args.command=='validate-data':
        from .data import load_manifest
        manifest=load_manifest(args.manifest,allow_pending=args.metadata_only)
        result=manifest.metadata_report()
        if not args.metadata_only:
            counts={}
            for split in ('train','dev'):
                counts[split]=sum(1 for _ in manifest.observations(split,modality=args.modality))
            result.update(sensor_frames_validated=counts,data_files_opened=True,labels_opened=False,modality=args.modality)
        return result
    if args.command=='train':
        from .training import train
        return train(ExperimentConfig.load(args.config),args.method,output_dir=args.output_dir)
    if args.command=='infer':
        from .pipeline import infer
        output=external_path(args.output_dir,field='output_dir') if args.output_dir else None
        if output is not None and output.exists():raise ValueError('Inference output-dir already exists; choose a new run directory')
        return infer(ExperimentConfig.load(args.config),args.method,split=args.split,device=args.device,output_dir=output)
    if args.command=='calibrate':
        from .data import load_manifest
        from .calibration import fit_calibration
        from .calibration_inputs import collect_calibration_inputs
        config=ExperimentConfig.load(args.config)
        if config.manifest is None:raise ValueError('Dataset pending: explicit ready manifest required')
        output=external_path(args.output,field='calibration output')
        if output.exists():raise ValueError('Calibration output exists; frozen artifacts are immutable')
        manifest=load_manifest(config.manifest)
        inputs=collect_calibration_inputs(manifest,args.predictions,
            model_config={'match_max_distance_m':config.raw.get('evaluation',{}).get('match_max_distance_m')})
        input_receipt=inputs.pop('receipt',None)
        artifact=fit_calibration(**inputs,manifest_sha256=manifest.sha256,train_episode_ids=list(manifest.splits['train']))
        artifact['input_receipt']=input_receipt
        write_json(output,artifact)
        return dict(status=artifact['status'],calibration=str(output),manifest_sha256=manifest.sha256)
    if args.command=='evaluate':
        from .evaluation import evaluate
        return evaluate(ExperimentConfig.load(args.config),args.predictions,output_dir=args.output_dir)
    if args.command=='compare':
        from .evaluation import paired_compare
        result=paired_compare(args.first,args.second,draws=args.draws,seed=args.seed)
        write_json(args.output,result)
        return result
    if args.command=='prepare-visual':
        from .data import prepare_visual
        module,name=args.detector.split(':',1)
        callback=getattr(importlib.import_module(module),name)
        output=external_path(args.output,field='prepared visual output')
        result=prepare_visual(args.rgb,output,detector_callback=callback,detector_sha256=args.detector_sha256)
        return dict(status='visual_prepared',output=str(output),receipt=result)
    if args.command=='prepare-radar':
        from .data import load_manifest,prepare_radar
        manifest=load_manifest(args.manifest)
        output=external_path(args.output,field='prepared radar output')
        result=prepare_radar(args.iq,output,rf=manifest.rf)
        return dict(status='radar_prepared',output=str(output),receipt=result)
    raise ValueError('Unknown command')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=execute(args)
    except (ValueError,FileNotFoundError,KeyError) as error:
        print('rvloc: '+str(error),file=sys.stderr)
        return 2
    print(json.dumps(jsonable(result),ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
