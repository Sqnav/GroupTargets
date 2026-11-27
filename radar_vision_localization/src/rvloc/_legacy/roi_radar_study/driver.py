"""Modality-safe private adaptation of the unchanged epoch trainer."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from d_evidence_speed.train import PredictionReplay
from precision_study.visual_data import VisualSensorReader,public_split_manifest
from precision_study.evaluation import _geometric_gate
from rvwm.data import sha256

from . import checkpoint

ROOT=Path(__file__).resolve().parents[1]
PRIVATE='epoch_training._roi_radar_driver'
TRAINER=ROOT/'epoch_training/train.py'


def execution_identity():
    return dict(adapter='roi_radar_study.driver',adapter_sha256=sha256(__file__),
        trainer='epoch_training/train.py',trainer_sha256=sha256(TRAINER),
        private_module=PRIVATE,model_factory=checkpoint.MODEL_FACTORY,
        shared_modules_modified=False)


def public_context(config):
    """Only public trajectory and membership metadata; never IQ or spectrum IO."""
    path,manifest,checksum=public_split_manifest(config)
    splits={role:list(manifest['splits'][role]['episode_ids']) for role in ('train','val','test')}
    readers={role:VisualSensorReader(config,role,load_detector=False) for role in ('train','val')}
    roles={e:role for role in ('train','val') for e in splits[role]}
    class PublicFrames:
        def frames_in_phase(self,episode,phase):
            if episode not in roles:
                raise ValueError('No final-test observations in training metadata planning')
            rows=readers[roles[episode]].episode(episode)['capture']
            return [r['frame_idx'] for r in rows if r['phase']==phase]
    return PublicFrames(),splits,dict(policy='same frozen whole-episode scene-family membership',
        path=str(path),sha256=checksum,counts={r:len(v) for r,v in splits.items()},
        observation_payloads_read=False,IQ_or_spectra_read=False,test_observations_read=False)


def evaluate_batched(model,samples,manifest,config,evaluation,*,batch_size=32,store=None,allow_untrained=False):
    """No radar packer/prefilter is reachable for visual-only validation."""
    from .data import StudyFeatureStore
    visual=config['study_arm']=='visual'
    def rejection(features):
        if visual:
            return _geometric_gate(features,config)
        from rvwm.ray_scoring import observation_rejection_reason
        base={k:v for k,v in features.items() if k!='detector_roi_features'}
        return observation_rejection_reason(base,config)
    eligible=[s for s in samples if rejection(s['features']) is None]
    if store is None and eligible:
        store=StudyFeatureStore([{k:v for k,v in s.items() if k!='positions'} for s in eligible],
                                config,device=next(model.parameters()).device)
    previous=model.training;model.eval();scenes={}
    cuda=next(model.parameters()).is_cuda
    if cuda:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0,len(eligible),batch_size):
                indices=list(range(start,min(start+batch_size,len(eligible))))
                packed,_=store.batch(indices,dropout_probability=0.)
                outputs=model.predict_batch(packed,allow_untrained_validation=allow_untrained)
                if len(outputs)!=len(indices):
                    raise ValueError('Batch cardinality differs from current sensor inventory')
                scenes.update((id(eligible[i]['features']),scene) for i,scene in zip(indices,outputs))
        if cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        replay=PredictionReplay(scenes,trained=model.trained,config=config)
        report=evaluation.evaluate(replay,samples,manifest,config,allow_untrained=allow_untrained)
        summary=report['summary']
        summary['cpu_scene_replay_seconds_per_frame']=summary.pop('inference_seconds_per_frame',None)
        summary['inference_seconds_per_frame']=dict(count=len(scenes),mean=elapsed/max(1,len(scenes)),
            estimand='amortized resident actual batched prediction, not score replay latency')
        summary['latency_scope']='resident sensor-only batched prediction; excludes loading, ROI extraction, FFT and label scoring'
        summary['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if cuda else 0
        summary['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if cuda else 0
        report['execution_timing']=dict(actual_batched_forward=True,predicted_frames=len(scenes),inference_seconds=elapsed,
            radar_features_read=not visual,labels_used_for_prediction=False)
        return report,store
    finally:
        model.train(previous)


def load_driver():
    if PRIVATE in sys.modules:
        module=sys.modules[PRIVATE]
        if module._study_trainer_sha256!=sha256(TRAINER):raise ValueError('Private trainer source changed')
        return module
    from .data import StudyFeatureStore,load_samples
    from .model import StudyDepthLocalizer
    spec=importlib.util.spec_from_file_location(PRIVATE,TRAINER)
    module=importlib.util.module_from_spec(spec);sys.modules[PRIVATE]=module
    try:
        spec.loader.exec_module(module)
        original_validate=module.validate_options;original_plan=module.make_plan
        def validate_options(args,config):
            original_validate(args,dict(config,precision_radar_mode='ra_re'))
            checkpoint.validate_config(config)
        def make_plan(config,args,world_size=1):
            plan=original_plan(config,args,world_size)
            plan.update(study_arm=config['study_arm'],study_protocol_sha256=config['study_protocol_sha256'],
                        study_execution=execution_identity())
            return plan
        module.validate_options=validate_options;module.make_plan=make_plan
        module._dataset_context=public_context
        module.FastDepthLocalizer=StudyDepthLocalizer;module.GPUFeatureStore=StudyFeatureStore
        module.load_samples=load_samples;module.evaluate_batched=evaluate_batched;module.checkpoint=checkpoint
        module._study_trainer_sha256=sha256(TRAINER)
    except Exception:
        sys.modules.pop(PRIVATE,None);raise
    return module


def parser():
    return load_driver().parser()


def train(args,*,prepared=None):
    if prepared is not None:
        prepared=dict(prepared,plan=deepcopy(prepared['plan']))
        prepared['plan']['study_execution']=execution_identity()
    return load_driver().train(args,prepared=prepared)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.config is None or args.output is None:raise ValueError('Explicit study config and independent output are required')
    if args.dry_run:
        config=json.loads(Path(args.config).read_text());load_driver().validate_options(args,config)
        value=dict(status='dry_run',epochs=args.epochs,per_rank_batch_size=args.batch_size,
                   execution=execution_identity(),observations_or_labels_read=False,model_constructed=False)
        print(json.dumps(value));return value
    return train(args)


if __name__=='__main__':
    main()
