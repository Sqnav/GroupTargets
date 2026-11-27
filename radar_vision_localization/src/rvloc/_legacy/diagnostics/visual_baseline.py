"""Retrain a matched ray/navigation/geometry-only control without radar measurements.

This module reuses the original optimizer, initialization, sampler, dropout and
deployed-primary evaluation *in this process only*.  Both measured power maps are
replaced by exact zeros.  The visual-only observation contract has no radar-signal
quality gate; it retains valid-ray and public sensor-geometry checks.  This is a
separately trained control, not the inference-time zero-power intervention.

The original RVWM source files and caches are never edited.  Checkpoints receive
a distinct kind and explicit transform/source receipts, so the normal fusion
loader refuses to treat them as measured-radar training results.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack,contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rvwm import group_training as gt
from rvwm.group_model import GroupSceneModel,GROUP_SCHEMA
from rvwm.group_observation import feature_digest,validate_features


PROJECT=Path(__file__).resolve().parents[1]
VARIANT='retrained_visual_geometry_only'
CHECKPOINT_KIND='diagnostic_visual_geometry_baseline'
TRANSFORM='replace_both_measured_RA_RE_power_maps_with_exact_zeros_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_source_hashes():
    """Only train-influential diagnostic code belongs to this control's identity."""
    result={'visual_baseline.py':_sha(__file__)}
    reader=Path(__file__).with_name('data.py')
    if reader.is_file():result['data.py']=_sha(reader)
    return result


def baseline_contract():
    return dict(variant=VARIANT,transform=TRANSFORM,
        measured_radar_used=False,
        actual_inputs=['bbox_center_ray_origins','bbox_center_ray_directions',
                       'observer_navigation','public_range_angle_axes_and_ray_support'],
        raw_rgb_appearance_used=False,public_sensor_geometry_retained=True,
        sensor_cache_modified=False,original_source_modified=False,
        quality_gate='valid_visual_rays_and_geometry_only; radar signal gate not applicable',
        source_sha256=diagnostic_source_hashes(),
        original_core_source_sha256=gt._source_hashes(),
        interpretation='matched retrained control; not a full RGB monocular-depth baseline')


def remove_radar_measurements(features):
    """New arrays, unchanged public geometry; neither mutate nor rewrite the cache."""
    validate_features(features)
    original_digest=feature_digest(features)
    result=dict(features)
    for name in ('ra_power','re_power'):
        result[name]=np.zeros_like(np.asarray(features[name]))
    metadata=deepcopy(features['metadata'])
    metadata['diagnostic_input']=dict(variant=VARIANT,transform=TRANSFORM,
        measured_radar_used=False,original_feature_digest=original_digest)
    result['metadata']=metadata
    validate_features(result)
    return result


def geometry_only_rejection_reason(features,config):
    """An explicit visual-control contract, never a replacement fusion gate."""
    validate_features(features)
    if any(np.any(np.asarray(features[name])!=0) for name in ('ra_power','re_power')):
        raise ValueError('Visual-only baseline received measured radar power')
    count=len(features['ray_directions'])
    if not count:return 'no_detections'
    if count>int(config.get('group_max_members',24)):return 'unsupported_detected_count'
    if not np.asarray(features['ray_valid'],bool).any():return 'no_valid_rays'
    return None


def make_configuration(config,output):
    result=deepcopy(config)
    result.update(world_checkpoint=str(Path(output).resolve()/'world.best.pt'),
        diagnostic_input_variant=VARIANT,diagnostic_baseline_contract=baseline_contract())
    for name in ('world_model_trained','completed_optimizer_steps','formation_head_trained'):
        result.pop(name,None)
    return result


@contextmanager
def visual_baseline_context(prepared_loader):
    """Install scoped control hooks; restore every original callable on exit.

    prepared_loader(config, ids, split, count=0) returns the same three values as
    group_training.prepare, after validating the *original* cached inputs and
    reading supervision separately. It must never use baseline feature digests
    to claim the original sensor cache has been changed.
    """
    original_prepare=gt.prepare;original_evaluate=gt.evaluate_prior
    original_checkpoint=gt.checkpoint;original_write=gt.write_json
    initial=baseline_contract()

    def assert_frozen():
        if (diagnostic_source_hashes()!=initial['source_sha256']
                or gt._source_hashes()!=initial['original_core_source_sha256']):
            raise ValueError('Baseline training source changed during the experiment')

    def prepare(dataset,config,ids,split,count,device,output,rank_status=None):
        assert_frozen()
        samples,rows,frontend=prepared_loader(config,ids,split,count=count)
        samples=[dict(sample) for sample in samples];rows=[deepcopy(row) for row in rows]
        by_key={(sample['episode_id'],int(sample['frame_idx'])):sample for sample in samples}
        transformed=[]
        for row in rows:
            key=(row['episode_id'],int(row['frame_idx']))
            sample=by_key.get(key)
            if sample is not None:
                original=sample['features'];original_digest=feature_digest(original)
                features=remove_radar_measurements(original);sample['features']=features
                reason=geometry_only_rejection_reason(features,config)
                if reason is not None:raise ValueError('Prepared baseline sample has unusable visual geometry: '+reason)
                row.update(original_feature_digest=original_digest,
                    diagnostic_feature_digest=feature_digest(features),
                    feature_digest=feature_digest(features),
                    diagnostic_transform=TRANSFORM,
                    original_cache_retained=True)
                transformed.append(sample)
            row['diagnostic_input_variant']=VARIANT
        if len(transformed)!=len(samples):raise ValueError('Baseline sample/manifest keys differ')
        metadata=dict(frontend,model_input_variant=VARIANT,measured_radar_used=False,
                      diagnostic_baseline_contract=initial)
        gt.write_json(Path(output)/f'{split}_features.json',dict(schema=GROUP_SCHEMA,split=split,
            episodes=list(ids),frames=rows,frontend=metadata,source_sha256=gt._source_hashes(),
            diagnostic_baseline_contract=initial,original_cache_files_unchanged=True))
        if rank_status is not None:
            path,status=rank_status
            gt.write_json(path,dict(status,stage='preparing',prepared_frames=len(rows),prepared_groups=len(transformed)))
        return transformed,rows,metadata

    def evaluate(*args,**kwargs):
        report=original_evaluate(*args,**kwargs)
        report['summary'].update(model_input_variant=VARIANT,measured_radar_used=False,
            comparison_contract='retrained bbox-centre ray/navigation/public-geometry baseline')
        for row in report['frames']:
            if row.get('observation_status')=='valid_uncalibrated':row['observation_status']='valid_visual_geometry_only'
        return report

    def checkpoint(model,steps,**metadata):
        assert_frozen()
        value=original_checkpoint(model,steps,**metadata)
        value.update(kind=CHECKPOINT_KIND,diagnostic_baseline_contract=initial)
        return value

    def write(path,value):
        if isinstance(value,dict):
            value=dict(value,diagnostic_input_variant=VARIANT)
            if Path(path).name=='run_manifest.json':
                value.update(model_variant=VARIANT,diagnostic_baseline_contract=initial,
                    world_model_inputs=initial['actual_inputs'],
                    decoder='same architecture; zero measured RA/RE; visual geometry and navigation only',
                    raw_measured_radar_used=False)
        return original_write(path,value)

    with ExitStack() as stack:
        stack.enter_context(patch.object(gt,'prepare',prepare))
        stack.enter_context(patch.object(gt,'evaluate_prior',evaluate))
        stack.enter_context(patch.object(gt,'observation_rejection_reason',geometry_only_rejection_reason))
        stack.enter_context(patch.object(gt,'checkpoint',checkpoint))
        stack.enter_context(patch.object(gt,'write_json',write))
        yield initial


def load_visual_baseline(path,config=None,device='cpu'):
    """Strict standalone loader; the normal fusion checkpoint loader rejects this kind."""
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value.get('kind')!=CHECKPOINT_KIND or value.get('group_schema')!=GROUP_SCHEMA
            or int(value.get('steps',0))<1 or value.get('model_trained') is not True):
        raise ValueError('A genuinely trained diagnostic visual baseline is required')
    expected=baseline_contract()
    if value.get('diagnostic_baseline_contract')!=expected:
        raise ValueError('Visual baseline source/transform identity changed')
    if value.get('feature_source_sha256')!=expected['original_core_source_sha256']:
        raise ValueError('Visual baseline underlying core source changed')
    model_config=value['model_config']
    if (model_config.get('diagnostic_input_variant')!=VARIANT
            or model_config.get('diagnostic_baseline_contract')!=expected):
        raise ValueError('Visual baseline configuration lost its measurement-removal contract')
    if config is not None:
        for name in ('architecture_schema','group_hidden_dim','group_latent_dim','group_max_members',
                     'group_attention_heads','group_interaction_layers'):
            if name in config and config[name]!=model_config.get(name):
                raise ValueError('Visual baseline model architecture differs: '+name)
    model=GroupSceneModel(model_config);model.load_state_dict(value['state_dict'],strict=True)
    model.to(device).eval();model.trained=True
    model.checkpoint_metadata={key:item for key,item in value.items() if key!='state_dict'}
    return model


def run_training(args,prepared_loader):
    output=Path(args.output).resolve();source=Path(args.config).resolve()
    original=json.loads(source.read_text())
    reference_checkpoint=Path(original.get('world_checkpoint','')).resolve()
    if output==reference_checkpoint.parent:
        raise ValueError('Visual baseline must use a separate output directory from the fusion weights')
    output.mkdir(parents=True,exist_ok=True)
    configuration=make_configuration(original,output)
    private=output/'baseline_input_configuration.json'
    # torchrun ranks initialize before TrainingWorld exists. Serialize only this
    # small config publication; never make six ranks contend for a run-long lock.
    with (output/'baseline_configuration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if private.exists() and json.loads(private.read_text())!=configuration:
            raise ValueError('Existing baseline output belongs to a different configuration/source')
        if not private.exists():
            with tempfile.NamedTemporaryFile(mode='w',dir=output,delete=False) as stream:
                json.dump(configuration,stream,ensure_ascii=False,indent=2);stream.write('\n');name=stream.name
            os.replace(name,private)
    owned=SimpleNamespace(**vars(args));owned.config=private;owned.output=output
    owner=None
    try:
        if int(os.environ.get('RANK','0'))==0:
            owner=(output/'baseline_training.lock').open('a')
            try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise RuntimeError('This visual baseline already has a running trainer') from error
        with visual_baseline_context(prepared_loader):
            result=gt.train(owned)
    finally:
        if owner is not None:owner.close()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--max-episodes',type=int,default=0)
    parser.add_argument('--frames-per-episode',type=int,default=0)
    parser.add_argument('--validation-episodes',type=int,default=0)
    parser.add_argument('--validation-frames',type=int,default=0)
    parser.add_argument('--validate-every',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--physics-every',type=int,default=0)
    parser.add_argument('--physics-weight',type=float,default=0.)
    parser.add_argument('--detection-dropout',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=20261118)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume',type=Path)
    args=parser.parse_args(argv)
    from .data import load_cached_samples
    run_training(args,load_cached_samples)
    return 0


if __name__=='__main__':raise SystemExit(main())
