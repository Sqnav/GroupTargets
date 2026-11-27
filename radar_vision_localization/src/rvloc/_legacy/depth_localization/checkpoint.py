"""Separate, exact-source-bound weights for the new deterministic depth model."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import tempfile
import os
import torch
from rvwm.data import sha256
from rvwm.group_training import _source_hashes

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='single_frame_geometry_guided_depth_v2'
FEATURE_SCHEMA='bbox_scale_base_rays_ra_re_v2'
KIND='geometry_guided_single_frame_depth'


def source_hashes():
    values={'rvwm/'+name:value for name,value in _source_hashes().items()}
    values.update({'depth_localization/'+path.name:sha256(path)
                   for path in sorted(Path(__file__).parent.glob('*.py'))})
    values.update({'diagnostics/'+name:sha256(ROOT/'diagnostics'/name)
                   for name in ('data.py','geometry.py')})
    return values


def make_checkpoint(model,steps,**metadata):
    if isinstance(steps,bool) or not isinstance(steps,int) or steps<1 or model.trained is not True:
        raise ValueError('Depth checkpoint requires completed optimizer steps and a trained model')
    if model.config.get('architecture_schema')!=SCHEMA:raise ValueError('Wrong depth architecture')
    reserved={'kind','schema','feature_schema','steps','model_trained','model_config','state_dict','source_sha256',
              'input_frames','output','sampled_layouts','uncertainty_calibrated'}
    if reserved&metadata.keys():raise ValueError('Cannot override depth checkpoint identity')
    return dict(kind=KIND,schema=SCHEMA,feature_schema=FEATURE_SCHEMA,steps=steps,model_trained=True,
        model_config=deepcopy(model.config),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        source_sha256=source_hashes(),input_frames=1,output='one continuous camera-ray slant depth per detected member',
        sampled_layouts=False,uncertainty_calibrated=False,**metadata)


def save_checkpoint(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:temporary=Path(stream.name)
    try:
        torch.save(value,temporary)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def load_checkpoint(path,config=None,device='cpu'):
    path=Path(path).resolve();value=torch.load(path,map_location='cpu',weights_only=False)
    if (value.get('kind')!=KIND or value.get('schema')!=SCHEMA or value.get('feature_schema')!=FEATURE_SCHEMA
            or value.get('model_trained') is not True or isinstance(value.get('steps'),bool)
            or not isinstance(value.get('steps'),int) or value['steps']<1):
        raise ValueError('A genuinely trained matching depth-v2 checkpoint is required; legacy weights are incompatible')
    if value.get('source_sha256')!=source_hashes():raise ValueError('Depth model/input source changed after training')
    trained=value['model_config']
    if trained.get('architecture_schema')!=SCHEMA:raise ValueError('Checkpoint model configuration differs')
    if config is not None:
        runtime={'depth_checkpoint','depth_feature_cache','depth_model_trained','depth_initialization_m'}
        names={k for k in set(trained)|set(config) if k.startswith('depth_') and k not in runtime}
        names|={'architecture_schema','window','visual_point_mode','ra_re_window','ra_re_azimuth_bins',
                'ra_re_elevation_bins','ra_re_spatial_oversampling','detector_weights','detector_confidence',
                'split_manifest','split_manifest_sha256','track_detection_dir','dataset_root','echo_subdir'}
        for name in names:
            if config.get(name)!=trained.get(name):raise ValueError('Deployed depth configuration differs: '+name)
    from .model import DepthLocalizer
    model=DepthLocalizer(trained);model.load_state_dict(value['state_dict'],strict=True)
    model.to(device).eval();model.trained=True
    model.checkpoint_metadata={key:item for key,item in value.items() if key!='state_dict'}
    model.checkpoint_metadata['checkpoint_sha256']=sha256(path)
    return model
