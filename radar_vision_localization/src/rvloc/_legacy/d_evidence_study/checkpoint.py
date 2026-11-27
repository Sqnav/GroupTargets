"""Independent evidence-study checkpoints, bound to unchanged D code identity."""
from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import torch

from depth_localization.checkpoint import save_checkpoint
from precision_study.deployed_infer import DEFAULT_DESCRIPTOR,load_descriptor
from precision_study.deployed_train import source_identity
from precision_study.versions import frozen_study_package,load_frozen_study_model
from rvwm.data import sha256
from .model import EvidenceDepthLocalizer,SCHEMA,configured_flags

ROOT=Path(__file__).resolve().parents[1]
KIND='single_frame_d_evidence_depth_v1'
FEATURE_SCHEMA='bbox_scale_base_rays_ra_re_v2_optional_frozen_detector_roi_v1'


def backbone_identity():
    description,identity=load_descriptor(DEFAULT_DESCRIPTOR)
    if description['model_name']!='D_wide_residual_ra_re':
        raise ValueError('This study is bound to the authenticated original D')
    source_identity(description)
    alias,_=frozen_study_package(description['source_root'])
    return dict(identity,source_root=str(description['source_root']),
        source_sha256=importlib.import_module(alias+'.checkpoint').source_hashes())


def source_hashes():
    result=dict(backbone_identity()['source_sha256'])
    for name in ('model.py','checkpoint.py','data.py','train.py','roi_export.py'):
        result['d_evidence_study/'+name]=sha256(ROOT/'d_evidence_study'/name)
    for name in ('data.py','train.py'):
        result['bandwidth_study/'+name]=sha256(ROOT/'bandwidth_study'/name)
    return result


def make_checkpoint(model,steps,**metadata):
    if (not isinstance(model,EvidenceDepthLocalizer) or model.trained is not True
            or isinstance(steps,bool) or not isinstance(steps,int) or steps<1):
        raise ValueError('A trained evidence model and completed optimizer steps are required')
    reserved={'kind','schema','feature_schema','steps','model_trained','model_config','state_dict',
        'source_sha256','backbone_identity','input_frames','output','sampled_layouts',
        'uncertainty_calibrated','d_evidence_arm'}
    if reserved.intersection(metadata):
        raise ValueError('Cannot override independent evidence checkpoint identity')
    return dict(kind=KIND,schema=SCHEMA,feature_schema=FEATURE_SCHEMA,steps=steps,model_trained=True,
        model_config=deepcopy(model.config),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        source_sha256=source_hashes(),backbone_identity=backbone_identity(),input_frames=1,
        d_evidence_arm=model.evidence_arm,output='one continuous current camera-ray depth per detected member',
        sampled_layouts=False,uncertainty_calibrated=False,**metadata)


def load_checkpoint(path,config=None,device='cpu'):
    path=Path(path).resolve();value=torch.load(path,map_location='cpu',weights_only=False)
    if (value.get('kind')!=KIND or value.get('schema')!=SCHEMA or value.get('feature_schema')!=FEATURE_SCHEMA
            or value.get('model_trained') is not True or isinstance(value.get('steps'),bool)
            or not isinstance(value.get('steps'),int) or value['steps']<1):
        raise ValueError('Require an actual trained evidence checkpoint, without legacy fallback')
    if value.get('source_sha256')!=source_hashes() or value.get('backbone_identity')!=backbone_identity():
        raise ValueError('Evidence or original D model/data/training source identity changed')
    trained=value['model_config']
    if trained.get('d_evidence_schema')!=SCHEMA or trained.get('d_evidence_arm')!=value.get('d_evidence_arm'):
        raise ValueError('Evidence model configuration and arm identity differ')
    if config is not None:
        runtime={'device','depth_checkpoint','depth_model_trained','completed_optimizer_steps','d_evidence_checkpoint',
            'd_evidence_capture_candidates'}
        for key in (set(trained)|set(config))-runtime:
            if trained.get(key)!=config.get(key):
                raise ValueError('Evidence deployment configuration differs: '+key)
    model=EvidenceDepthLocalizer(trained);model.load_state_dict(value['state_dict'],strict=True)
    model.to(device).eval();model.trained=True
    model.checkpoint_metadata={k:v for k,v in value.items() if k!='state_dict'}
    model.checkpoint_metadata['checkpoint_sha256']=sha256(path)
    return model


def load_original_d_checkpoint(path,config=None,device='cpu'):
    """Explicit flags-off compatibility test only; never a training initializer."""
    description,_=load_descriptor(DEFAULT_DESCRIPTOR)
    source_identity(description)
    original=load_frozen_study_model(description['source_root'],path,None,device)
    selected=deepcopy(original.config if config is None else config)
    runtime={'device','depth_checkpoint','depth_model_trained','completed_optimizer_steps'}
    for key,value in original.config.items():
        if key not in runtime and selected.get(key)!=value:
            raise ValueError('Flags-off compatibility must retain the original D configuration: '+key)
    arm,flags=configured_flags(selected)
    if arm!='control' or any(flags.values()):
        raise ValueError('Original D weights may only be loaded with every evidence switch off')
    model=EvidenceDepthLocalizer(selected)
    model.load_state_dict(original.state_dict(),strict=True)
    model.to(device).eval();model.trained=True
    model.checkpoint_metadata=dict(original.checkpoint_metadata,
        flags_off_original_d_compatibility=True,evidence_training_initializer=False)
    return model
