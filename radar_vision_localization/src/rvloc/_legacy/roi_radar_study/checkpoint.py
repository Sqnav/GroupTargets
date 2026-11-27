"""Independent inference exports and exact epoch resumes for three modalities."""
from copy import deepcopy
from pathlib import Path
import re

import torch

from d_evidence_speed import checkpoint as execution
from roi_followup.checkpoint import source_hashes as preserved_sources
from rvwm.data import sha256
from rvwm.group_training import atomic_torch_save

ROOT=Path(__file__).resolve().parents[1]
KIND='roi_radar_epoch_depth_v1'
RESUME_KIND='roi_radar_epoch_resume_v1'
MODEL_FACTORY='roi_radar_study.model.StudyDepthLocalizer'
ARMS=('visual','32mhz','64mhz')


def source_hashes():
    result=preserved_sources()
    result.update({'roi_radar_study/'+p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    return result


def validate_config(config):
    if not isinstance(config,dict) or config.get('study_arm') not in ARMS:
        raise ValueError('An explicit visual/32mhz/64mhz arm is required')
    if config.get('d_evidence_arm')!='roi':
        raise ValueError('Keep the same authenticated frozen-ROI visual input')
    checksum=config.get('study_protocol_sha256')
    if not isinstance(checksum,str) or not re.fullmatch('[0-9a-f]{64}',checksum):
        raise ValueError('A frozen study protocol SHA256 is required')
    if config.get('study_protocol_path') and sha256(config['study_protocol_path'])!=checksum:
        raise ValueError('The frozen study protocol changed')
    expected='none' if config['study_arm']=='visual' else 'ra_re'
    if config.get('precision_radar_mode')!=expected:
        raise ValueError('Visual/radar configuration does not match the explicit input modality')
    if (config['study_arm']=='visual') != (config.get('precision_mode')=='visual'):
        raise ValueError('Evaluation input gate must match the actual visual/radar modality')
    return config['study_arm'],checksum


def _identity(model):
    from .model import StudyDepthLocalizer
    if not isinstance(model,StudyDepthLocalizer):
        raise ValueError('The actual study model factory is required')
    arm,checksum=validate_config(model.config)
    return dict(model_factory=MODEL_FACTORY,study_arm=arm,study_protocol_sha256=checksum)


def make_checkpoint(model,*,epoch,optimizer_updates,identity,validation=None):
    if (not model.trained or isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<1
            or isinstance(optimizer_updates,bool) or not isinstance(optimizer_updates,int) or optimizer_updates<1):
        raise ValueError('Actual trained weights and a complete epoch are required')
    return dict(kind=KIND,**_identity(model),model_config=deepcopy(model.config),
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        completed_epochs=epoch,optimizer_updates=optimizer_updates,model_trained=True,
        source_sha256=source_hashes(),training_identity=deepcopy(identity),
        validation=None if validation is None else dict(epoch=validation['epoch'],summary=deepcopy(validation['summary'])),
        input_frames=1,detector_frozen=True,production_default_changed=False,
        output='one current visual-ray depth per observed detection')


def save(value,path):
    atomic_torch_save(value,Path(path))


def _check(value):
    arm,checksum=validate_config(value.get('model_config'))
    if (value.get('model_factory')!=MODEL_FACTORY or value.get('study_arm')!=arm
            or value.get('study_protocol_sha256')!=checksum or value.get('source_sha256')!=source_hashes()):
        raise ValueError('Actual model factory, modality, protocol or sources changed')


def load_checkpoint(path,device='cpu'):
    value=torch.load(Path(path),map_location='cpu',weights_only=False)
    if value.get('kind')!=KIND or value.get('model_trained') is not True:
        raise ValueError('An independently trained ROI/radar study checkpoint is required')
    _check(value)
    for name in ('completed_epochs','optimizer_updates'):
        if isinstance(value.get(name),bool) or not isinstance(value.get(name),int) or value[name]<1:
            raise ValueError('A real completed epoch/update boundary is required')
    from .model import StudyDepthLocalizer
    model=StudyDepthLocalizer(value['model_config']);model.load_state_dict(value['state_dict'],strict=True)
    model.to(device).eval();model.trained=True
    model.checkpoint_metadata={k:v for k,v in value.items() if k!='state_dict'}
    return model


def make_resume(model,optimizer,sampler,*,ranks,identity,progress,history,best,elapsed_s,epoch_totals,inference_exports):
    return dict(kind=RESUME_KIND,**_identity(model),source_sha256=source_hashes(),
        runtime_identity=execution.runtime_identity(),training_identity=deepcopy(identity),
        model_config=deepcopy(model.config),model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state_dict=deepcopy(optimizer.state_dict()),rank_states=deepcopy(ranks),
        optimizer_updates=sampler.state_dict()['optimizer_step'],progress=deepcopy(progress),
        history=deepcopy(history),best=deepcopy(best),elapsed_s=float(elapsed_s),
        epoch_totals=deepcopy(epoch_totals),inference_exports=deepcopy(inference_exports))


def load_resume(path,*,identity,world_size):
    value=torch.load(Path(path),map_location='cpu',weights_only=False)
    required={'model_state_dict','optimizer_state_dict','rank_states','history','best','progress',
              'epoch_totals','inference_exports','elapsed_s'}
    if value.get('kind')!=RESUME_KIND or not required.issubset(value):
        raise ValueError('Exact resume requires this study complete epoch_resume.pt')
    _check(value)
    if value.get('runtime_identity')!=execution.runtime_identity() or value.get('training_identity')!=identity:
        raise ValueError('Numerical runtime, data, batch, topology or budget changed')
    ranks=value['rank_states'];updates=value.get('optimizer_updates')
    if (not isinstance(ranks,list) or len(ranks)!=world_size
            or [r.get('rank') for r in ranks]!=list(range(world_size))):
        raise ValueError('Distributed rank snapshots incomplete')
    if (isinstance(updates,bool) or not isinstance(updates,int) or updates<1
            or any(r.get('sampler',{}).get('optimizer_step')!=updates for r in ranks)
            or value['progress'].get('optimizer_updates')!=updates):
        raise ValueError('Sampler, progress and optimizer boundary differ')
    if (len(value['history'])!=updates or [r['optimizer_updates'] for r in value['history']]!=list(range(1,updates+1))):
        raise ValueError('Committed metric history differs')
    for r in ranks:
        if (r['sampler']['draws']!=value['progress']['frames_seen']
                or r['sampler']['completed_epochs']!=value['progress']['completed_epochs']):
            raise ValueError('Sampler epoch/frame boundary differs')
    for export in value['inference_exports'].values():
        if export.get('kind')!=KIND or export.get('optimizer_updates',updates+1)>updates:
            raise ValueError('Incompatible inference export in resume')
        _check(export)
    return value


capture_rank_state=execution.capture_rank_state
restore_rank_state=execution.restore_rank_state
