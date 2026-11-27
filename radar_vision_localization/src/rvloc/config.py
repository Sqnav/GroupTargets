"""Source-only project configuration, method identities and storage separation."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json


METHODS={
    1:dict(name='fft',input='radar',model=None,temporal=None),
    2:dict(name='rgb',input='rgb',model='rgb',temporal=None),
    3:dict(name='single_frame',input='rgb_radar',model='single_frame',temporal=None),
    4:dict(name='fft_kf',input='radar',model=None,temporal='kalman'),
    5:dict(name='rgb_kf',input='rgb',model='rgb',temporal='kalman'),
    6:dict(name='single_frame_kf',input='rgb_radar',model='single_frame',temporal='kalman'),
    7:dict(name='world_fusion',input='rgb_radar',model='world_fusion',temporal='internal_world_model'),
}

NEURAL_ARCHITECTURE='single_query_depth_v3'


def _single_query_model_contract(value):
    if not isinstance(value,dict):raise ValueError('model must be an object')
    if value.get('architecture',NEURAL_ARCHITECTURE)!=NEURAL_ARCHITECTURE:
        raise ValueError('The active seven-method experiment requires single_query_depth_v3')
    if value.get('architecture_schema',NEURAL_ARCHITECTURE)!=NEURAL_ARCHITECTURE:
        raise ValueError('The model architecture_schema must be single_query_depth_v3')
    steps=value.get('depth_refinement_steps',0)
    if type(steps) is not int or steps!=0:
        raise ValueError('Single-query methods have no iterative depth-refinement stages')
    if 'query_rounds' in value and (type(value['query_rounds']) is not int or value['query_rounds'] not in (0,1)):
        raise ValueError('Only RGB zero-query or fusion one-query configurations are supported')
    if 'local_query_enabled' in value and type(value['local_query_enabled']) is not bool:
        raise ValueError('local_query_enabled must be boolean when explicitly supplied')
    for key in ('local_query','d_evidence_candidate_sequence','d_evidence_aligned_first_query'):
        if key in value and value[key] is not False:
            raise ValueError('Local candidate/refinement switches have been removed: '+key)
    retired={'depth_offsets_m','depth_fine_offsets_m','depth_max_correction_m',
             'depth_fine_max_correction_m','depth_initial_loss_weight',
             'depth_geometry_enabled','d_evidence_capture_candidates'}
    if retired.intersection(value):
        raise ValueError('Remove retired refinement/initial-loss settings: '+str(sorted(retired.intersection(value))))


def code_root():
    return Path(__file__).resolve().parents[2]


def external_path(value, *, field, allow_none=False, base=None):
    if value is None and allow_none:return None
    if not isinstance(value,(str,Path)) or not str(value):raise ValueError(f'{field} must be an external path')
    p=Path(value).expanduser()
    p=(Path(base or code_root())/p).resolve() if not p.is_absolute() else p.resolve()
    if p==code_root() or p.is_relative_to(code_root()):
        raise ValueError(f'{field} cannot be stored in the code-only project: {p}')
    return p


def resolve_method(value):
    for i,m in METHODS.items():
        if str(value) in (str(i),m['name']):return i,dict(m)
    raise ValueError('Method must be 1..7 or one of: '+', '.join(m['name'] for m in METHODS.values()))


def _forbid_risk(value,path='config'):
    if isinstance(value,dict):
        for k,v in value.items():
            key=str(k).lower()
            if any(s in key for s in ('risk','oracle','ground_truth_input','teacher_state')):
                raise ValueError(f'Forbidden inference/config capability: {path}.{k}')
            _forbid_risk(v,path+'.'+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):_forbid_risk(v,f'{path}[{i}]')


@dataclass(frozen=True)
class ExperimentConfig:
    path:Path
    raw:dict
    manifest:Path|None
    output_root:Path
    cache_root:Path

    @classmethod
    def load(cls,path):
        p=Path(path).resolve();raw=json.loads(p.read_text());_forbid_risk(raw)
        if raw.get('schema')!='rvloc.experiment.v1':raise ValueError('Expected rvloc.experiment.v1')
        if raw.get('bandwidth_hz')!=32e6:raise ValueError('This experiment fixes bandwidth to32 MHz')
        if raw.get('center_contract')!='observer_scatter_at_camera_optical_center':raise ValueError('Explicit co-centered contract required')
        if raw.get('evaluation',{}).get('primary')!='position_rmse_3d_m':raise ValueError('Primary must be3D position RMSE')
        _single_query_model_contract(raw.get('model',{}))
        storage=raw.get('storage',{})
        output=external_path(storage.get('output_root'),field='output_root',base=p.parent)
        cache=external_path(storage.get('cache_root'),field='cache_root',base=p.parent)
        manifest=external_path(raw.get('manifest'),field='manifest',allow_none=True,base=p.parent)
        external_path(raw.get('calibration_artifact'),field='calibration_artifact',allow_none=True,base=p.parent)
        for name,value in raw.get('checkpoints',{}).items():external_path(value,field='checkpoint '+name,allow_none=True,base=p.parent)
        result=cls(p,raw,manifest,output,cache)
        result.weights_root  # Validate effective storage separation without creating directories.
        return result

    @property
    def weights_root(self):
        """Dedicated model storage, with a compatible sibling default.

        Keeping this computed preserves the positional dataclass constructor
        used by existing clients. Older configurations without weights_root
        use a sibling weights directory, never the result run directory.
        """
        value=self.raw.get('storage',{}).get('weights_root')
        weights=external_path(value if value is not None else self.output_root.parent/'weights',
                              field='weights_root',base=self.path.parent)
        for field,other in (('output_root',self.output_root),('cache_root',self.cache_root)):
            other=Path(other).resolve()
            if weights==other or weights.is_relative_to(other) or other.is_relative_to(weights):
                raise ValueError(f'weights_root must be separate from {field}')
        return weights

    def checkpoint(self,name):
        return external_path(self.raw.get('checkpoints',{}).get(name),field='checkpoint '+name,allow_none=True,base=self.path.parent)

    @property
    def sha256(self):return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def plan(self):
        return dict(status='code_ready_data_pending' if self.manifest is None else 'configured_not_executed',
            methods=[dict(id=i,**m) for i,m in METHODS.items()],manifest=str(self.manifest) if self.manifest else None,
            outputs=str(self.output_root),weights=str(self.weights_root),cache=str(self.cache_root),
            checkpoints={k:str(self.checkpoint(k)) if self.checkpoint(k) else None for k in self.raw.get('checkpoints',{})},
            neural_architecture=NEURAL_ARCHITECTURE,depth_refinement_steps=0,
            query_rounds=dict(rgb=0,single_frame=1,world_fusion=1),
            geometry_contract='shared_public_camera_ray_support',
            config_sha256=self.sha256,training_started=False,evaluation_started=False)
