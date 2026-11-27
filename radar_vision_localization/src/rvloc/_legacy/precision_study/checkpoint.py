"""Independent weights and exact provenance for the precision study.

Baseline source files are never modified or relabelled as new checkpoints.
Only model/data/training code participates in this fingerprint; offline report
formatting can evolve without invalidating a completed model.
"""
from copy import deepcopy
from pathlib import Path
import torch
from rvwm.data import sha256
from depth_localization.checkpoint import source_hashes as baseline_sources
from depth_localization.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'single_frame_geometry_guided_depth_v2'
KIND = 'single_frame_precision_study_depth_v1'
CORE = ('model.py', 'visual.py', 'visual_data.py', 'volume.py', 'cache.py',
        'data.py', 'training.py', 'checkpoint.py', 'evaluation.py')


def source_hashes():
    result = baseline_sources()
    for name in CORE:
        result['precision_study/' + name] = sha256(Path(__file__).parent / name)
    return result


def make_checkpoint(model, steps, **metadata):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or model.trained is not True:
        raise ValueError('A real optimizer update is required for a trained precision checkpoint')
    protected = {'kind', 'schema', 'steps', 'model_trained', 'model_config',
                 'state_dict', 'source_sha256', 'radar_mode', 'input_frames',
                 'uncertainty_calibrated', 'sampled_layouts'}
    if protected & metadata.keys():
        raise ValueError('Cannot override checkpoint identity')
    mode = model.config.get('precision_radar_mode', 'ra_re')
    if mode not in ('none', 'ra_re', 'joint_volume'):
        raise ValueError('Unknown measured-input mode')
    return dict(kind=KIND, schema=SCHEMA, steps=steps, model_trained=True,
        model_config=deepcopy(model.config), radar_mode=mode, input_frames=1,
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        source_sha256=source_hashes(), uncertainty_calibrated=False,
        sampled_layouts=False, **metadata)


def load_checkpoint(path, config=None, device='cpu'):
    path = Path(path).resolve()
    value = torch.load(path, map_location='cpu', weights_only=False)
    if (value.get('kind') != KIND or value.get('schema') != SCHEMA
            or value.get('model_trained') is not True
            or isinstance(value.get('steps'), bool) or not isinstance(value.get('steps'), int)
            or value['steps'] < 1):
        raise ValueError('Require a trained precision-study checkpoint, without legacy fallback')
    if value.get('source_sha256') != source_hashes():
        raise ValueError('Precision model, sensor reader or training sources changed')
    trained = value['model_config']
    mode = trained.get('precision_radar_mode', 'ra_re')
    if mode != value.get('radar_mode'):
        raise ValueError('Checkpoint input-mode identity differs')
    if config is not None:
        runtime = {'device', 'depth_checkpoint', 'depth_model_trained',
                   'completed_optimizer_steps', 'precision_checkpoint'}
        names = {k for k in set(trained) | set(config)
                 if k.startswith(('depth_', 'precision_')) and k not in runtime}
        names |= {'architecture_schema', 'window', 'visual_point_mode', 'ra_re_window',
                  'ra_re_azimuth_bins', 'ra_re_elevation_bins', 'ra_re_spatial_oversampling',
                  'detector_weights', 'detector_confidence', 'split_manifest',
                  'split_manifest_sha256', 'track_detection_dir', 'dataset_root', 'echo_subdir'}
        for name in names:
            if trained.get(name) != config.get(name):
                raise ValueError('Precision deployment configuration differs: ' + name)
    from .model import make_model
    model = make_model(trained)
    model.load_state_dict(value['state_dict'], strict=True)
    model.to(device).eval(); model.trained = True
    model.checkpoint_metadata = {k:v for k,v in value.items() if k != 'state_dict'}
    model.checkpoint_metadata['checkpoint_sha256'] = sha256(path)
    return model
