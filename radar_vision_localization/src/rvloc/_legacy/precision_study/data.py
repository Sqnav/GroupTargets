"""Separate supervision and strict current-frame measured sensor inputs.

Training/evaluation may explicitly load labels. Ordinary inference uses the
allowlist sensor reader, never Dataset._load/targets/observations. Visual-only
mode never opens IQ or either radar feature cache.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from rvwm.data import Dataset, sha256
from rvwm.group_observation import build_spectral_features, spectra_from_iq
from depth_localization.features import enrich_features, validate_depth_features
from .visual_data import VisualSensorReader

DETECTION_FIELDS = ('ray_origins','ray_directions','ray_valid','ray_depth_bounds',
                    'bbox_xyxy','detection_scores','visual_scale')


def drop_detection_features(features, retained_bool):
    if features.get('schema') == 'visual_bbox_base_rays_v1':
        from .visual import drop_visual_detections
        return drop_visual_detections(features, retained_bool)
    validate_depth_features({k:v for k,v in features.items() if k != 'joint_volume'})
    keep = np.asarray(retained_bool)
    if keep.dtype != np.bool_ or keep.shape != (len(features['ray_directions']),):
        raise ValueError('A detection-aligned boolean augmentation mask is required')
    # Shared measured arrays are immutable. Do not duplicate a 2MB volume for
    # each detection-dropout update or change it when removing visual rays.
    result = dict(features)
    for key in DETECTION_FIELDS:
        result[key] = np.asarray(features[key])[keep].copy()
    return result


def _measured_maps(config, sensor, episode, index, split, want_volume):
    volume = None; spectra = None
    if config.get('precision_volume_cache'):
        try:
            from .cache import load_volume, load_spectra
            if want_volume:
                volume = load_volume(config, episode, index, split)
            spectra = load_spectra(config, episode, index, split)
            spectra['metadata'] = dict(spectra.get('metadata',{}),
                runtime_native_iq_read=False, loading_method='authenticated unchanged RA/RE cache')
        except FileNotFoundError:
            # Unseen deployment frames can be processed from their measured
            # IQ. Authentication errors are not caught or silently bypassed.
            volume = None; spectra = None
    if spectra is None or (want_volume and volume is None):
        if sha256(sensor['iq_path']) != sensor['iq_sha256']:
            raise ValueError('Current measured IQ differs from its frozen source receipt')
        iq = np.load(sensor['iq_path'], allow_pickle=False)
        spectra = spectra_from_iq(iq, sensor['radar'], config)
        spectra['metadata'] = dict(spectra.get('metadata',{}),
            runtime_native_iq_read=True, loading_method='native current-frame measured IQ')
        if want_volume:
            from .volume import build_power_volume
            built = build_power_volume(iq, sensor['radar'], config)
            volume = {k:built[k] for k in ('volume_power','range_m','u_axis','v_axis','angular_valid','metadata')}
            volume['metadata'].update(uses_labels=False,runtime_native_iq_read=True,
                loading_method='native current-frame IQ; no cache written')
    return spectra, volume


def _strict_sample(config, reader, episode, index):
    observed, row, station = reader.observation(episode, index)
    vf = observed['features']
    sensor = reader.sensor_frame(episode, index)
    joint = config.get('precision_radar_mode', 'ra_re') == 'joint_volume'
    maps, volume = _measured_maps(config, sensor, episode, index, reader.split, joint)
    sensor_receipt = dict(vf['metadata'],radar_features_read=True,
        iq_read=bool(maps['metadata'].get('runtime_native_iq_read')),
        precision_radar_mode=config.get('precision_radar_mode','ra_re'))
    base = build_spectral_features(ray_origins=vf['ray_origins'], ray_directions=vf['ray_directions'],
        observer_position=vf['observer_position'], range_scale_m=vf['range_scale_m'],
        ray_valid=vf['ray_valid'], ra_power=maps['ra_power'], re_power=maps['re_power'],
        range_m=maps['range_m'], azimuth_rad=maps['azimuth_rad'], elevation_rad=maps['elevation_rad'],
        ra_valid=maps.get('ra_valid'), re_valid=maps.get('re_valid'),
        metadata=dict(sensor_receipt, spectrum=maps.get('metadata',{}), coordinate_frame='base_local'))
    detections = [dict(bbox_xyxy=b,score=s,center_px=(b[:2]+b[2:])/2)
                  for b,s in zip(vf['bbox_xyxy'],vf['detection_scores'])]
    features = enrich_features(base, detections, vf['camera_K'], vf['camera_image_size_wh'],
        provenance=dict(sensor_receipt,ordinary_inference=True,labels_read=False,history_read=False,future_read=False))
    if joint:
        features['joint_volume'] = volume
    observed['features'] = features
    row['hashes'] = dict(row['hashes'],iq=sensor['iq_sha256'])
    return observed, row, station


def load_samples(config, ids, split, count=0, include_labels=True):
    if type(include_labels) is not bool:
        raise ValueError('Explicit boolean label-access switch required')
    mode = config.get('precision_radar_mode', 'ra_re')
    if mode == 'none':
        from .visual_data import load_samples as load_visual
        return load_visual(config, ids, split, count, include_labels)
    if mode not in ('ra_re','joint_volume'):
        raise ValueError('Unknown measured modality')
    if include_labels and split in ('train','val'):
        # Explicit training/offline-scoring path reuses the exact authenticated
        # original RA/RE, so the query experiment changes no global preprocessing.
        from depth_localization.data import load_samples as load_original
        samples, rows, frontend = load_original(config, ids, split, count, True)
        if mode == 'joint_volume':
            from .cache import load_volume
            for sample, row in zip(samples, rows):
                volume = load_volume(config,sample['episode_id'],sample['frame_idx'],split)
                sample['features'] = dict(sample['features'], joint_volume=volume)
                row['joint_volume_receipt'] = deepcopy(volume['metadata'])
        return samples, rows, dict(frontend,precision_radar_mode=mode)
    reader = VisualSensorReader(config, split)
    labels = Dataset(config['dataset_root'],echo_subdir=config.get('echo_subdir','echoes')) if include_labels else None
    samples=[];rows=[]
    try:
        for episode in ids:
            for index in reader.frame_indices(episode,count):
                sample,row,station = _strict_sample(config,reader,episode,index)
                if include_labels:
                    truth=labels.targets(episode,index)
                    sample.update(positions=station.to_local_positions(truth.positions).astype(np.float32),formation_index=None)
                    row['label_count']=len(truth.positions)
                samples.append(sample);rows.append(row)
        return samples,rows,dict(reader.metadata,precision_radar_mode=mode)
    finally:
        reader.close()


def load_observation_features(config, episode, frame_idx, split):
    if type(frame_idx) is not int or frame_idx < 0:
        raise ValueError('Current frame index must be a nonnegative integer')
    if config.get('precision_radar_mode','ra_re') == 'none':
        from .visual_data import load_observation_features as load_visual
        return load_visual(config,episode,frame_idx,split)
    reader = VisualSensorReader(config,split)
    try:
        return _strict_sample(config,reader,episode,frame_idx)[0]['features']
    finally:
        reader.close()
