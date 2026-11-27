"""Read the original immutable prepared features, with separate GT for diagnostics.

Sensor payloads were hashed when these features were prepared for training.
This reader rechecks cache bytes, source manifests, feature digests and detector
receipts; it does not reread/FFT all native IQ or modify the prepared cache.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
import torch

from rvwm.coordinates import FixedStationFrame
from rvwm.data import Dataset,sha256
from rvwm.geometry import rays
from rvwm.group_model import FORMATIONS
from rvwm.group_observation import feature_digest,validate_features
from rvwm.partitions import active_splits
from rvwm.phase_selection import configured_phases,selected_frame_indices
from rvwm.types import Detection,Camera
from rvwm.visual_refinement import ExportedRGBFrontend

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):return json.loads(Path(path).read_text())


def original_root(config):
    return Path(config.get('diagnostic_training_root',ROOT/'results/training/single')).resolve()


def verify_original_sources(config):
    root=original_root(config);run=read_json(root/'run_manifest.json')
    if run.get('status')!='complete' or run.get('optimizer_steps')!=20000:
        raise ValueError('Diagnostics require the original complete 20k fusion experiment')
    for name,digest in run['source_sha256'].items():
        path=(ROOT/'rvwm'/name).resolve()
        if not path.is_relative_to(ROOT/'rvwm') or sha256(path)!=digest:
            raise ValueError('Original trained source changed: '+name)
    original=run['configuration']
    for name in ('dataset_root','split_manifest','visual_split_manifest','detector_weights','track_detection_dir'):
        if Path(config[name]).resolve()!=Path(original[name]).resolve():raise ValueError('Original input path differs: '+name)
    for name in ('split_manifest_sha256','detector_backend','detector_confidence','visual_point_mode','ra_re_window'):
        if config.get(name)!=original.get(name):raise ValueError('Original input setting differs: '+name)
    return run


@lru_cache(maxsize=8)
def _inventory(root,split):
    root=Path(root)
    if split=='train':
        inventory=read_json(root/'train_inventory.json')
        front=read_json(root/'ranks/rank_0/train_features.json')['frontend']
    elif split=='val':
        inventory=read_json(root/'val_features.json');front=inventory['frontend']
    else:raise ValueError('Prepared training cache only covers train/val; test requires separate sensor preprocessing')
    rows=inventory['frames'];lookup={(r['episode_id'],int(r['frame_idx'])):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate original cached frame inventory')
    return rows,lookup,front


def inventory_records(config,split):
    verify_original_sources(config)
    return _inventory(str(original_root(config)),split)[0]


def _world_frame(episode,loaded,index):
    row=loaded[5][index];station=np.asarray(row['station_position_ned_m'],float)
    angle=math.radians(float(row['station_heading_deg']));c,s=math.cos(angle),math.sin(angle)
    basis=np.array([[c,-s,0],[s,c,0],[0,0,-1.]])
    return FixedStationFrame(station,basis,episode)


def _frame(config,dataset,frontend,row,expected_sources):
    episode,index=row['episode_id'],row['frame_idx']
    if row['feature_status']!='prepared':
        raise ValueError('This cache reader requires explicit handling of excluded original frames')
    path=Path(row['cache']['path']).resolve()
    if not path.is_relative_to(Path(config['group_feature_cache']).resolve()) or sha256(path)!=row['cache']['sha256']:
        raise ValueError('Original feature cache bytes changed: '+str(path))
    value=torch.load(path,map_location='cpu',weights_only=False);features=value['features'];receipt=value['receipt']
    if (receipt['episode_id']!=episode or receipt['frame_idx']!=index
            or receipt['raw_hashes']!=row['hashes'] or receipt['source_sha256']!=expected_sources
            or value['feature_digest']!=row['feature_digest'] or feature_digest(features)!=row['feature_digest']):
        raise ValueError('Cached frame provenance differs from original training inventory')
    validate_features(features)
    loaded=dataset._load(episode);public=loaded[6]
    if public['capture_hash']!=row['hashes']['source_trajectory'] or public['metadata_hash']!=row['hashes']['source_metadata']:
        raise ValueError('Current capture/metadata differs from cached source')
    sensor=receipt['sensor_identity'];width,height=sensor['camera']['width'],sensor['camera']['height']
    # The feature cache hashes dtype+shape+pixels; the detector hashes pixels
    # alone. Those different digests cannot be compared. The immutable cache
    # and detector export share the exact encoded RGB SHA and capture SHA.
    boxes,scores=frontend.track_store.get(episode,index,row['hashes']['rgb'],
        source_frames_sha256=public['capture_hash'],image_size_wh=(width,height))
    uv=(boxes[:,:2]+boxes[:,2:])/2
    expected_uv=np.asarray([r['uv'] for r in receipt['detections']],float).reshape(-1,2)
    if len(boxes)!=row['detected_count'] or not np.array_equal(uv,expected_uv):raise ValueError('Detection export differs from cached rays')
    camera=Camera(np.asarray(sensor['camera']['K']),np.asarray(sensor['camera']['position']),
        np.asarray(sensor['camera']['rotation']),width,height)
    if not np.allclose(rays(uv,camera),features['ray_directions'],rtol=0,atol=1e-7):raise ValueError('Cached ray directions do not match detector boxes')
    frame=_world_frame(episode,loaded,index)
    observed_camera=np.asarray(loaded[4][index]['camera']['camera_position_ned'],float)
    if not np.allclose(frame.to_local_positions(observed_camera),features['ray_origins'],rtol=0,atol=1e-6):
        raise ValueError('Cached rays use another fixed-station coordinate transform')
    # Source features are verified before the following evaluation/training labels are opened.
    targets=dataset.targets(episode,index);positions=frame.to_local_positions(targets.positions).astype(np.float32)
    metadata=loaded[2];name=metadata.get('group_structure',metadata.get('scenario_spec',{}).get('formation_type'))
    detections=[Detection(point.copy(),np.eye(2)*float(config.get('pixel_std',3.))**2,box.copy(),float(score),np.zeros(128))
                for point,box,score in zip(uv,boxes,scores)]
    return dict(features=features,positions=positions,episode_id=episode,frame_idx=index,detections=detections,
        formation_index=FORMATIONS.index(name) if name in FORMATIONS else None,formation_type=name or 'unknown',
        complete_detection_cardinality=len(detections)==len(positions))


def load_cached_samples(config,ids,split,count=0):
    run=verify_original_sources(config);_,lookup,stored_frontend=_inventory(str(original_root(config)),split)
    dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir'],verify_sources=True)
    partitions=active_splits(dataset,config)
    if not set(ids)<=set(partitions[split]):raise ValueError('Cached diagnostics requested episodes outside frozen split')
    frontend=ExportedRGBFrontend(config,split=split,device='cpu')
    for key in ('weights_sha256','detector_contract','track_detection_manifest_sha256','visual_split_manifest_sha256'):
        if frontend.metadata[key]!=stored_frontend[key]:raise ValueError('Detector identity changed: '+key)
    samples=[];manifest=[]
    try:
        for episode in ids:
            for index in selected_frame_indices(dataset,episode,configured_phases(config),count):
                key=(episode,index)
                if key not in lookup:raise ValueError('Original prepared inventory is missing '+str(key))
                row=lookup[key];sample=_frame(config,dataset,frontend,row,run['source_sha256'])
                samples.append(sample);manifest.append(deepcopy(row))
        result=deepcopy(stored_frontend)
        result['diagnostic_sensor_loading']='original authenticated feature cache; native IQ/RGB payloads not reread'
        return samples,manifest,result
    finally:frontend.close()
