"""Authenticated bbox/navigation input without IQ, spectra, depth or target reads.

Capture JSONL mixes sensor and supervision fields. The scanner skips unknown
JSON values lexically, so a normal observation call does not decode group or
target values. Only the explicit include_labels path invokes Dataset.targets.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

from rvwm.data import Dataset, sha256
from rvwm.coordinates import FixedStationFrame
from rvwm.geometry import quaternion_rotation, rays
from rvwm.group_observation import _ray_intervals
from rvwm.types import Camera, RadarConfig
from rvwm.track_detection_store import TrackDetectionStore, FrameNotExported
from rvwm.phase_selection import configured_phases, validate_phases, inventory_phases, phase_label
from rvwm.detector_contract import detector_contract
from rvwm.vision import frontend_metadata, validate_visual_configuration
from .visual import SCHEMA, validate_visual_features


def _value_end(text, start):
    """Locate a JSON value without decoding its contents."""
    string = False; escaped = False; stack=[]; i=start
    while i < len(text):
        c=text[i]
        if string:
            if escaped: escaped=False
            elif c=='\\': escaped=True
            elif c=='"': string=False
        else:
            if c=='"': string=True
            elif c in '[{': stack.append(c)
            elif c in ']}':
                if not stack: return i
                stack.pop()
            elif c==',' and not stack: return i
        i+=1
    return i


def selected_json(text, fields, raw_fields=()):
    """Decode selected object members; other member values remain opaque text."""
    if isinstance(text, bytes): text=text.decode('utf-8')
    decoder=json.JSONDecoder(); i=0; fields=set(fields); raw_fields=set(raw_fields); result={}
    while i<len(text) and text[i].isspace(): i+=1
    if i==len(text) or text[i]!='{': raise ValueError('Expected sensor JSON object')
    i+=1
    while True:
        while i<len(text) and (text[i].isspace() or text[i]==','): i+=1
        if i>=len(text): raise ValueError('Truncated sensor JSON')
        if text[i]=='}': break
        key,end=decoder.raw_decode(text,i)
        if not isinstance(key,str): raise ValueError('JSON object key must be string')
        i=end
        while i<len(text) and text[i].isspace(): i+=1
        if text[i]!=':': raise ValueError('Missing JSON colon')
        i+=1
        while i<len(text) and text[i].isspace(): i+=1
        end=_value_end(text,i); raw=text[i:end].strip()
        if key in fields:
            if key in result: raise ValueError('Duplicate selected sensor field: '+key)
            result[key]=raw if key in raw_fields else json.loads(raw)
        i=end
    return result


def json_array_values(text):
    """Yield opaque JSON array item slices without decoding item contents."""
    if isinstance(text,bytes):text=text.decode('utf-8')
    i=0
    while i<len(text) and text[i].isspace():i+=1
    if i==len(text) or text[i]!='[':raise ValueError('Expected a JSON array')
    i+=1
    while True:
        while i<len(text) and (text[i].isspace() or text[i]==','):i+=1
        if i>=len(text):raise ValueError('Truncated JSON array')
        if text[i]==']':return
        end=_value_end(text,i);raw=text[i:end].strip()
        if not raw:raise ValueError('Empty JSON array item')
        yield raw
        i=end


def public_split_manifest(config):
    """Bind original split bytes while decoding only public membership fields.

    Asset, formation, true member count and other label summaries in the same
    frozen file remain opaque. The SHA is the original file's full SHA, not
    the SHA of this allowlisted view.
    """
    if not config.get('split_manifest'):raise ValueError('An explicit frozen split_manifest is required')
    path=Path(config['split_manifest']).expanduser().resolve();raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if config.get('split_manifest_sha256') not in (None,digest):raise ValueError('Canonical split manifest SHA256 changed')
    if config.get('visual_split_manifest') and sha256(config['visual_split_manifest'])!=digest:
        raise ValueError('Detector and model split manifests differ in exact SHA256')
    if config.get('split_policy','unified_episode_manifest_v1')!='unified_episode_manifest_v1':
        raise ValueError('Unknown active split policy')
    top=selected_json(raw,('schema','dataset_root','echo_subdir','splits','episodes','request'),('splits','episodes','request'))
    roles=selected_json(top['splits'],('train','val','test'),('train','val','test'))
    if set(roles)!={'train','val','test'}:raise ValueError('Explicit train/val/test membership required')
    splits={};seen=set()
    for role in ('train','val','test'):
        value=selected_json(roles[role],('episode_ids','episode_count','frame_count'))
        ids=value.get('episode_ids')
        if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('Unique episode identifiers required in each frozen role')
        if seen.intersection(ids):raise ValueError('Episode occurs in multiple frozen roles')
        if value.get('episode_count',len(ids))!=len(ids):raise ValueError('Frozen episode count is inconsistent')
        seen.update(ids);splits[role]=value
    episode_fields=('episode','frames_sha256','metadata_sha256','frame_count','capture_frame_count',
        'selected_frame_indices','selected_phases','phase','capture_status','capture_modalities',
        'depth_required','depth_saved','dataset_split_policy','training_sampling_policy')
    episodes=[selected_json(item,episode_fields) for item in json_array_values(top['episodes'])]
    indexed={e.get('episode'):e for e in episodes}
    if len(indexed)!=len(episodes) or set(indexed)!=seen:raise ValueError('Frozen episode inventory is missing or duplicated')
    request=selected_json(top.get('request','{}'),('selected_phases','phase'))
    result={k:v for k,v in top.items() if k not in ('splits','episodes','request')}
    result.update(splits=splits,episodes=episodes,request=request)
    return path,result,digest


class PublicTrackDetectionStore(TrackDetectionStore):
    """Same frozen export checks, without decoding split labels/future boxes.

    The legacy get() geometry/score checks are retained. Only the manifest
    reader and current-frame decoding boundary differ from its implementation.
    """
    def __init__(self,directory,*,config,split,identity):
        self.directory=Path(directory).expanduser().resolve()
        manifest_bytes=(self.directory/'manifest.json').read_bytes()
        self.manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
        _,partition,split_sha=public_split_manifest(config)
        self.phases=validate_phases(configured_phases(config))
        if self.phases!=inventory_phases(partition):raise ValueError('Export phases differ from frozen public inventory')
        expected=dict(schema='rvwm_track_rgb_detections_v1' if self.phases==('track',) else 'rvwm_selected_phase_rgb_detections_v2',
            phase=phase_label(self.phases),weights_sha256=identity['weights_sha256'],confidence=identity['confidence'],
            imgsz=None,split_manifest_sha256=split_sha,classes={'0':'drone'})
        expected.update(identity)
        if self.phases!=('track',):expected['selected_phases']=list(self.phases)
        manifest=selected_json(manifest_bytes,set(expected)|{'complete','dataset_root','detector_backend'})
        if manifest.get('complete') is not True:raise ValueError('Frozen RGB detection export is incomplete')
        if Path(manifest['dataset_root']).resolve()!=Path(config['dataset_root']).resolve():
            raise ValueError('Detection export dataset root mismatch')
        if identity.get('detector_backend')!='dinov3_fpn' or manifest.get('detector_backend')!='dinov3_fpn':
            raise ValueError('Only the frozen DINOv3 export is supported')
        required={'detector_backend','architecture','weights_sha256','confidence','imgsz','preprocessing',
            'adapter_sha256','loader_sha256','detector_split_manifest_sha256','run_manifest_sha256','training_config_sha256'}
        if not required<=set(identity):raise ValueError('Complete frozen detector identity is required')
        for key,value in expected.items():
            if manifest.get(key)!=value:raise ValueError('Frozen RGB export identity mismatch: '+key)
        self.split=split;self.backend='dinov3_fpn';self.detector_contract=dict(identity)
        self.manifest=manifest;self.confidence=identity['confidence']
        self.episodes=frozenset(partition['splits'][split]['episode_ids'])
        self._inventory={e['episode']:e for e in partition['episodes'] if e['episode'] in self.episodes}
        if set(self._inventory)!=self.episodes:raise ValueError('Frozen RGB export episode inventory mismatch')
        self._episode=None;self._rows={}

    def _load(self,episode):
        if episode not in self.episodes:raise ValueError('Episode outside frozen detection split')
        receipt=selected_json((self.directory/'receipts'/f'{episode}.json').read_bytes(),
            ('episode_id','split','prediction_sha256','source_frames_sha256','frames'))
        raw=(self.directory/self.split/f'{episode}.jsonl').read_bytes();source=self._inventory[episode]
        if (receipt['episode_id']!=episode or receipt['split']!=self.split
                or receipt['prediction_sha256']!=hashlib.sha256(raw).hexdigest()
                or receipt['source_frames_sha256']!=source['frames_sha256']):
            raise ValueError('Frozen RGB shard receipt mismatch')
        rows={}
        fields=('frame_idx','episode_id','split','phase','source_frames_sha256','rgb_sha256',
                'rgb_content_sha256','image_size_wh','detections')
        for line in raw.splitlines():
            if not line.strip():continue
            row=selected_json(line,fields,('detections',));index=row['frame_idx']
            if (type(index) is not int or index<0 or index in rows or row['episode_id']!=episode
                    or row['split']!=self.split or row['phase'] not in self.phases
                    or row['source_frames_sha256']!=source['frames_sha256']):
                raise ValueError('Invalid frozen RGB shard header')
            rows[index]=row
        if len(rows)!=source['frame_count'] or len(rows)!=receipt['frames']:
            raise ValueError('Frozen RGB shard frame count mismatch')
        if 'selected_frame_indices' in source and sorted(rows)!=source['selected_frame_indices']:
            raise ValueError('Frozen RGB shard selected frames mismatch')
        self._episode=episode;self._rows=rows

    def get(self,episode_id,frame_idx,rgb_sha256,*,rgb_content_sha256=None,source_frames_sha256=None,image_size_wh=None):
        if episode_id!=self._episode:self._load(episode_id)
        if frame_idx not in self._rows:raise FrameNotExported((episode_id,frame_idx))
        raw_row=self._rows[frame_idx];row=dict(raw_row)
        row['detections']=[selected_json(item,('bbox_xyxy','score','class_id','detection_id'))
                           for item in json_array_values(raw_row['detections'])]
        self._rows[frame_idx]=row
        try:return super().get(episode_id,frame_idx,rgb_sha256,rgb_content_sha256=rgb_content_sha256,
            source_frames_sha256=source_frames_sha256,image_size_wh=image_size_wh)
        finally:self._rows[frame_idx]=raw_row


class PublicExportedRGBFrontend:
    """Replay the same trained RGB detector using a public-only frozen reader."""
    def __init__(self,config,*,split=None,keypoint_checkpoint=None,device='cpu'):
        validate_visual_configuration(config,keypoint_checkpoint)
        identity=detector_contract(config)
        if sha256(identity['weights'])!=identity['weights_sha256']:raise ValueError('Frozen DINO weights changed')
        self.config=dict(config);self.device=str(device);self.backend='dinov3_fpn'
        self.detector_contract=identity
        self.track_store=PublicTrackDetectionStore(config['track_detection_dir'],config=config,
            split=split or config.get('track_detection_split','train'),identity=identity)
        self.metadata=frontend_metadata(config,identity)
        from rvwm import visual_refinement as legacy_frontend_module
        legacy=Path(legacy_frontend_module.__file__).resolve()
        self.metadata.update(track_detection_dir=str(self.track_store.directory),track_detection_split=self.track_store.split,
            track_detection_manifest_sha256=self.track_store.manifest_sha256,
            detector_execution='strict public-only frozen RGB replay; only requested-frame detections decoded',
            replay_adapter_sha256=sha256(__file__),runtime_reader_sha256=sha256(__file__),
            legacy_export_frontend_sha256=sha256(legacy),split_label_fields_decoded=False,
            future_detection_values_decoded=False)
        self.keypoint=None;self.cache_hits=0;self.cache_misses=0

    def close(self):pass


# Preserve the test/injection point while replacing the legacy split reader.
ExportedRGBFrontend=PublicExportedRGBFrontend


def _visual_scale(boxes, scores, k):
    w,h=(boxes[:,2:]-boxes[:,:2]).T
    return np.column_stack((w/k[0,0],h/k[1,1],np.sqrt(w*h/(k[0,0]*k[1,1])),
        np.log(w/k[0,0]),np.log(h/k[1,1]),np.log(w/h),scores))


class VisualSensorReader:
    def __init__(self, config, split, frontend=None, *, load_detector=True):
        if split not in ('train','val','test'): raise ValueError('Unknown frozen split')
        self.config=dict(config);self.root=Path(config['dataset_root']).resolve();self.split=split
        _,manifest,digest=public_split_manifest(config)
        self.split_digest=digest
        self.ids=frozenset(manifest['splits'][split]['episode_ids'])
        self.inventory={x['episode']:x for x in manifest['episodes'] if x['episode'] in self.ids}
        self.frontend=(frontend or ExportedRGBFrontend(config,split=split,device='cpu')) if load_detector else None
        self._loaded={}
        self.metadata=dict(self.frontend.metadata if self.frontend is not None else {}, visual_only=True, radar_features_read=False,
            iq_read=False, target_fields_decoded=False, visual_schema=SCHEMA)

    def close(self):
        if self.frontend is not None:self.frontend.close()

    def episode(self, episode):
        if episode not in self.ids or re.fullmatch(r'episode_[0-9]{6,}',episode) is None:
            raise ValueError('Episode outside requested frozen split')
        if episode in self._loaded:return self._loaded[episode]
        d=self.root/'data'/episode;e=d/self.config.get('echo_subdir','echoes')
        metadata_raw=(d/'metadata.json').read_bytes()
        dm=selected_json(metadata_raw,('status','resolution','reported_camera_fov_deg','fov_deg','frame_count','camera_offset_body_m'))
        em=selected_json((e/'metadata.json').read_bytes(),('status','frame_count','frames_sha256','provenance'),('provenance',))
        provenance=selected_json(em['provenance'],('source_capture_frames_sha256','source_capture_metadata_sha256','source_sha256','metadata_sha256','config'),('config',))
        public=selected_json(provenance['config'],('carrier_hz','bandwidth_hz','pulse_interval_s','frame_interval_s','iq_shape','spacing_y_m','spacing_z_m'))
        capture=(d/'frames.jsonl').read_bytes();echo=(e/'frames.jsonl').read_bytes()
        capture_sha=hashlib.sha256(capture).hexdigest();metadata_sha=hashlib.sha256(metadata_raw).hexdigest()
        if (capture_sha!=self.inventory[episode]['frames_sha256']
                or capture_sha!=provenance.get('source_capture_frames_sha256',provenance.get('source_sha256'))
                or metadata_sha!=provenance.get('source_capture_metadata_sha256',provenance.get('metadata_sha256'))
                or hashlib.sha256(echo).hexdigest()!=em['frames_sha256']):
            raise ValueError('Frozen sensor metadata/trajectory source changed')
        if self.inventory[episode].get('metadata_sha256',metadata_sha)!=metadata_sha:
            raise ValueError('Frozen public split metadata SHA256 changed')
        if dm['status']!='complete' or em['status']!='complete': raise ValueError('Incomplete sensor source')
        capture_fields=('frame_idx','sim_timestamp_ns','phase','observer','rgb_path','camera')
        echo_fields=('frame_idx','physical_time_s','station_position_ned_m','station_heading_deg',
                     'source_sim_timestamp_ns','iq_path','iq_sha256')
        # Keep pose values opaque until a caller requests that exact frame.
        # Reading/hashing a source manifest must not decode future observations.
        dr=[selected_json(line,capture_fields,('camera','observer')) for line in capture.splitlines() if line.strip()]
        er=[selected_json(line,echo_fields) for line in echo.splitlines() if line.strip()]
        if len(dr)!=len(er) or len(dr)!=dm['frame_count'] or len(er)!=em['frame_count']:
            raise ValueError('Sensor frame count mismatch')
        station=None;basis=None
        for i,(a,b) in enumerate(zip(dr,er)):
            if a['frame_idx']!=i or b['frame_idx']!=i or a['sim_timestamp_ns']!=b['source_sim_timestamp_ns']:
                raise ValueError('Sensor synchronization mismatch')
            heading=math.radians(float(b['station_heading_deg']));c,s=math.cos(heading),math.sin(heading)
            current=np.array([[c,-s,0],[s,c,0],[0,0,-1.]])
            if station is None:station=np.asarray(b['station_position_ned_m'],float);basis=current
            if not np.allclose(b['station_position_ned_m'],station,atol=1e-6,rtol=0) or not np.allclose(current,basis,atol=1e-8,rtol=0):
                raise ValueError('Single fixed station required per episode')
        frame=FixedStationFrame(station,basis,episode)
        result=dict(root=d,metadata=dm,public=public,capture=dr,echo=er,frame=frame,
                    capture_sha=capture_sha,metadata_sha=metadata_sha)
        self._loaded[episode]=result;return result

    def frame_indices(self,episode,count=0):
        record=self.episode(episode);phases=tuple(self.config.get('training_phases',['approach']))
        indices=[r['frame_idx'] for r in record['capture'] if r['phase'] in phases]
        if count:indices=[indices[i] for i in np.linspace(0,len(indices)-1,min(count,len(indices)),dtype=int)]
        return indices

    def sensor_frame(self,episode,index):
        """Public calibration/paths only; never opens RGB pixels, IQ or labels.

        ``iq_path``/``iq_sha256`` are frozen file provenance, not IQ-derived
        measurements. A volume builder may explicitly open that path later.
        """
        r=self.episode(episode)
        if type(index) is not int or not 0<=index<len(r['capture']):raise IndexError(index)
        dm=r['metadata'];row=r['capture'][index];echo=r['echo'][index]
        width,height=map(int,dm['resolution'])
        reported=dm.get('reported_camera_fov_deg');fov=float(reported if reported is not None else dm['fov_deg'])
        focal=width/(2*math.tan(math.radians(fov)/2));k=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
        cam=selected_json(row['camera'],('camera_position_ned','camera_orientation_wxyz'))
        body=selected_json(row.get('observer','{}'),('position_ned','orientation_wxyz'))
        if not isinstance(body,dict) or 'position_ned' not in body:raise ValueError('Measured observer navigation required')
        camera=Camera(k,np.asarray(cam['camera_position_ned'],float),quaternion_rotation(cam['camera_orientation_wxyz']),width,height)
        public=dict(r['public']);half_wave=299792458./(2*float(public['carrier_hz']))
        radar=RadarConfig(float(public['carrier_hz']),float(public['bandwidth_hz']),float(public['pulse_interval_s']),
            tuple(public['iq_shape']),float(public.get('spacing_y_m',half_wave)),float(public.get('spacing_z_m',half_wave)),
            float(public['frame_interval_s']))
        rgb=(r['root']/row['rgb_path']).resolve();echo_root=r['root']/self.config.get('echo_subdir','echoes')
        iq=(echo_root/echo['iq_path']).resolve()
        if not rgb.is_relative_to(r['root']) or not iq.is_relative_to(echo_root):raise ValueError('Sensor file path escapes episode')
        return dict(episode_id=episode,frame_idx=index,phase=row['phase'],physical_time_s=echo['physical_time_s'],
            camera=camera,observer_position_ned=np.asarray(body['position_ned'],float),radar=radar,
            radar_config=public,station_position_ned=np.asarray(echo['station_position_ned_m'],float),
            station_heading_deg=float(echo['station_heading_deg']),station_frame=r['frame'],
            iq_path=iq,iq_sha256=echo['iq_sha256'],rgb_path=rgb,
            capture_hash=r['capture_sha'],metadata_hash=r['metadata_sha'],
            source_trajectory_sha256=r['capture_sha'],source_metadata_sha256=r['metadata_sha'])

    def observation(self,episode,index):
        if self.frontend is None:raise RuntimeError('Detection loader was explicitly disabled')
        r=self.episode(episode);dm=r['metadata'];row=r['capture'][index];echo=r['echo'][index]
        if row['phase'] not in tuple(self.config.get('inference_phases',['approach'])):
            raise ValueError('Frame outside requested current-frame phase')
        width,height=map(int,dm['resolution'])
        reported=dm.get('reported_camera_fov_deg');fov=float(reported if reported is not None else dm['fov_deg'])
        focal=width/(2*math.tan(math.radians(fov)/2));k=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
        camera=selected_json(row['camera'],('camera_position_ned','camera_orientation_wxyz'))
        body=selected_json(row.get('observer','{}'),('position_ned','orientation_wxyz'))
        if not isinstance(body,dict) or 'position_ned' not in body: raise ValueError('Measured observer navigation required')
        source=Camera(k,np.asarray(camera['camera_position_ned'],float),quaternion_rotation(camera['camera_orientation_wxyz']),width,height)
        rgb=(r['root']/row['rgb_path']).resolve()
        if not rgb.is_relative_to(r['root']): raise ValueError('RGB path escapes episode')
        rgb_sha=sha256(rgb)
        boxes,scores=self.frontend.track_store.get(episode,index,rgb_sha,source_frames_sha256=r['capture_sha'],image_size_wh=(width,height))
        uv=(boxes[:,:2]+boxes[:,2:])/2
        directions=r['frame'].to_local_vectors(rays(uv,source))
        origins=np.repeat(r['frame'].to_local_positions(source.position)[None],len(boxes),axis=0)
        observer=r['frame'].to_local_positions(np.asarray(body['position_ned'],float))
        public=r['public'];radius=299792458.*int(public['iq_shape'][1])/(2*float(public['bandwidth_hz']))
        bounds,valid=_ray_intervals(origins,directions,np.ones(len(boxes),bool),radius)
        features=dict(schema=SCHEMA,depth_schema='visual_only_bbox_scale_v1',ray_origins=origins,
            ray_directions=directions,ray_valid=valid,ray_depth_bounds=bounds,observer_position=observer,
            range_scale_m=float(radius),bbox_xyxy=boxes.copy(),detection_scores=scores.copy(),
            visual_scale=_visual_scale(boxes,scores,k),camera_K=k,camera_image_size_wh=np.array([width,height]),
            metadata=dict(episode_id=episode,frame_idx=index,coordinate_frame='base_local',
                sensor_only=True,radar_features_read=False,iq_read=False,target_fields_decoded=False,
                split_label_fields_decoded=False,future_detection_values_decoded=False,
                runtime_reader_sha256=sha256(__file__),
                camera_origin='reported optical centre; installation offset already included',
                hashes=dict(rgb=rgb_sha,source_trajectory=r['capture_sha'],source_metadata=r['metadata_sha'])))
        validate_visual_features(features)
        receipt=dict(episode_id=episode,frame_idx=index,split=self.split,phase=row['phase'],
            feature_status='prepared',detected_count=len(boxes),configured_window=1,
            hashes=deepcopy(features['metadata']['hashes']),frame_interval_s=float(public['frame_interval_s']),
            visual_schema=SCHEMA,radar_features_read=False,iq_read=False)
        return dict(features=features,episode_id=episode,frame_idx=index),receipt,r['frame']


def load_samples(config, ids, split, count=0, include_labels=True):
    if type(include_labels) is not bool: raise ValueError('Explicit label-access switch required')
    reader=VisualSensorReader(config,split);samples=[];rows=[]
    dataset=Dataset(config['dataset_root'],echo_subdir=config.get('echo_subdir','echoes'),verify_sources=True) if include_labels else None
    try:
        for episode in ids:
            for index in reader.frame_indices(episode,count):
                sample,row,frame=reader.observation(episode,index)
                if include_labels:
                    labels=dataset.targets(episode,index)
                    sample.update(positions=frame.to_local_positions(labels.positions).astype(np.float32),formation_index=None)
                    row['label_count']=len(labels.positions)
                samples.append(sample);rows.append(row)
        return samples,rows,dict(reader.metadata)
    finally:reader.close()


def load_observation_features(config,episode,frame_idx,split):
    if type(frame_idx) is not int or frame_idx<0: raise ValueError('Nonnegative current frame index required')
    reader=VisualSensorReader(config,split)
    try:return reader.observation(episode,frame_idx)[0]['features']
    finally:reader.close()


def load_sensor_frame(config,episode,frame_idx,split):
    """One-shot public sensor descriptor; use a reader for many frames."""
    reader=VisualSensorReader(config,split,load_detector=False)
    try:return reader.sensor_frame(episode,frame_idx)
    finally:reader.close()
