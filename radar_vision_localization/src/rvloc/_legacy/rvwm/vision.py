"""Frozen DINOv3 RGB detections and bbox-centre geometry, without ROI heads."""
from __future__ import annotations
import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import numpy as np

from .types import Detection, Observations
from .track_detection_store import FrameNotExported, TrackDetectionStore
from .phase_selection import configured_phases

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DETECTION_CACHE_SCHEMA = 'rvwm-dinov3-fpn-rgb-detection-v1'


def detection_cache_schema(backend='dinov3_fpn'):
    if backend != 'dinov3_fpn':
        raise ValueError('Only the frozen DINOv3 detector is supported')
    return DETECTION_CACHE_SCHEMA


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_visual_configuration(config, keypoint_checkpoint=None):
    if config.get('detector_backend', 'dinov3_fpn') != 'dinov3_fpn':
        raise ValueError('Only the frozen DINOv3 detector is supported')
    if config.get('visual_point_mode', 'bbox_center') != 'bbox_center':
        raise ValueError('Visual points must be detection bbox centres')
    if keypoint_checkpoint is not None or config.get('keypoint_checkpoint') is not None:
        raise ValueError('ROI keypoint checkpoints are not part of bbox-centre localization')
    if not config.get('visual_geometry_only', True):
        raise ValueError('Localization uses visual ray geometry only')
    pixel_std = float(config.get('pixel_std', 3.))
    if not np.isfinite(pixel_std) or pixel_std <= 0:
        raise ValueError('pixel_std must be finite and positive')


def frontend_metadata(config, identity):
    split = Path(config.get('visual_split_manifest', identity['detector_split_manifest']))
    trained = json.loads(Path(identity['run_manifest']).read_text())
    return dict(detector='dinov3_vits16_fpn_fasterrcnn_uav_finetuned',
        detector_backend='dinov3_fpn', architecture=identity['architecture'],
        weights=identity['weights'], weights_sha256=identity['weights_sha256'],
        confidence=identity['confidence'], preprocessing=identity['preprocessing'],
        detector_contract=dict(identity), frozen=True,
        online_inputs=['RGB pixels'], oracle_inputs_used=False,
        visual_split_manifest=str(split.resolve()), visual_split_manifest_sha256=_file_sha(split),
        detector_training_episodes=trained['training_episode_ids'],
        detector_training_split_manifest_sha256=identity['detector_split_manifest_sha256'],
        keypoint_mode='bbox_center', appearance='not_used_by_localizer',
        covariance_status='configured_pixel_noise_not_empirically_calibrated',
        adapter_sha256=_file_sha(__file__), worker_sha256=identity['loader_sha256'],
        cache_schema=DETECTION_CACHE_SCHEMA)


class _DINOv3Worker:
    """Frozen native DINOv3/FPN execution without changing the RVWM venv."""
    worker_name='DINOv3'

    def __init__(self,config,path,device,contract):
        cache=PROJECT/'cache'/'dinov3_worker';cache.mkdir(parents=True,exist_ok=True)
        python=Path(config.get('dinov3_python',WORKSPACE/'uav_detection/.venv_deimv2/bin/python'))
        if not python.is_file():raise FileNotFoundError('DINOv3 Python environment missing: '+str(python))
        self.cache=cache
        self.log=tempfile.NamedTemporaryFile(prefix='worker_',suffix='.log',dir=cache,delete=False)
        environment=os.environ.copy()
        environment['PYTHONPATH']=os.pathsep.join([str(PROJECT),str(WORKSPACE),environment.get('PYTHONPATH','')])
        environment['OMP_NUM_THREADS']=str(config.get('detector_threads',1));environment['OPENBLAS_NUM_THREADS']='1'
        command=[str(python),'-m','rvwm.dinov3_worker','--weights',str(path),'--device',str(device),
                 '--confidence',str(contract['confidence']),'--threads',str(config.get('detector_threads',1))]
        self.process=subprocess.Popen(command,cwd=str(WORKSPACE),env=environment,stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        atexit.register(self.close)
        try:
            reply=self._read(float(config.get('detector_timeout_s',180)))
            self.metadata=reply['metadata']
            for name in ('architecture','weights_sha256','confidence'):
                if self.metadata.get(name)!=contract[name]:
                    raise ValueError('DINOv3 worker/checkpoint contract mismatch: '+name)
            native=self.metadata['preprocessing']
            for name in ('min_size','max_size','size_divisible','image_mean','image_std'):
                if native.get(name)!=contract['preprocessing'][name]:
                    raise ValueError('DINOv3 native preprocessing mismatch: '+name)
            self.metadata.update(worker_python=str(python),
                worker_protocol='persistent RGB-only cache NPY and JSON-lines; no labels or tracking',
                worker_log=self.log.name,native_preprocessing=dict(native),preprocessing=contract['preprocessing'])
        except Exception:
            self.close()
            raise

    def _read(self,timeout=180.):
        name=self.worker_name
        ready,_,_=select.select([self.process.stdout],[],[],timeout)
        if not ready:
            self.close();raise TimeoutError(name+' worker response timeout; inspect '+self.log.name)
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(name+' worker exited; inspect '+self.log.name)
        reply=json.loads(line)
        if not reply.get('ok'):raise RuntimeError(name+' worker: '+reply.get('error','unknown error'))
        return reply

    def detect(self,rgb):
        fd,path=tempfile.mkstemp(prefix='rgb_',suffix='.npy',dir=self.cache);os.close(fd)
        try:
            np.save(path,rgb,allow_pickle=False)
            self.process.stdin.write(json.dumps(dict(command='detect',rgb_npy=path))+'\n');self.process.stdin.flush()
            result=self._read()
            return np.asarray(result['boxes'],float).reshape(-1,4),np.asarray(result['scores'],float)
        finally:Path(path).unlink(missing_ok=True)

    def close(self):
        process=getattr(self,'process',None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"command":"close"}\n');process.stdin.flush();process.wait(timeout=2)
            except Exception:process.terminate()
        log=getattr(self,'log',None)
        if log is not None and not log.closed:log.close()


class RGBFrontend:
    """Frozen RGB-only DINO detections; visual locations are bbox midpoints."""
    def __init__(self, config, weights=None, device='cpu', keypoint_checkpoint=None):
        from .detector_contract import detector_contract
        validate_visual_configuration(config, keypoint_checkpoint)
        self.config = dict(config)
        self.device = 'cuda:' + str(device) if str(device).isdigit() else str(device)
        self.backend = 'dinov3_fpn'
        self.detector_contract = detector_contract(self.config, weights)
        path = Path(self.detector_contract['weights'])
        self.config['detector_confidence'] = self.detector_contract['confidence']
        self.metadata = frontend_metadata(self.config, self.detector_contract)
        self.keypoint = None
        self.track_store = None
        directory = self.config.get('track_detection_dir')
        if directory is not None:
            self.track_store = TrackDetectionStore(directory,
                dataset_root=config['dataset_root'], split_manifest=self.metadata['visual_split_manifest'],
                weights_sha256=self.detector_contract['weights_sha256'],
                confidence=self.detector_contract['confidence'], imgsz=None,
                split=config.get('track_detection_split', 'train'),
                phases=configured_phases(self.config), backend=self.backend,
                detector_contract=self.detector_contract)
            self.metadata.update(track_detection_dir=str(self.track_store.directory),
                track_detection_split=self.track_store.split,
                track_detection_manifest_sha256=self.track_store.manifest_sha256)
        self.detector = _DINOv3Worker(self.config, path, self.device, self.detector_contract)
        self.metadata.update(worker_python=self.detector.metadata.get('worker_python'),
            worker_protocol=self.detector.metadata.get('worker_protocol'),
            worker_log=self.detector.metadata.get('worker_log'))
        self._receipt = dict(schema=DETECTION_CACHE_SCHEMA, backend=self.backend,
            weights_sha256=self.detector_contract['weights_sha256'],
            confidence=self.detector_contract['confidence'], imgsz=None, device=self.device,
            adapter_sha256=self.metadata['adapter_sha256'], worker_sha256=self.metadata['worker_sha256'],
            detector_contract=dict(self.detector_contract))
        self.cache = PROJECT / 'cache' / 'dinov3_rgb_detections'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0

    def _raw_detect(self, observations):
        return self.detector.detect(observations.rgb)

    def detect(self, observations):
        if not isinstance(observations, Observations):
            raise TypeError('RGBFrontend accepts Observations only')
        rgb = observations.rgb
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB uint8 HWC')
        exported = None
        store = self.track_store
        if store is not None and observations.episode_id in store.episodes:
            try:
                exported = store.get(observations.episode_id, observations.frame_idx,
                    observations.source_hashes.get('rgb'),
                    rgb_content_sha256=hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
                    source_frames_sha256=observations.source_hashes.get('source_trajectory'),
                    image_size_wh=(rgb.shape[1], rgb.shape[0]))
            except FrameNotExported:
                pass
        if exported is not None:
            boxes, scores = exported
        elif hasattr(self, '_receipt') and self.config.get('detector_cache', True):
            receipt = dict(self._receipt,
                rgb_content_sha256=hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
                rgb_shape=list(rgb.shape), source_rgb_file_sha256=observations.source_hashes.get('rgb'))
            key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
            cached = self.cache / (key + '.json')
            if cached.exists():
                stored = json.loads(cached.read_text())
                if stored.get('receipt') != receipt:
                    raise ValueError('RGB detection cache provenance mismatch')
                boxes = np.asarray(stored['boxes'], float).reshape(-1, 4)
                scores = np.asarray(stored['scores'], float)
                self.cache_hits += 1
            else:
                boxes, scores = self._raw_detect(observations)
                self.cache_misses += 1
                fd, temporary = tempfile.mkstemp(prefix=key, suffix='.json', dir=self.cache)
                os.close(fd)
                try:
                    Path(temporary).write_text(json.dumps(dict(receipt=receipt,
                        boxes=boxes.tolist(), scores=scores.tolist()), allow_nan=False) + '\n')
                    Path(temporary).replace(cached)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        else:
            boxes, scores = self._raw_detect(observations)
        height, width = rgb.shape[:2]
        records = []
        for box, score in zip(boxes, scores):
            box = np.clip(box, [0, 0, 0, 0], [width, height, width, height]).astype(float)
            if not np.isfinite(box).all() or not np.isfinite(score) or np.any(box[2:] <= box[:2]):
                continue
            records.append(Detection((box[:2] + box[2:]) / 2,
                np.eye(2) * float(self.config.get('pixel_std', 3.)) ** 2,
                box, float(score), np.zeros(128, dtype=np.float32)))
        return records

    def close(self):
        detector = getattr(self, 'detector', None)
        if detector is not None:
            detector.close()
