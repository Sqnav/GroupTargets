"""Matched ROI observations with a genuinely radar-free visual input path.

The visual arm reads public RGB/detection/navigation metadata and the frozen
ROI payload only. Radar-derived caches, IQ payloads and spectrum builders are
never called on that path. Radar arms authenticate existing BW study caches;
cache misses fail instead of rebuilding. Labels are separate opt-in arguments.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from bandwidth_study import data as bw
from d_evidence_study import data as evidence
from d_evidence_speed import data as resident
from depth_localization.features import DETECTION_FIELDS, feature_digest, validate_depth_features
from precision_study.visual import (ALLOWED_FIELDS as VISUAL_FIELDS, SCHEMA as VISUAL_SCHEMA,
                                    drop_visual_detections, validate_visual_features)
from precision_study.visual_data import VisualSensorReader
from rvwm.data import Dataset, sha256
from rvwm.detector_contract import detector_contract
from rvwm.group_model import match_ray_labels

ROOT = Path(__file__).resolve().parents[1]
BW_ROOT = ROOT/'results/bandwidth_double_20261124'
ROI_ROOT = ROOT/'results/d_evidence_20261124/roi'
ARMS = ('visual', '32mhz', '64mhz')
PackedBatch = resident.PackedBatch
MatchPlan = resident.MatchPlan
MEMBER_FIELDS = (*DETECTION_FIELDS, 'detector_roi_features')
VISUAL_FRAME_FIELDS = ('observer_position', 'range_scale_m', 'camera_K', 'camera_image_size_wh')
COMPARABLE_FIELDS = (*DETECTION_FIELDS, 'observer_position', 'camera_K', 'camera_image_size_wh')


def read(path):
    return json.loads(Path(path).read_text())


def study_arm(config):
    arm = config.get('study_arm')
    if arm not in ARMS:
        raise ValueError('Explicit study_arm visual/32mhz/64mhz required')
    return arm


def _roi_checked(features):
    value = resident._numpy(features.get('detector_roi_features'))
    if value.shape != (len(features['ray_directions']), 1024) or not np.isfinite(value).all():
        raise ValueError('Finite frozen 1024-dimensional ROI vectors must align with detections')
    return value


def _visual_checked(features):
    if set(features) != VISUAL_FIELDS | {'detector_roi_features'}:
        raise ValueError('Visual observations must contain only public visual/navigation fields and ROI; radar fields rejected')
    base = {k: resident._numpy(v) if isinstance(v, torch.Tensor) else v
            for k, v in features.items() if k != 'detector_roi_features'}
    validate_visual_features(base)
    _roi_checked(features)
    return base


def visual_feature_digest(features):
    base = _visual_checked(features)
    h = hashlib.sha256(base['schema'].encode())
    for key in sorted(VISUAL_FIELDS - {'schema', 'depth_schema', 'metadata', 'range_scale_m'}):
        a = np.ascontiguousarray(resident._numpy(base[key]))
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(repr(float(base['range_scale_m'])).encode())
    h.update(json.dumps(base['metadata'], sort_keys=True, allow_nan=False).encode())
    h.update(np.ascontiguousarray(_roi_checked(features)).tobytes())
    return h.hexdigest()


class ROIAuthenticator:
    """Authenticate RGB-only ROI payloads without reopening their radar parent.

    Old sensor_cache/feature_digest receipt fields stay opaque provenance. The
    actual reused ROI depends on RGB, observed boxes, detector/native backend,
    image size and immutable capture/calibration identity, all rechecked here.
    """
    def __init__(self, config):
        self.config = dict(config)
        self.cache = Path(config.get('d_evidence_roi_cache', ROI_ROOT/'cache')).resolve()
        self.root = self.cache.parent
        path = self.root/'export_receipt.json'
        self.export_sha = sha256(path)
        expected = config.get('study_roi_export_receipt_sha256')
        if expected is not None and expected != self.export_sha:
            raise ValueError('Frozen ROI export receipt changed')
        self.export = read(path)
        self.sources = evidence.source_hashes()
        if (self.export.get('status') != 'complete' or self.export.get('spec') != evidence.ROI_SPEC
                or self.export.get('source_sha256') != self.sources
                or self.export.get('cache') != str(self.cache)
                or set(self.export.get('requested_splits', [])) != {'train', 'val'}
                or self.export.get('max_episodes') != 0 or self.export.get('frames_per_episode') != 0
                or self.export.get('labels_read') is not False
                or self.export.get('trainable_tokens_cached') is not False):
            raise ValueError('Expected complete, immutable, label-free frozen ROI export')
        inventory = Path(self.export['inventory']['path']).resolve()
        if not inventory.is_relative_to(self.root) or sha256(inventory) != self.export['inventory']['sha256']:
            raise ValueError('ROI export inventory changed')
        rows = read(inventory)['frames']
        self.rows = {(r['split'], r['episode_id'], int(r['frame_idx'])):r for r in rows}
        if len(rows) != len(self.rows) or len(rows) != self.export['frames']:
            raise ValueError('ROI export inventory is missing or duplicated')
        self.contract = detector_contract(config)
        self.split_sha = sha256(config['split_manifest'])
        if config.get('split_manifest_sha256', self.split_sha) != self.split_sha:
            raise ValueError('Current split manifest changed')

    def load(self, features, row, rgb_path):
        key = row['split'], row['episode_id'], int(row['frame_idx'])
        if key not in self.rows:
            raise ValueError('Current frame is outside frozen ROI inventory')
        item = self.rows[key]
        payload = self.cache/key[1]/f'frame_{key[2]:05d}.npz'
        receipt_path = payload.with_suffix('.json')
        if (item['path'] != str(payload) or item['receipt_path'] != str(receipt_path)
                or sha256(receipt_path) != item['receipt_sha256'] or sha256(payload) != item['sha256']):
            raise ValueError('Frozen ROI payload or per-frame receipt changed')
        receipt = read(receipt_path); identity = receipt.get('identity', {})
        expected = dict(spec=evidence.ROI_SPEC, source_sha256=self.sources,
            detector_contract=self.contract, episode_id=key[1], frame_idx=key[2],
            rgb_path=str(Path(rgb_path).resolve()), rgb_sha256=row['hashes']['rgb'],
            rgb_stat=bw._stamp(rgb_path), image_size_wh=np.asarray(features['camera_image_size_wh']).tolist(),
            bbox_xyxy=np.asarray(features['bbox_xyxy']).tolist(),
            detection_scores=np.asarray(features['detection_scores']).tolist(),
            bbox_order='exact existing frozen detector order; no GT matching or sorting',
            split_manifest_sha256=self.split_sha)
        if (any(identity.get(k) != v for k, v in expected.items())
                or receipt.get('payload_sha256') != item['sha256']
                or receipt.get('eval_mode') is not True or receipt.get('inference_mode') is not True
                or receipt.get('labels_read') is not False or receipt.get('no_placeholder_features') is not True
                or not receipt.get('backend_source_sha256')):
            raise ValueError('ROI RGB, detector, boxes, calibration lineage or inference identity differs')
        for source, expected_sha in receipt['backend_source_sha256'].items():
            if evidence.verified_file_sha256(source) != expected_sha:
                raise ValueError('Frozen native ROI backend changed')
        with np.load(payload, allow_pickle=False) as stored:
            if set(stored.files) != {'features', 'boxes', 'scores'}:
                raise ValueError('ROI payload must contain only feature vectors, observed boxes and scores')
            value = stored['features']
            if (value.dtype != np.float32 or value.shape != (len(features['bbox_xyxy']), 1024)
                    or not np.isfinite(value).all()
                    or not np.array_equal(stored['boxes'], features['bbox_xyxy'])
                    or not np.array_equal(stored['scores'], features['detection_scores'])
                    or len(value) and np.any(np.linalg.norm(value, axis=1) == 0)):
                raise ValueError('ROI vectors are not aligned with current frozen observations')
        return value, dict(path=str(payload), sha256=item['sha256'], receipt_path=str(receipt_path),
            receipt_sha256=item['receipt_sha256'], export_receipt_sha256=self.export_sha,
            dimension=1024, radar_parent_receipt_fields_reopened=False,
            camera_geometry_binding='immutable split and source-capture SHA; current optical-centre calibration')


def sensor_config(config):
    """Separate BW recipe selection; never call the old 32MHz-only adapter."""
    arm = study_arm(config)
    if arm == 'visual':
        raise ValueError('Visual arm has no radar sensor configuration')
    parent_arm = 'baseline' if arm == '32mhz' else 'bandwidth2'
    path = Path(config.get('study_sensor_config_path', BW_ROOT/'configs'/(parent_arm+'.json')))
    base = bw.arm_config(read(path))
    if base['bandwidth_arm'] != parent_arm or base['bandwidth_hz'] != (32e6 if arm == '32mhz' else 64e6):
        raise ValueError('Requested study arm and registered BW recipe disagree')
    for key in ('dataset_root', 'split_manifest', 'split_manifest_sha256', 'visual_split_manifest',
                'track_detection_dir', 'training_phases', 'inference_phases'):
        if key in config and config[key] != base.get(key):
            raise ValueError('Study changed common visual/capture recipe: '+key)
    return base


class RadarCache:
    """Existing BW caches only: no rebuild, no IQ payload decoding."""
    def __init__(self, config, split):
        if split not in ('train', 'val'):
            raise ValueError('Final test observations are disabled')
        self.config = sensor_config(config)
        self.reader = bw.Reader(self.config, split)
        parent_arm = self.config['bandwidth_arm']
        preparation = Path(config.get('study_sensor_preparation', BW_ROOT/'prepare'/parent_arm))
        receipt = bw.validate_preparation(self.config, preparation, verify_cache=False)
        if (receipt.get('max_episodes') or receipt.get('frames_per_episode')
                or set(receipt['requested_splits']) != {'train', 'val'}):
            raise ValueError('Complete registered BW train/validation cache is required')
        rows = read(receipt['inventories'][split]['path'])['frames']
        self.rows = {(r['episode_id'], int(r['frame_idx'])):r for r in rows}
        if len(rows) != len(self.rows):
            raise ValueError('Duplicate existing BW cache frames')
        self.metadata = dict(self.reader.metadata, cache_rebuild_allowed=False,
            registered_preparation_sha256=sha256(preparation/'preparation_receipt.json'))

    def close(self):
        self.reader.close()

    def frame(self, episode, index):
        key = episode, index
        if key not in self.rows:
            raise ValueError('Current frame lacks a registered existing BW cache')
        row = self.rows[key]; path = Path(row['cache']['path']).resolve()
        identity = self.reader.episode(episode)
        sensor = self.reader.measured.sensor_frame(episode, index)
        if (not path.is_relative_to(Path(self.config['bandwidth_feature_cache']))
                or not path.is_file() or sha256(path) != row['cache']['sha256']
                or bw._stamp(sensor['iq_path']) != row['iq_stat']
                or sensor['iq_sha256'] != row['hashes']['iq']
                or identity['echo_metadata_sha256'] != row['echo_metadata_sha256']
                or identity['echo_frames_sha256'] != row['echo_frames_sha256']):
            raise ValueError('BW source/cache changed; rebuild is forbidden')
        stored = torch.load(path, map_location='cpu', weights_only=False)
        features, source = stored['features'], stored['receipt']
        if set(features) - evidence.SENSOR_FIELDS:
            raise ValueError('Unknown/non-sensor fields in BW cached observation')
        validate_depth_features(features)
        if (bw._digest(source) != path.stem or source['source_sha256'] != bw.source_hashes()
                or source['frame_idx'] != index or source['episode']['episode_id'] != episode
                or feature_digest(features) != stored['feature_digest']
                or stored['feature_digest'] != row['feature_digest']):
            raise ValueError('Existing BW feature or source identity changed')
        return features, deepcopy(row)


def _same_visual(visual, radar):
    for key in COMPARABLE_FIELDS:
        a, b = np.asarray(visual[key]), np.asarray(radar[key])
        if a.dtype == np.bool_ or b.dtype == np.bool_:
            equal = np.array_equal(a, b)
        else:
            # Device execution uses FP32 for both paths. Require the actual
            # model inputs to agree, not merely a loose geometric tolerance.
            equal = np.array_equal(a.astype(np.float32), b.astype(np.float32))
        if not equal:
            raise ValueError('BW and visual-only observations disagree in '+key)
    if float(visual['range_scale_m']) != float(radar['range_scale_m']):
        raise ValueError('BW variants changed common public geometric support')


def load_samples(config, ids, split, count=0, include_labels=True):
    if split not in ('train', 'val'):
        raise ValueError('This study never reads final test observations or labels')
    if type(include_labels) is not bool or type(count) is not int or count < 0:
        raise ValueError('Explicit label access and nonnegative frame limit required')
    arm = study_arm(config)
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate requested episodes')
    visual = VisualSensorReader(config, split)
    roi = radar = labels = None
    samples, rows = [], []
    try:
        if not set(ids) <= visual.ids:
            raise ValueError('Requested episode outside fixed train/validation membership')
        roi = ROIAuthenticator(config)
        # Do not instantiate a radar reader or cache on the visual path.
        if arm != 'visual':
            radar = RadarCache(config, split)
        if include_labels:
            labels = Dataset(config['dataset_root'], echo_subdir=config.get('echo_subdir', 'echoes'), verify_sources=True)
        for episode in ids:
            for index in visual.frame_indices(episode, count):
                sample, row, station = visual.observation(episode, index)
                features = sample['features']
                record = visual.episode(episode)
                rgb = (record['root']/record['capture'][index]['rgb_path']).resolve()
                vectors, receipt = roi.load(features, row, rgb)
                if radar is not None:
                    measured, radar_row = radar.frame(episode, index)
                    _same_visual(features, measured)
                    features = measured
                    row.update(radar_cache=radar_row['cache'], radar_feature_digest=radar_row['feature_digest'],
                               radar_calibration=radar_row['radar_calibration'])
                features = dict(features, detector_roi_features=vectors)
                sample = dict(features=features, episode_id=episode, frame_idx=index)
                digest = visual_feature_digest(features) if arm == 'visual' else feature_digest(
                    {k:v for k,v in features.items() if k != 'detector_roi_features'})
                row.update(study_arm=arm, detector_roi_cache=receipt, depth_feature_digest=digest,
                           iq_payload_read=False, radar_features_read=radar is not None,
                           shared_visual_inputs_checked=True)
                if include_labels:
                    truth = labels.targets(episode, index)
                    positions = station.to_local_positions(truth.positions).astype(np.float32)
                    matched = match_ray_labels(features, positions, config)
                    if not matched['diagnostics']['supervision_usable']:
                        raise ValueError('No usable supervision; refusing silent frame deletion')
                    sample.update(positions=positions, formation_index=None,
                        complete_detection_cardinality=len(features['ray_directions']) == len(positions))
                    row.update(label_count=len(positions), supervision=matched['diagnostics'])
                samples.append(sample); rows.append(row)
        metadata = dict(visual.metadata, study_arm=arm, roi_export_sha256=roi.export_sha,
            visual_only=arm == 'visual',
            radar_features_read=radar is not None, iq_read=False, iq_payload_read=False,
            radar_cache_read=radar is not None, radar_loaded_then_dropped=False,
            target_labels_loaded=include_labels, source='public current RGB/detection/navigation plus authenticated frozen ROI')
        if radar is not None:
            metadata['radar_frontend'] = radar.metadata
        return samples, rows, metadata
    finally:
        visual.close()
        if radar is not None:
            radar.close()


def pack_batch(features_list, positions=None, matches=None, *, device='cpu', config=None):
    """Separate packers: visual banks have no radar fields or fake spectra."""
    config = {} if config is None else config
    if study_arm(config) != 'visual':
        return resident.pack_batch(features_list, positions, matches, device=device, config=config)
    features = tuple(features_list)
    if not features:
        raise ValueError('At least one current frame is required')
    validated = tuple(_visual_checked(f) for f in features)
    if any(float(f['range_scale_m']) != float(features[0]['range_scale_m']) for f in features[1:]):
        raise ValueError('Visual batches require one common public geometric support')
    lengths = tuple(len(f['ray_directions']) for f in features)
    width = max(1, max(lengths))
    inputs = dict(member_mask=torch.arange(width)[None] < torch.tensor(lengths)[:, None])
    for key in MEMBER_FIELDS:
        padded = []
        for f, n in zip(features, lengths):
            value = resident._tensor(f[key], key=key)
            padded.append(torch.cat((value, value.new_zeros((width-n,)+tuple(value.shape[1:]))), 0))
        inputs[key] = torch.stack(padded)
    for key in VISUAL_FRAME_FIELDS:
        inputs[key] = torch.stack([resident._tensor(f[key], key=key) for f in features])
    targets = actual = truths = None
    if positions is None:
        if matches is not None:
            raise ValueError('Assignments require separately supplied supervision')
    else:
        truths = tuple(resident._truth(value) for value in positions)
        if len(truths) != len(features) or matches is not None and len(matches) != len(features):
            raise ValueError('One supervision set and assignment per current frame required')
        targets, actual = resident._target_arrays(validated, truths, matches, width, config)
    device = torch.device(device)
    inputs = {key:value.to(device) for key, value in inputs.items()}
    if targets is not None:
        targets = {key:value.to(device) for key, value in targets.items()}
    return PackedBatch(inputs, lengths, targets, actual, truths,
        dict(validated_on_cpu=True, model_inputs_contain_gt=False, visual_only=True, radar_features_read=False))


class StudyFeatureStore(resident.GPUFeatureStore):
    """Reuse exact gather/dropout/rematching, with an independently packed bank."""
    def __init__(self, samples, config, device='cpu'):
        start = perf_counter()
        self.samples = tuple(samples); self.config = dict(config)
        if not self.samples:
            raise ValueError('A resident store requires at least one frame')
        self.features = tuple(s['features'] for s in self.samples)
        labeled = tuple('positions' in s and s['positions'] is not None for s in self.samples)
        if any(labeled) != all(labeled):
            raise ValueError('Mixed labeled and inference-only observations')
        self.plans = tuple(MatchPlan(f, s['positions'], config) for f, s in zip(self.features, self.samples)) if all(labeled) else None
        self.positions = tuple(p.truth for p in self.plans) if self.plans is not None else None
        full = tuple(p.full for p in self.plans) if self.plans is not None else None
        self.bank = pack_batch(self.features, self.positions, full, device=device, config=config)
        self.device = torch.device(device)
        self.valid = tuple(resident._numpy(f['ray_valid']).astype(bool, copy=True) for f in self.features)
        self.width = self.bank.inputs['member_mask'].shape[1]
        if self.plans is not None:
            truth_width = max(1, max(map(len, self.positions)))
            truth = np.zeros((len(self.samples), truth_width, 3), np.float32)
            projections = np.zeros((len(self.samples), self.width, truth_width), np.float32)
            for i, plan in enumerate(self.plans):
                truth[i, :len(plan.truth)] = plan.truth
                projections[i, :plan.n, :len(plan.truth)] = plan.all_depth
            self.truth_bank = torch.from_numpy(truth).to(self.device)
            self.projection_bank = torch.from_numpy(projections).to(self.device)
        self.initialization_seconds = perf_counter()-start


def drop_detection_features(features, keep):
    if features.get('schema') == VISUAL_SCHEMA:
        _visual_checked(features)
        base = {k:v for k,v in features.items() if k != 'detector_roi_features'}
        result = drop_visual_detections(base, keep)
        result['detector_roi_features'] = np.asarray(features['detector_roi_features'])[np.asarray(keep)].copy()
        _visual_checked(result)
        return result
    return evidence.drop_detection_features(features, keep)
