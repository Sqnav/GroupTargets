"""Authenticated old sensor cache plus observed box-scale inputs.

Training labels remain separate sample fields. The public observation loader
never calls ``Dataset.targets`` and never writes the original cache; unseen
test frames can be preprocessed directly from the measured sensor observation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from diagnostics import data as legacy_data
from rvwm.coordinates import FixedStationFrame
from rvwm.data import Dataset, sha256
from rvwm.geometry import rays
from rvwm.group_model import FORMATIONS
from rvwm.group_observation import (describe_observation, feature_digest as base_feature_digest,
                                     validate_features as validate_base_features)
from rvwm.partitions import active_splits
from rvwm.phase_selection import configured_phases, required_phase, selected_frame_indices
from rvwm.types import Camera, Detection
from rvwm.visual_refinement import ExportedRGBFrontend

from .features import DEPTH_SCHEMA, enrich_features, feature_digest

ARCHITECTURE_SCHEMA = 'single_frame_geometry_guided_depth_v2'


def _check_config(config):
    if config.get('architecture_schema') != ARCHITECTURE_SCHEMA:
        raise ValueError('Depth localization requires architecture_schema=' + ARCHITECTURE_SCHEMA)
    if type(config.get('window', 1)) is not int or config.get('window', 1) != 1:
        raise ValueError('Depth localization accepts one current frame only')
    if config.get('group_temporal_enabled', False) is not False or any(
            key in config for key in ('group_temporal_layers', 'group_temporal_heads')):
        raise ValueError('Historical or temporal model inputs are unsupported')


def _cache_value(config, row, expected_sources):
    """Authenticate bytes and the sensor identity before inspecting calibration."""
    if row.get('feature_status') != 'prepared' or not row.get('cache'):
        return None
    path = Path(row['cache']['path']).resolve()
    if not path.is_relative_to(Path(config['group_feature_cache']).resolve()):
        raise ValueError('Original cache path escapes the explicitly bound cache root')
    if not path.exists():
        return None
    if sha256(path) != row['cache']['sha256']:
        raise ValueError('Original feature cache bytes changed: ' + str(path))
    stored = torch.load(path, map_location='cpu', weights_only=False)
    receipt, base = stored['receipt'], stored['features']
    if (receipt['episode_id'] != row['episode_id'] or receipt['frame_idx'] != row['frame_idx']
            or receipt['raw_hashes'] != row['hashes']
            or receipt['source_sha256'] != expected_sources
            or stored['feature_digest'] != row['feature_digest']
            or base_feature_digest(base) != row['feature_digest']):
        raise ValueError('Original cache receipt/features differ from the authenticated inventory')
    validate_base_features(base)
    return base, receipt


def _detections(boxes, scores, pixel_std):
    uv = (boxes[:, :2] + boxes[:, 2:]) / 2
    return [Detection(point.copy(), np.eye(2) * pixel_std ** 2,
                      box.copy(), float(score), np.zeros(128))
            for point, box, score in zip(uv, boxes, scores)]


def _input_receipt(config, frontend, row, base_digest, sensor, *, method):
    # Explicit allowlist: no formation, member-count, target-ID or truth fields.
    return dict(schema=DEPTH_SCHEMA, episode_id=row['episode_id'], frame_idx=int(row['frame_idx']),
        split=row['split'], source_hashes=deepcopy(row['hashes']),
        original_base_feature_digest=base_digest,
        original_cache=deepcopy(row.get('cache')),
        camera_K=np.asarray(sensor['K']).tolist(),
        camera_image_size_wh=[int(sensor['width']), int(sensor['height'])],
        detector_weights_sha256=frontend.metadata['weights_sha256'],
        detector_export_manifest_sha256=frontend.metadata['track_detection_manifest_sha256'],
        split_manifest_sha256=config['split_manifest_sha256'],
        loading_method=method, original_cache_written=False,
        uses_target_labels=False, uses_history=False, uses_future=False)


def _enrich_sample(config, frontend, sample, row, base, sensor, method):
    receipt = _input_receipt(config, frontend, row, base_feature_digest(base), sensor, method=method)
    features = enrich_features(base, sample['detections'], sensor['K'],
                               (sensor['width'], sensor['height']), provenance=receipt)
    new_row = deepcopy(row)
    new_row.update(depth_feature_digest=feature_digest(features), depth_input_receipt=receipt,
                   depth_schema=DEPTH_SCHEMA, original_cache_written=False)
    result = dict(sample, features=features)
    return result, new_row


def _sensor_frame(config, dataset, frontend, episode, index, split, lookup, expected_sources):
    """Sensor-only half of the original cache reader, with no supervision call."""
    original = lookup.get((episode, index))
    cached = _cache_value(config, original, expected_sources) if original is not None else None
    if cached is not None:
        base, receipt = cached
        loaded = dataset._load(episode)
        public = loaded[6]
        if (public['capture_hash'] != original['hashes']['source_trajectory']
                or public['metadata_hash'] != original['hashes']['source_metadata']):
            raise ValueError('Current source capture/metadata differs from the original cache')
        phase = loaded[4][index]['phase']
        if phase not in configured_phases(config):
            raise ValueError('Requested observation is outside the configured capture phases')
        sensor = receipt['sensor_identity']['camera']
        boxes, scores = frontend.track_store.get(episode, index, original['hashes']['rgb'],
            source_frames_sha256=public['capture_hash'], image_size_wh=(sensor['width'], sensor['height']))
        uv = (boxes[:, :2] + boxes[:, 2:]) / 2
        expected_uv = np.asarray([r['uv'] for r in receipt['detections']], float).reshape(-1, 2)
        if len(boxes) != original['detected_count'] or not np.array_equal(uv, expected_uv):
            raise ValueError('Frozen detector boxes differ from the authenticated cached rays')
        camera = Camera(np.asarray(sensor['K']), np.asarray(sensor['position']),
                        np.asarray(sensor['rotation']), sensor['width'], sensor['height'])
        if not np.allclose(rays(uv, camera), base['ray_directions'], rtol=0, atol=1e-7):
            raise ValueError('Frozen detector midpoint directions differ from the cached rays')
        frame = legacy_data._world_frame(episode, loaded, index)
        actual_camera = np.asarray(loaded[4][index]['camera']['camera_position_ned'])
        if not np.allclose(frame.to_local_positions(actual_camera), base['ray_origins'], rtol=0, atol=1e-6):
            raise ValueError('Cached rays use another fixed-station sensor coordinate transform')
        # Do not return original inventory fields containing label statistics.
        row = {k: deepcopy(original[k]) for k in ('episode_id', 'frame_idx', 'split', 'phase',
               'hashes', 'detected_count', 'frame_interval_s', 'feature_status',
               'configured_window', 'cache', 'feature_digest') if k in original}
        detections = _detections(boxes, scores, float(config.get('pixel_std', 3.)))
        sample = dict(features=base, episode_id=episode, frame_idx=index, detections=detections)
        return (*_enrich_sample(config, frontend, sample, row, base, sensor,
                              'authenticated original sensor cache and strict detector export'), frame)
    raw = dataset.observations(episode, index, required_phase=required_phase(configured_phases(config)))
    frame = FixedStationFrame.from_observation(raw)
    adapted = frame.observations(raw)
    detections = frontend.detect(raw)
    base = describe_observation(adapted, detections, config)
    sensor = dict(K=raw.camera.K, width=raw.camera.width, height=raw.camera.height)
    row = dict(episode_id=episode, frame_idx=index, split=split,
        phase=dataset._load(episode)[4][index]['phase'], hashes=dict(raw.source_hashes),
        detected_count=len(detections), frame_interval_s=float(raw.radar.frame_interval_s),
        feature_status='prepared' if detections else 'no_detections', configured_window=1,
        feature_digest=base_feature_digest(base), cache=None)
    sample = dict(features=base, episode_id=episode, frame_idx=index, detections=detections)
    return (*_enrich_sample(config, frontend, sample, row, base, sensor,
                          'native sensor preprocessing; original cache untouched'), frame)


def _context(config, ids, split):
    _check_config(config)
    if split not in ('train', 'val', 'test'):
        raise ValueError('Expected train, val or test split')
    run = legacy_data.verify_original_sources(config)
    dataset = Dataset(config['dataset_root'], echo_subdir=config['echo_subdir'], verify_sources=True)
    partitions = active_splits(dataset, config)
    if len(ids) != len(set(ids)) or not set(ids) <= set(partitions[split]):
        raise ValueError('Requested episodes are duplicated or outside the frozen split')
    lookup = {}; stored_frontend = None
    if split in ('train', 'val'):
        _, lookup, stored_frontend = legacy_data._inventory(str(legacy_data.original_root(config)), split)
    frontend = ExportedRGBFrontend(config, split=split, device='cpu')
    if stored_frontend is not None:
        for key in ('weights_sha256', 'detector_contract', 'track_detection_manifest_sha256',
                    'visual_split_manifest_sha256'):
            if frontend.metadata[key] != stored_frontend[key]:
                frontend.close()
                raise ValueError('Frozen detector identity differs from the original cache: ' + key)
    return dataset, frontend, lookup, run['source_sha256']


def load_samples(config, ids, split, count=0, include_labels=True):
    """Return enriched samples, authenticated frame rows and detector receipt.

    Only ``include_labels=True`` opens ``Dataset.targets``. Those arrays remain
    outside ``sample['features']``. Test frames use measured IQ/RGB directly
    when the original experiment has no prepared cache for that split.
    """
    ids = list(ids)
    _check_config(config)
    if type(include_labels) is not bool:
        raise ValueError('include_labels must be explicitly boolean')
    if include_labels and split in ('train', 'val'):
        samples, rows, metadata = legacy_data.load_cached_samples(config, ids, split, count=count)
        sources = legacy_data.verify_original_sources(config)['source_sha256']
        # The legacy reader already checked strict detector identity and all
        # source bindings. Its authenticated cache receipt supplies calibration.
        enriched, manifests = [], []
        class FrontendReceipt:
            pass
        frontend = FrontendReceipt(); frontend.metadata = metadata
        for sample, row in zip(samples, rows):
            value = _cache_value(config, row, sources)
            if value is None:
                raise ValueError('An authenticated training calibration receipt is unavailable')
            _, receipt = value
            item, manifest = _enrich_sample(config, frontend, sample, row, sample['features'],
                receipt['sensor_identity']['camera'], 'authenticated original training cache and strict detector export')
            enriched.append(item); manifests.append(manifest)
        return enriched, manifests, dict(metadata, depth_schema=DEPTH_SCHEMA,
            depth_sensor_only_features=True, original_cache_written=False)
    dataset, frontend, lookup, sources = _context(config, ids, split)
    samples, rows = [], []
    try:
        for episode in ids:
            for index in selected_frame_indices(dataset, episode, configured_phases(config), count):
                sample, row, frame = _sensor_frame(config, dataset, frontend, episode, index, split, lookup, sources)
                if include_labels:
                    targets = dataset.targets(episode, index)
                    positions = frame.to_local_positions(targets.positions).astype(np.float32)
                    source_metadata = dataset._load(episode)[2]
                    name = source_metadata.get('group_structure', source_metadata.get('scenario_spec', {}).get('formation_type'))
                    sample.update(positions=positions, formation_type=name or 'unknown',
                        formation_index=FORMATIONS.index(name) if name in FORMATIONS else None,
                        complete_detection_cardinality=len(sample['detections']) == len(positions))
                samples.append(sample); rows.append(row)
        return samples, rows, dict(frontend.metadata, depth_schema=DEPTH_SCHEMA,
            depth_sensor_only_features=True, original_cache_written=False)
    finally:
        frontend.close()


def load_observation_features(config, episode, frame_idx, split):
    """Return one deployment feature dictionary, without opening target labels."""
    if type(frame_idx) is not int or frame_idx < 0:
        raise ValueError('frame_idx must be a nonnegative integer')
    dataset, frontend, lookup, sources = _context(config, [episode], split)
    try:
        sample, _, _ = _sensor_frame(config, dataset, frontend, episode, frame_idx, split, lookup, sources)
        return sample['features']
    finally:
        frontend.close()
