"""Read-only full-TRAIN RMSE of an immutable checkpoint and fixed ROI labels.

GT labels are used only for this diagnostic. No training, checkpoint selection,
DEV/TEST access, output-to-GT matching, or oracle world-state IDs occur here.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from rvloc.config import ExperimentConfig, external_path
from rvloc.data import load_manifest
from rvloc.models import build_model
from rvloc.models.checkpoint import SCHEMA as CHECKPOINT_SCHEMA, ARCHITECTURE
from rvloc.runtime import source_identity, write_json, file_sha256
from rvloc.training import _association_factory, _associate

GROUPS = {'rgb': 2, 'single_frame': 3, 'world_fusion': 7}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _checkpoint(path, manifest_sha256, device):
    path = Path(path).expanduser().resolve()
    match = re.fullmatch(r'epoch_(\d{4,})\.pt', path.name)
    _check(match is not None, 'Only committed epoch_NNNN.pt checkpoints may be monitored')
    # Decode the exact byte snapshot whose SHA is recorded, never reopen after
    # hashing. The trainer publishes epoch files using atomic rename.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    _check(payload.get('schema') == CHECKPOINT_SCHEMA and payload.get('architecture') == ARCHITECTURE,
           'Unsupported checkpoint architecture/schema')
    _check(payload.get('config', {}).get('architecture') == ARCHITECTURE,
           'Checkpoint model configuration uses another architecture')
    group = payload.get('group')
    _check(group in GROUPS, 'Only independently trained groups 2/3/7 are supported')
    metadata = payload.get('metadata', {})
    epoch = metadata.get('completed_epochs')
    _check(type(epoch) is int and epoch > 0 and epoch == int(match.group(1)),
           'Checkpoint filename and committed epoch disagree')
    _check(metadata.get('optimizer_updates', 0) > 0 and metadata.get('synthetic_only') is not True,
           'An actually trained checkpoint is required')
    _check(metadata.get('manifest_sha256') == manifest_sha256,
           'Checkpoint belongs to another data manifest')
    _check(metadata.get('center_contract') == 'observer_scatter_at_camera_optical_center'
           and metadata.get('observer_position_reference') == 'camera_optical_center'
           and metadata.get('body_navigation_is_model_input') is False,
           'Checkpoint does not certify camera optical-centre neural inputs')
    _check(metadata.get('method_id') == GROUPS[group], 'Checkpoint method provenance differs')
    model = build_model(group, payload['config'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.core.trained = True
    model.to(device).eval()
    return model, metadata, dict(path=str(path), sha256=digest, method_id=GROUPS[group],
                                 completed_epochs=epoch, bytes=len(raw))


def _index(path, manifest):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    index = json.loads(raw)
    _check(index.get('schema') == 'rvloc.fixed_roi_gt_index.v1'
           and index.get('supervision_only') is True and index.get('normal_inference_input') is False,
           'A frozen supervision-only ROI label index is required')
    _check(index.get('manifest_sha256') == manifest.sha256, 'Fixed labels use another manifest')
    refs = {}
    for row in index['frames']:
        if row['split'] != 'train':
            continue  # DEV metadata may exist in the index; its frame labels are never opened.
        key = row['episode_id'], row['frame_idx']
        _check(key not in refs, 'Duplicate TRAIN fixed label frame')
        refs[key] = row
    expected = {(r.episode_id, r.frame_idx):r for r in manifest.frames('train')}
    _check(refs.keys() == expected.keys() and bool(refs), 'Fixed labels must cover all TRAIN frames')
    return path, hashlib.sha256(raw).hexdigest(), refs, expected


def _bind(observation, frozen, record):
    _check(frozen.get('schema') == 'rvloc.fixed_roi_gt_frame.v1', 'Unsupported fixed frame')
    _check((frozen['episode_id'], frozen['frame_idx'], frozen['physical_time_s']) ==
           (observation.episode_id, observation.frame_idx, observation.physical_time_s), 'Frame/time mismatch')
    source = observation.source_hashes
    _check(frozen['normal_sha256'] == record.normal_sha256 == source['normal']
           and frozen['labels_sha256'] == record.labels_sha256
           and frozen['visual_sha256'] == source['visual'], 'Fixed labels and public sensor identities differ')
    features = observation.features
    n = len(features['bbox_xyxy'])
    _check(n == frozen['detection_count'] and np.array_equal(features['bbox_xyxy'],
           np.asarray(frozen['bbox_xyxy']).reshape(-1, 4)), 'Detection order or bbox content differs')
    for field in ('ray_origins', 'ray_directions', 'ray_depth_bounds'):
        _check(np.array_equal(features[field], np.asarray(frozen[field]).reshape(np.asarray(features[field]).shape)),
               'Fixed ray geometry differs: '+field)
    ids = frozen['gt_member_ids']
    truth = np.asarray(frozen['gt_positions_m'], float).reshape(-1, 3)
    _check(truth.shape == (len(ids), 3) and np.isfinite(truth).all() and len(set(ids)) == len(ids),
           'Malformed fixed GT positions/IDs')
    rows, columns = [], []
    for pair in frozen['training_equivalent_assignment']:
        row, col = pair['detection_index'], pair['gt_index']
        _check(type(row) is int and 0 <= row < n and type(col) is int and 0 <= col < len(ids),
               'Fixed assignment index out of bounds')
        _check(ids[col] == pair['gt_member_id']
               and np.array_equal(truth[col], pair['gt_position_m']), 'Fixed pair GT identity/position disagrees')
        rows.append(row); columns.append(col)
    _check(len(set(rows)) == len(rows) and len(set(columns)) == len(columns), 'Fixed pairs are not one-to-one')
    return dict(rows=np.asarray(rows, dtype=np.int64), columns=np.asarray(columns, dtype=np.int64),
                truth=truth, detection_count=n, gt_count=len(ids))


def _inputs(manifest, index_path, index_sha, refs, expected, modality, cache):
    key = (manifest.sha256, index_sha, modality)
    snapshots = cache.setdefault('input_snapshots', {})
    if key in snapshots:
        return snapshots[key], True
    options = dict(modality=modality)
    if modality == 'rgb_radar':
        options['prepared_only'] = True
    values, seen = [], set()
    for observation in manifest.observations('train', **options):
        frame_key = observation.episode_id, observation.frame_idx
        _check(frame_key in expected and frame_key not in seen, 'Unexpected/duplicate TRAIN observation')
        seen.add(frame_key)
        _check(getattr(observation, 'iq', None) is None, 'Monitoring must not retain/read native IQ')
        ref = refs[frame_key]
        path = (index_path.parent/ref['path']).resolve()
        _check(path.is_relative_to(index_path.parent), 'Fixed annotation path escaped index root')
        raw = path.read_bytes()
        _check(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Fixed frame SHA mismatch')
        fixed = _bind(observation, json.loads(raw), expected[frame_key])
        values.append((observation, fixed))
    _check(seen == expected.keys(), 'Full TRAIN sensor coverage is required')
    snapshots[key] = values
    return values, False


def _errors(positions, valid, fixed):
    positions = np.asarray(positions, dtype=float)
    valid = np.asarray(valid)
    n = fixed['detection_count']
    _check(positions.shape == (n, 3) and valid.shape == (n,) and valid.dtype == np.bool_,
           'Prediction rows do not align with frozen detections')
    _check(np.isfinite(positions[valid]).all(), 'Valid predictions contain nonfinite positions')
    selected = valid[fixed['rows']]
    rows = fixed['rows'][selected]; cols = fixed['columns'][selected]
    # Unit is metres, and the three coordinate squares are summed, not averaged.
    errors = np.linalg.norm(positions[rows]-fixed['truth'][cols], axis=-1)
    return errors, dict(gt_members=fixed['gt_count'], matched_members=len(errors),
                        frozen_pairs=len(fixed['rows']), detections=n,
                        unmatched_outputs=n-len(fixed['rows']), invalid_outputs=int((~valid).sum()))


def compute(config_path, checkpoint_path, fixed_index_path, *, device='cpu', output_dir=None, cache=None):
    """Measure one completed checkpoint on full TRAIN; reuse a caller's dict cache.

    ``cache`` contains only immutable CPU public inputs and offline fixed labels,
    never trainable radar-CNN maps, model state, or GT-based track IDs. The fixed
    checkpoint's normal public visual association is replayed from episode zero.
    """
    started = time.perf_counter()
    config = ExperimentConfig.load(config_path)
    _check(config.manifest is not None, 'A ready explicit manifest is required')
    manifest = load_manifest(config.manifest)
    index_path, index_sha, refs, expected = _index(fixed_index_path, manifest)
    model, metadata, checkpoint = _checkpoint(checkpoint_path, manifest.sha256, device)
    method = checkpoint['method_id']
    modality = 'rgb' if method == 2 else 'rgb_radar'
    data_started = time.perf_counter()
    values, reused = _inputs(manifest, index_path, index_sha, refs, expected, modality,
                             {} if cache is None else cache)
    loading_s = time.perf_counter()-data_started
    association_factory = _association_factory(config, manifest) if method == 7 else None
    if method == 7:
        _check(metadata.get('calibration_sha256') == association_factory.calibration_sha256,
               'World checkpoint and normal visual association calibration differ')
    all_errors, counts = [], dict(gt_members=0, matched_members=0, frozen_pairs=0,
                                  detections=0, unmatched_outputs=0, invalid_outputs=0)
    previous_episode = None
    memory = associator = None
    prior_frame = prior_time = None
    model_started = time.perf_counter()
    with torch.inference_mode():
        for observation, fixed in values:
            if observation.episode_id != previous_episode:
                memory = None
                associator = association_factory() if method == 7 else None
                previous_episode = observation.episode_id
                prior_frame = prior_time = None
            if prior_frame is not None:
                _check(observation.frame_idx == prior_frame+1
                       and np.isclose(observation.physical_time_s-prior_time, 1., atol=1e-9),
                       'TRAIN replay requires causal consecutive physical frames')
            prior_frame, prior_time = observation.frame_idx, observation.physical_time_s
            features = dict(observation.features)
            if method == 7:
                # IDs come from actual public visual matching, never fixed GT IDs.
                ids = _associate(associator, observation)
                features.update(episode_id=observation.episode_id, member_ids=ids,
                                frame_index=observation.frame_idx, timestamp_s=observation.physical_time_s)
            prediction = model(features, memory)
            memory = prediction['state'] if method == 7 else None
            errors, current = _errors(prediction['member_positions'].detach().cpu().numpy(),
                                       prediction['valid_mask'].detach().cpu().numpy(), fixed)
            all_errors.extend(errors.tolist())
            for field, count in current.items():
                counts[field] += count
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    model_s = time.perf_counter()-model_started
    errors = np.asarray(all_errors, float)
    _check(len(values) == len(expected), 'Metric omitted TRAIN frames')
    n = len(errors); total = counts['gt_members']
    result = dict(schema='rvloc.checkpoint_train_rmse.v1', status='evaluated',
        scope='fixed_checkpoint_train_full', split='train', method_id=method,
        completed_epochs=checkpoint['completed_epochs'], checkpoint_path=checkpoint['path'],
        checkpoint_sha256=checkpoint['sha256'], frames=len(values), episodes=len(manifest.splits['train']),
        position_rmse_3d_m=float(np.sqrt(np.mean(errors**2))) if n else None,
        mae_m=float(errors.mean()) if n else None, p95_m=float(np.quantile(errors,.95)) if n else None,
        p99_m=float(np.quantile(errors,.99)) if n else None,
        sse_m2=float(np.square(errors).sum()), no_localization_members=total-n,
        all_gt_success={str(t)+'m':float((errors <= t).sum()/total) if total else None for t in (1,3,5)},
        over_5m=int((errors > 5).sum()), over_10m=int((errors > 10).sum()), **counts,
        labels_used_only_for_scoring=True, inference_uses_gt_ids=False,
        read_dev=False, read_test=False, online_training_statistic=False,
        checkpoint_selection='fixed final epoch unchanged; this diagnostic does not select checkpoints',
        identity=dict(config_path=str(config.path), config_sha256=config.sha256,
                      manifest_path=str(manifest.path), manifest_sha256=manifest.sha256,
                      fixed_index_path=str(index_path), fixed_index_sha256=index_sha,
                      checkpoint=checkpoint, source=source_identity(), tool_sha256=file_sha256(__file__),
                      calibration_sha256=metadata.get('calibration_sha256')),
        elapsed_s=time.perf_counter()-started,
        cost=dict(device=str(device), public_cpu_input_cache_reused=reused,
                  loading_s=loading_s, model_and_scoring_s=model_s, frames=len(values),
                  includes_rgb_detector=False, includes_raw_iq_fft=False,
                  includes_asset_authentication=not reused, parallel_workers=1))
    # Check that a concurrently replaced path cannot be misreported as the
    # checkpoint whose immutable byte snapshot was actually decoded.
    _check(file_sha256(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint changed during evaluation')
    if output_dir is not None:
        output = external_path(output_dir, field='TRAIN RMSE diagnostic output')
        output.mkdir(parents=True, exist_ok=False)
        result['metrics_path'] = str(output/'metrics.json')
        write_json(output/'metrics.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    result = compute(args.config, args.checkpoint, args.index, device=args.device, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
