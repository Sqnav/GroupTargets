"""Frozen visual/32MHz/64MHz ROI evaluation; no final-test observations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bandwidth_study.evaluate import read, sha, write
from precision_study.deployed_infer import DEFAULT_DESCRIPTOR, load_descriptor
from precision_study.evaluation import evaluate_samples
from precision_study.evaluate_run import sanitize_diagnostics
from roi_followup.evaluate import (paired_statistics, episode_differences, compact_metrics, decision,
                                   _member_errors, _difference_stats)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('visual', '32mhz', '64mhz')
PAIRS = {'32mhz_minus_visual': ('visual', '32mhz'), '64mhz_minus_32mhz': ('32mhz', '64mhz'),
         '64mhz_minus_visual': ('visual', '64mhz')}
PRIMARY_PAIRS = ('32mhz_minus_visual', '64mhz_minus_32mhz')
BASELINE_SHA256 = 'f978f0f8f3057047f1cf3207c33dbe331a033feebf59352946e63eb9dee6c396'
EPOCHS = 80
GLOBAL_BATCH = 96
PER_RANK_BATCH = 48
WORLD_SIZE = 2
SEED = 20261125
RUNTIME_CONFIG_KEYS = {'device', 'depth_model_trained', 'completed_optimizer_steps',
                       'completed_training_epochs', 'depth_checkpoint', 'd_evidence_checkpoint',
                       'd_evidence_capture_candidates'}
ARTIFACTS = ('members.jsonl', 'frames.jsonl', 'detections.jsonl', 'summary.json',
             'episodes.json', 'run_info.json', 'input_manifest.json', 'serialization.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def registered_inputs(exp):
    """Authenticate protocol/config/source locks before observation access."""
    from .checkpoint import source_hashes
    exp = Path(exp).resolve()
    plan = read(exp/'run_plan.json')
    if plan.get('protocol_sha256') != sha(exp/'PROTOCOL.json'):
        raise ValueError('Registered protocol changed')
    if plan.get('roster_sha256') != sha(exp/'split_roster.json'):
        raise ValueError('Registered split roster changed')
    if plan.get('source_sha256') != source_hashes():
        raise ValueError('Registered model/training/evaluation sources changed')
    configs = {arm: sha(exp/'configs'/f'{arm}.json') for arm in ARMS}
    if plan.get('config_sha256') != configs:
        raise ValueError('Registered arm configurations changed')
    roster = read(exp/'split_roster.json')
    val = roster['splits']['val']
    ids = val['episode_ids']
    if len(ids) != 150 or len(set(ids)) != 150 or val['frame_count'] != 2569:
        raise ValueError('This protocol requires all 150 validation episodes / 2569 frames')
    train_ids = set(roster['splits']['train']['episode_ids'])
    if len(train_ids) != 700 or roster['splits']['train']['frame_count'] != 11649:
        raise ValueError('Require the complete 700-episode / 11649-frame training inventory')
    if train_ids.intersection(ids):
        raise ValueError('Training and validation episodes overlap')
    split = Path(roster['split_manifest'])
    if sha(split) != roster['split_manifest_sha256']:
        raise ValueError('Original scene split manifest changed')
    return dict(plan=plan, roster=roster, configs=configs)


def identities(exp):
    """Verify all three finished training runs without opening observations."""
    from . import checkpoint
    exp = Path(exp).resolve()
    registered = registered_inputs(exp)
    _, deployment = load_descriptor(DEFAULT_DESCRIPTOR)
    if deployment['checkpoint_sha256'] != BASELINE_SHA256:
        raise ValueError('Production default D has changed')
    roster = registered['roster']
    models = {}
    current_sources = checkpoint.source_hashes()
    for arm in ARMS:
        folder = exp/'training'/arm
        plan = read(folder/'training_plan.json')
        run = read(folder/'run_manifest.json')
        progress = read(folder/'progress.json')
        if (run.get('status') != 'complete' or progress.get('stage') != 'complete'
                or run.get('completed_epochs') != EPOCHS or progress.get('completed_epochs') != EPOCHS
                or plan.get('requested_epochs') != EPOCHS
                or plan.get('maximum_global_batch_size') != GLOBAL_BATCH
                or plan.get('per_rank_batch_size') != PER_RANK_BATCH or plan.get('world_size') != WORLD_SIZE):
            raise ValueError('Require completed 80-epoch, two-GPU per-rank48/global-batch96 training: '+arm)
        schedule = plan.get('learning_rate_schedule', {})
        expected_schedule = dict(schema='frame_epoch_learning_rate_v1', epochs=80.,
            max_learning_rate=3e-4, warmup_epochs=2., minimum_learning_rate_ratio=.1, mode='cosine')
        if (schedule != expected_schedule
                or plan.get('epoch_intervals') != dict(validation=5, save=5)):
            raise ValueError('Registered epoch learning-rate/validation/save budget differs: '+arm)
        if (set(plan.get('validation_episodes', [])) != set(roster['splits']['val']['episode_ids'])
                or plan.get('expected_validation_frames') != 2569
                or set(plan.get('train_episodes', [])) != set(roster['splits']['train']['episode_ids'])
                or plan.get('frames_per_epoch') != roster['splits']['train']['frame_count']):
            raise ValueError('Training used a different episode/frame inventory: '+arm)
        if run.get('source_sha256') != current_sources:
            raise ValueError('Completed training used another source version: '+arm)
        supplied = read(exp/'configs'/f'{arm}.json')
        if supplied.get('study_arm') != arm or ((supplied.get('precision_mode') == 'visual') != (arm == 'visual')):
            raise ValueError('Registered arm and visual/radar input mode differ: '+arm)
        if (supplied.get('study_protocol_sha256') != registered['plan']['protocol_sha256']
                or Path(supplied.get('study_protocol_path', '')).resolve() != exp/'PROTOCOL.json'):
            raise ValueError('Configuration is not bound to this registered protocol: '+arm)
        config_path = exp/'configs'/f'{arm}.json'
        identity = run['training_identity']
        if (identity.get('configuration_sha256') != registered['configs'][arm]
                or identity.get('seed') != 20261118 or identity.get('dropout_probability') != .2):
            raise ValueError('Training configuration identity mismatch: '+arm)
        weights = folder/'depth.rmse.best.pt'
        value = torch.load(weights, map_location='cpu', weights_only=False)
        if (value.get('kind') != checkpoint.KIND or value.get('model_trained') is not True
                or value.get('study_arm') != arm
                or value.get('source_sha256') != current_sources
                or value.get('training_identity') != identity
                or not 1 <= value.get('completed_epochs', 0) <= EPOCHS):
            raise ValueError('Selected epoch checkpoint identity mismatch: '+arm)
        trained = value['model_config']
        # Constructors may materialize frozen defaults absent from the supplied
        # config.  Every explicitly registered semantic key must still match.
        for key in set(supplied) - RUNTIME_CONFIG_KEYS:
            if trained.get(key) != supplied[key]:
                raise ValueError('Selected checkpoint configuration differs: '+arm+':'+key)
        models[arm] = dict(configuration_path=str(config_path), configuration_sha256=sha(config_path),
            checkpoint_path=str(weights), checkpoint_sha256=sha(weights),
            run_sha256=sha(folder/'run_manifest.json'), training_plan_sha256=sha(folder/'training_plan.json'),
            progress_sha256=sha(folder/'progress.json'), training_identity_sha256=digest(identity),
            selected_epoch=value['completed_epochs'],
            model_parameter_count=run['model_parameter_count'],
            checkpoint_added_config_keys=sorted(set(trained)-set(supplied)))
    return dict(status='frozen_validation_only', models=models,
        source_sha256=current_sources, protocol_sha256=sha(exp/'PROTOCOL.json'),
        roster_sha256=sha(exp/'split_roster.json'), run_plan_sha256=sha(exp/'run_plan.json'),
        production_default=deployment, test_observations_or_labels_read=False,
        validation_historically_exposed=True)


def freeze(exp):
    exp = Path(exp).resolve()
    value = identities(exp)
    target = exp/'frozen_validation_identity.json'
    if target.exists():
        if read(target) != value:
            raise ValueError('Frozen validation identity changed')
    else:
        write(target, value)
    return value


def _expected_receipt(frozen, arm):
    return dict(arm=arm, split='val', model_identity=frozen['models'][arm],
        source_sha256=frozen['source_sha256'], protocol_sha256=frozen['protocol_sha256'],
        roster_sha256=frozen['roster_sha256'])


def _verify_completed_arm(exp, arm, frozen):
    out = Path(exp)/'evaluation/val'/arm
    receipt = read(out/'completion_receipt.json')
    if receipt.get('identity') != _expected_receipt(frozen, arm):
        raise ValueError('Completed evaluation identity differs: '+arm)
    actual = {name: sha(out/name) for name in ARTIFACTS}
    if receipt.get('artifact_sha256') != actual:
        raise ValueError('Completed evaluation evidence changed: '+arm)
    return actual


def evaluate_method(exp, arm, device):
    from .checkpoint import load_checkpoint
    from .data import load_samples
    exp = Path(exp).resolve()
    if arm not in ARMS:
        raise ValueError('Unknown registered arm')
    frozen = read(exp/'frozen_validation_identity.json')
    if identities(exp) != frozen:
        raise ValueError('Identity changed before validation observation/label access')
    out = exp/'evaluation/val'/arm
    if (out/'completion_receipt.json').exists():
        _verify_completed_arm(exp, arm, frozen)
        return
    if out.exists():
        raise FileExistsError('Preserve partial evaluation for review before retry: '+str(out))
    item = frozen['models'][arm]
    config = read(item['configuration_path'])
    model = load_checkpoint(item['checkpoint_path'], device=device).eval()
    roster = read(exp/'split_roster.json')['splits']['val']
    tick = time.perf_counter()
    samples, manifest, frontend = load_samples(config, roster['episode_ids'], 'val', count=0, include_labels=True)
    loading_s = time.perf_counter()-tick
    keys = [(r['episode_id'], int(r['frame_idx'])) for r in manifest]
    if (len(manifest) != roster['frame_count'] or len(set(keys)) != len(keys)
            or set(r['episode_id'] for r in manifest) != set(roster['episode_ids'])):
        raise ValueError('Missing, duplicated, or undeclared validation frames')
    info = dict(**_expected_receipt(frozen, arm), device=device,
        input_frontend=frontend, data_loading_s=loading_s, labels_used_for_prediction=False,
        labels_used_only_for_scoring=True, test_opened=False,
        validation_has_checkpoint_selection_and_historical_exposure=True,
        latency_scope='real scalar sensor-only model prediction and stage capture on cached feature inputs; excludes RGB detector/ROI extraction, FFT, disk I/O, and GT scoring; no prediction/score replay',
        observation_scope=('single current RGB/ray/ROI/navigation; no IQ or derived spectrum' if arm == 'visual' else 'single current RGB/ray/ROI/navigation plus registered '+arm+' mixed-IQ RA/RE'),
        radar_observations_read=arm != 'visual', offline_strata_never_used_for_prediction=True)
    def pulse(row):
        print(json.dumps(dict(stage='roi_radar_evaluation', arm=arm, **row)), flush=True)
    assert_observation_contract(samples, arm)
    result = evaluate_samples(model, samples, manifest, config, radar_required=arm != 'visual', capture_stages=True,
                              progress_callback=pulse, run_info=info)
    attach_offline_strata(result['members'], samples, read(exp/'split_roster.json'))
    out.mkdir(parents=True)
    nonfinite = []
    for group in ('members', 'frames', 'detections'):
        with (out/(group+'.jsonl')).open('x') as stream:
            for index, row in enumerate(result[group]):
                safe = sanitize_diagnostics(row, f'{group}[{index}]', nonfinite)
                stream.write(json.dumps(safe, ensure_ascii=False, allow_nan=False)+'\n')
    for group in ('summary', 'episodes'):
        write(out/(group+'.json'), result[group])
    write(out/'run_info.json', info)
    write(out/'input_manifest.json', dict(frames=manifest, frontend=frontend))
    write(out/'serialization.json', dict(primary_metrics_changed=False, diagnostic_nonfinite=nonfinite))
    write(out/'completion_receipt.json', dict(identity=_expected_receipt(frozen, arm),
                                            artifact_sha256={name: sha(out/name) for name in ARTIFACTS}))


def assert_observation_contract(samples, arm):
    """A is a true visual input route, not a zeroed radar-payload ablation."""
    if arm not in ARMS:
        raise ValueError('Unknown study arm')
    forbidden = {'range_m', 'azimuth_rad', 'elevation_rad', 'spectrum', 'power_volume', 'complex_iq'}
    for sample in samples:
        fields = set(sample['features'])
        if arm == 'visual':
            found = sorted(key for key in fields if key in forbidden or str(key).lower().startswith(
                ('ra_', 're_', 'rd_', 'ae_', 'iq', 'radar_')))
            if found:
                raise ValueError('Visual route contains forbidden radar observations/axes: '+','.join(found))
        elif not {'ra_power', 're_power', 'range_m', 'azimuth_rad', 'elevation_rad'} <= fields:
            raise ValueError('Radar arm is missing its declared spectra/axes')


def attach_offline_strata(members, samples, roster):
    """Attach measured box context after prediction; GT only assigns rows."""
    observed = {(s['episode_id'], int(s['frame_idx'])): s['features'] for s in samples}
    assets = {}
    for row in roster.get('episodes', []):
        for field in ('model_asset', 'uav_model', 'asset', 'vehicle_model'):
            if isinstance(row.get(field), str) and row[field]:
                assets[row.get('episode', row.get('episode_id'))] = row[field]
                break
    for row in members:
        context = {}
        point = row.get('truth_position_base_m')
        if point is not None:
            context['GT_base_range_m'] = float(np.linalg.norm(np.asarray(point, float)))
        asset = assets.get(row['episode_id'])
        if asset is not None:
            context['metadata_model_asset'] = asset
        f = observed.get((row['episode_id'], int(row['frame_idx'])))
        index = row.get('detection_index')
        if f is not None and row.get('matched') and index is not None:
            boxes = np.asarray(f.get('bbox_xyxy', []), float).reshape(-1, 4)
            scale = np.asarray(f.get('visual_scale', []), float)
            if len(scale) > index and scale.ndim == 2 and scale.shape[1] >= 3:
                value = float(scale[index, 2])
                if np.isfinite(value):
                    context['observed_sqrt_normalized_bbox_area'] = value
            if len(boxes) > index:
                box = boxes[index]
                lo = np.maximum(box[:2], boxes[:, :2]); hi = np.minimum(box[2:], boxes[:, 2:])
                overlap = np.maximum(hi-lo, 0.).prod(axis=-1)
                area = np.maximum(boxes[:, 2:]-boxes[:, :2], 0.).prod(axis=-1)
                own = max(float(np.prod(np.maximum(box[2:]-box[:2], 0.))), 0.)
                iou = np.divide(overlap, own+area-overlap, out=np.zeros(len(boxes)), where=own+area-overlap > 0)
                iou[index] = 0.
                context['observed_max_other_box_IoU'] = float(iou.max()) if len(iou) else 0.
        row['offline_strata'] = context


def verify_fair_correspondence(members, frames):
    """Check actual observed geometry and fixed association, not just code names."""
    keyed = {arm:{(r['episode_id'], int(r['frame_idx']), int(r['truth_index'])):r for r in members[arm]}
             for arm in ARMS}
    ref = keyed['visual']
    for arm in ARMS:
        if len(keyed[arm]) != len(members[arm]) or set(keyed[arm]) != set(ref):
            raise ValueError('Deleted, duplicated or different GT members across arms')
        for key, row in keyed[arm].items():
            old = ref[key]
            if (bool(row.get('matched')) != bool(old.get('matched'))
                    or row.get('detection_index') != old.get('detection_index')):
                raise ValueError('Observed-bearing/support association differs across arms')
            a, b = row.get('truth_position_base_m'), old.get('truth_position_base_m')
            if a is None or b is None:
                if a != b:
                    raise ValueError('GT position availability differs across arms')
            elif np.asarray(a).shape != np.asarray(b).shape or not np.allclose(a,b,rtol=0.,atol=1e-7):
                raise ValueError('GT positions differ across arms')
    frame_keys = lambda rows:{(r['episode_id'], int(r['frame_idx'])):r for r in rows}
    reference = frame_keys(frames['visual'])
    for arm in ARMS:
        rows = frame_keys(frames[arm])
        if len(rows) != len(frames[arm]) or set(rows) != set(reference):
            raise ValueError('Different or duplicated declared evaluation frames')
        for key, row in rows.items():
            old = reference[key]
            for name in ('truth_count', 'detection_count'):
                if row.get(name) != old.get(name):
                    raise ValueError('Different visual detections or GT inventory across arms')
            for name in ('ray_origin_base_m', 'ray_direction_base', 'ray_valid', 'ray_depth_bounds_m', 'bbox_xyxy'):
                a, b = row.get(name), old.get(name)
                if a is None or b is None:
                    if a != b:
                        raise ValueError('Observed geometry availability differs: '+name)
                elif np.asarray(a).shape != np.asarray(b).shape or not np.allclose(a, b, rtol=0., atol=1e-6):
                    raise ValueError('Observed geometry differs: '+name)
    return dict(fixed_members=len(ref), frames=len(reference), observed_geometry_tolerance=1e-6,
                identical_fixed_bearing_support_assignment=True, different_prediction_validity_allowed=True)


def group_error_decomposition(members):
    """Offline common bias + relative-position MSE on identical member subsets.

    A missing/invalid localization is reported, never assigned a zero residual.
    The decomposition is a GT-assisted diagnostic and does not correct output.
    """
    keyed = {}
    for arm, rows in members.items():
        _member_errors(rows)
        keyed[arm] = {(r['episode_id'], int(r['frame_idx']), int(r['truth_index'])):r for r in rows}
    if not keyed:
        raise ValueError('No methods for group diagnostic')
    reference = next(iter(keyed.values()))
    if any(set(rows) != set(reference) for rows in keyed.values()):
        raise ValueError('Group diagnostic requires all GT members, including failures')
    groups = defaultdict(list)
    for key in sorted(reference):
        groups[key[:2]].append(key)
    accum = {arm:dict(squared_error=0., group_bias=0., relative=0.) for arm in keyed}
    frame_rows = []
    total_common = 0
    for (episode, frame), keys in groups.items():
        common = [key for key in keys if all(row[key].get('matched', False)
                  and row[key]['status'] == 'localized' for row in keyed.values())]
        n = len(common); total_common += n
        record = dict(episode_id=episode, frame_idx=frame, all_GT_members=len(keys),
            common_matched_localized_members=n, excluded_GT_members=len(keys)-n,
            failure_counts={arm:sum(row[key]['status'] != 'localized' for key in keys) for arm,row in keyed.items()},
            methods={})
        for arm, rows in keyed.items():
            if not n:
                record['methods'][arm] = dict(total_MSE_m2=None, group_bias_MSE_m2=None,
                                             relative_member_MSE_m2=None, identity_residual_m2=None)
                continue
            residuals = []
            for key in common:
                truth = np.asarray(rows[key]['truth_position_base_m'], float)
                original_truth = np.asarray(reference[key]['truth_position_base_m'], float)
                estimate = np.asarray(rows[key]['predicted_position_base_m'], float)
                if (truth.shape != (3,) or estimate.shape != (3,) or not np.isfinite(estimate).all()
                        or not np.isfinite(truth).all() or not np.allclose(truth, original_truth, atol=1e-7, rtol=0.)):
                    raise ValueError('Invalid or inconsistent group diagnostic positions')
                residual = estimate-truth
                if not np.isclose(np.linalg.norm(residual), rows[key]['error_3d_m'], atol=1e-5, rtol=1e-5):
                    raise ValueError('Stored position error differs from residual for group diagnostic')
                residuals.append(residual)
            residuals = np.asarray(residuals)
            mean = residuals.mean(axis=0)
            bias = float(mean@mean)
            relative = float(np.mean(np.sum((residuals-mean)**2, axis=1)))
            mse = float(np.mean(np.sum(residuals**2, axis=1)))
            identity_error = mse-bias-relative
            if not np.isclose(identity_error, 0., atol=1e-8, rtol=0.):
                raise ValueError('Group bias/relative decomposition identity failed')
            record['methods'][arm] = dict(total_MSE_m2=mse, group_bias_MSE_m2=bias,
                relative_member_MSE_m2=relative, group_bias_vector_m=mean.tolist(),
                identity_residual_m2=identity_error)
            accum[arm]['squared_error'] += n*mse
            accum[arm]['group_bias'] += n*bias
            accum[arm]['relative'] += n*relative
        frame_rows.append(record)
    summary = {}
    for arm, sums in accum.items():
        raw = sums['squared_error']/total_common if total_common else None
        bias = sums['group_bias']/total_common if total_common else None
        relative = sums['relative']/total_common if total_common else None
        summary[arm] = dict(common_matched_localized_members=total_common,
            total_MSE_m2=raw, group_bias_MSE_m2=bias, relative_member_MSE_m2=relative,
            total_RMSE_m=np.sqrt(raw).item() if raw is not None else None,
            group_bias_RMSE_m=np.sqrt(bias).item() if bias is not None else None,
            relative_member_RMSE_m=np.sqrt(relative).item() if relative is not None else None,
            group_bias_fraction=bias/raw if raw else (0. if raw == 0 else None),
            all_GT_members=len(reference), excluded_GT_members=len(reference)-total_common,
            method_failure_count=sum(row['status'] != 'localized' for row in keyed[arm].values()))
    return dict(scope='GT-assisted offline diagnostic only; not deployable correction or normal accuracy',
        subset='same matched and valid-localized GT members shared by all methods within each frame',
        aggregation='frame statistics weighted by common member count, not equal frame weights',
        identity='mean(||residual||^2) = ||mean(residual)||^2 + mean(||residual-mean(residual)||^2)',
        all_GT_members=len(reference), frames=len(groups), common_matched_localized_members=total_common,
        zero_common_member_frames=sum(r['common_matched_localized_members'] == 0 for r in frame_rows),
        summary=summary, frame_rows=frame_rows)


def stratified_metrics(members):
    """Fixed offline range/box-scale/overlap bins; unavailable rows stay visible."""
    reference = {(r['episode_id'], int(r['frame_idx']), int(r['truth_index'])):r for r in members['visual']}
    definitions = {
        'GT_base_range_m': [(0.,150.), (150.,250.), (250.,350.), (350.,float('inf'))],
        'observed_sqrt_normalized_bbox_area': [(0.,.02), (.02,.04), (.04,.08), (.08,float('inf'))],
        'observed_max_other_box_IoU': [(0.,1e-12), (1e-12,.1), (.1,.3), (.3,1.00000001)]}
    groups = {}
    coverage = {}
    for field, bins in definitions.items():
        labels = [f'[{low:g},{high:g})' for low,high in bins]
        grouped = {label:[] for label in [*labels, 'unavailable']}
        for key, row in reference.items():
            value = row.get('offline_strata', {}).get(field)
            label = 'unavailable'
            if value is not None and np.isfinite(value):
                label = next((name for name,(low,high) in zip(labels,bins) if low <= value < high), 'unavailable')
            grouped[label].append(key)
        groups[field] = grouped
        coverage[field] = len(reference)-len(grouped['unavailable'])
    assets = defaultdict(list)
    for key, row in reference.items():
        assets[row.get('offline_strata', {}).get('metadata_model_asset', 'unavailable')].append(key)
    groups['metadata_model_asset'] = dict(assets)
    coverage['metadata_model_asset'] = len(reference)-len(assets.get('unavailable', []))
    errors_by_arm = {arm:_member_errors(rows) for arm,rows in members.items()}
    if any(set(errors) != set(reference) for errors in errors_by_arm.values()):
        raise ValueError('Stratified comparisons require every declared GT member')
    results = {}
    for field, bins in groups.items():
        results[field] = {}
        for label, keys in bins.items():
            results[field][label] = {}
            for arm, errors in errors_by_arm.items():
                values = np.asarray([errors[key] for key in keys])
                finite = values[np.isfinite(values)]
                results[field][label][arm] = dict(all_GT_members=len(keys), localized_members=len(finite),
                    failure_count=len(keys)-len(finite), rmse_m=float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
                    p95_m=float(np.quantile(finite,.95)) if len(finite) else None,
                    all_GT_1m_success=float(np.mean(values <= 1.)) if len(keys) else None)
    return dict(scope='offline-only fixed strata; values selected from same visual/GT rows for every method',
        all_GT_members=len(reference), available_member_counts=coverage, strata=results,
        unavailable_labels_retained=True, box_overlap_is_not_SNR=True,
        model_asset_note='Only explicit scalar asset/model metadata is used; no type inferred from IDs')


def compare(exp):
    exp = Path(exp).resolve()
    frozen = read(exp/'frozen_validation_identity.json')
    members, frames, summaries = {}, {}, {}
    for arm in ARMS:
        _verify_completed_arm(exp, arm, frozen)
        folder = exp/'evaluation/val'/arm
        members[arm] = [json.loads(line) for line in (folder/'members.jsonl').read_text().splitlines()]
        frames[arm] = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        summaries[arm] = read(folder/'summary.json')
    correspondence = verify_fair_correspondence(members, frames)
    pairs = {name: paired_statistics(members[base], members[new], draws=2000, seed=SEED)
             for name,(base,new) in PAIRS.items()}
    differences = {name: episode_differences(members[base], members[new]) for name,(base,new) in PAIRS.items()}
    write(exp/'episode_differences.json', dict(paired_episodes=differences,
        difference_direction='second named arm minus first; positive net MSE gain favors second', all_GT_failures_retained=True))
    decomposition = group_error_decomposition(members)
    write(exp/'group_error_decomposition.json', decomposition)
    strata = stratified_metrics(members)
    write(exp/'strata.json', strata)
    decisions = {name:decision(summaries[PAIRS[name][0]], summaries[PAIRS[name][1]], pairs[name]) for name in PRIMARY_PAIRS}
    out = dict(status='complete', metrics={arm:compact_metrics(summaries[arm]) for arm in ARMS},
        comparisons=pairs, primary_comparisons=list(PRIMARY_PAIRS),
        auxiliary_comparisons=['64mhz_minus_visual'], decisions=decisions,
        fixed_correspondence_verification=correspondence,
        group_error_decomposition_summary=decomposition['summary'],
        validation_only=True, validation_historically_exposed=True, production_default_changed=False,
        test_observations_or_labels_read=False, more_experiments_started=False,
        evidence_paths={arm:str((exp/'evaluation/val'/arm).relative_to(exp)) for arm in ARMS})
    write(exp/'comparison.json', out)
    _write_report(exp, frozen, summaries, pairs, decisions, decomposition, strata)
    return out


def _format(value, precision=4):
    return '未定义' if value is None else f'{value:.{precision}f}'


def _write_report(exp, frozen, summaries, pairs, decisions, decomposition, strata):
    lines = ['# 原生 ROI 下视觉、32MHz 与64MHz雷达的同条件单帧定位对照', '',
        '三组分别为 A=visual（原生冻结ROI、检测框尺度、视觉射线和观察机导航；不读取IQ、动态谱或雷达派生特征），B=32mhz（相同视觉/导航加32MHz混合RA/RE），C=64mhz（相同视觉/导航加64MHz混合RA/RE）。A不是将雷达置零的推理消融，三组均独立从零训练。', '',
        '三组各用双卡、每卡batch48、global batch96，固定80完整frame epoch；训练700个episode/11649帧，每组真实帧曝光931920次。每epoch按全局96帧分批且保留尾批，122次更新/epoch、合计9760次内部优化器更新；用户预算仍为80epoch。seed20261118、检测dropout0.2；相同完整epoch sampler、等episode权重、warmup2epoch和3e-4→0.1倍cosine，均每5epoch验证并据开发验证RMSE选择权重。观察机导航在三组保持公平供给。参数量和成本分别报告，不声称视觉与雷达计算量相同。', '',
        '**全部评价仅使用150个历史开发验证episode、2569帧。该集合参与checkpoint选择，也已有历史选型暴露；此前历史test也不能视为全新独立证据，本轮不读取该test的观测或标签。没有独立新航迹泛化结论，生产默认D始终不变。**', '',
        '64MHz对应带宽和频率采样数一起变化（32→64MHz、128→256频点，频间隔不变），同时影响分辨率及处理增益；不能将全部收益归因于带宽分辨率，也不能直接对比不同历史训练日志归因。', '',
        '| 方法 | 参数量 | 最佳epoch | 3D RMSE m | P95 m | 全GT≤1m | 全GT≤3m | 全GT≤5m | 有效定位/GT | 实测缓存输入ms/帧 | GPU峰值分配MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        s=summaries[arm];e=s['member_position_error_m'];item=frozen['models'][arm]
        latency=s['inference_seconds_per_frame']['mean']
        lines.append(f"| {arm} | {item['model_parameter_count']} | {item['selected_epoch']} | {_format(e['rmse'])} | {_format(e['p95_abs'])} | {s['success_ratio_1m']:.2%} | {s['success_ratio_3m']:.2%} | {s['success_ratio_5m']:.2%} | {s['localized_matched_members']}/{s['truth_members']} | {_format(None if latency is None else latency*1000,2)} | {s.get('gpu_peak_allocated_bytes',0)/1024**2:.2f} |")
    lines += ['', '主要比较预先固定为 B−A（32MHz雷达相对视觉的增益）与 C−B（64MHz的额外增益）；C−A仅作辅助，不在结果出来后替换主比较。2000次整episode成对bootstrap、seed20261125，95%区间用于描述、97.5%区间用于两项主要比较的校正门槛。开发通过要求RMSE差的97.5%区间上界<0、P95不升、全GT1米成功率不降、定位失败数不增；只支持后续独立新航迹确认，不自动部署。', '']
    for name,pair in pairs.items():
        t=pair['all_GT_1m_transition']
        conclusion=('通过' if decisions[name]['development_criterion_passed'] else '未通过') if name in decisions else '辅助描述，不用于晋级'
        lines.append(f"- {name}：RMSE差 {_format(pair['rmse_m_delta'])} m；95%区间 {pair['rmse_m_delta_ci95']}；97.5%区间 {pair['rmse_m_delta_ci97_5']}；{conclusion}。1米内→外 {t['inside_to_outside']}，外→内 {t['outside_to_inside']}；全部转移、有害比例及净平方误差收益见comparison.json。")
    lines += ['', '整组共同偏差与成员相对误差：仅对每帧三组共同匹配且定位有效的同一GT成员子集，离线计算残差e。精确恒等式为 mean(||e||²)=||mean(e)||²+mean(||e−mean(e)||²)。跨帧按共同成员数量加权，不能把小群与大群等权后宣称是成员MSE。', '',
        f"共同有效成员 {decomposition['common_matched_localized_members']}/{decomposition['all_GT_members']}；零共同成员帧 {decomposition['zero_common_member_frames']}。漏检/未定位和被共同子集排除的成员数量完整保留在group_error_decomposition.json及主要指标中。此处使用GT中心只为解释误差，未对预测作GT校正，分解结果不能计为正常定位精度，也不能代表不可见完整群体。", '',
        '| 方法 | 共同子集RMSE m | 共同偏差RMSE m | 相对成员RMSE m | 共同偏差占MSE比例 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        d=decomposition['summary'][arm]
        lines.append(f"| {arm} | {_format(d['total_RMSE_m'])} | {_format(d['group_bias_RMSE_m'])} | {_format(d['relative_member_RMSE_m'])} | {_format(d['group_bias_fraction'])} |")
    lines += ['', '逐阶段initial、global proposal/blend、第一轮和第二轮误差及有害比例保存在各summary.json的stage_diagnostics/stage_update_diagnostics；逐成员、帧、检测与episode保存在evaluation/val/{arm}/。全部GT保留在1/3/5米成功率分母；RMSE/P95只统计有效有限定位，并同步报告失败数。逐episode差值单独保存为episode_differences.json。', '',
        '预固定离线分层保存在strata.json：GT基站距离、观测框归一化尺度、观测框最大IoU重叠及明确的机型/资产元数据。没有匹配框或缺失标签的成员进入unavailable分层，没有静默删除；框重叠不是SNR，未以此冒充噪声标定。',
        f"资产/机型标签实际覆盖 {strata['available_member_counts']['metadata_model_asset']}/{strata['all_GT_members']}；没有明确标签时不从ID猜测机型。", '',
        '推理时延实际执行冻结模型单帧sensor-only forward和阶段捕获，CUDA前后同步；不是准备好的评分/预测重放。输入已缓存，时延不含检测、ROI导出、FFT、读盘及GT评分，加载耗时另列run_info.json；GPU峰值显存保存在各summary.json。A的特征入口严禁谱payload和谱轴，评价显式radar_required=False。', '',
        '固定比较完成后停止，不根据评价增加网络、阈值、种子或组合。无论结果如何，旧代码、原IQ/图片、历史权重与生产入口保持原位；没有部署，因此没有生产回滚操作。', '',
        f'完整复现：`../.venv/bin/python -m roi_radar_study.evaluate --experiment {exp} --phase finalize --device cuda:0`。',
        f'只读核验：`../.venv/bin/python -m roi_radar_study.evaluate --experiment {exp} --phase validate`。',
        '配置/协议/来源：configs/、PROTOCOL.json、run_plan.json、split_roster.json、frozen_validation_identity.json；权重：training/{arm}/depth.rmse.best.pt；完整训练日志和进度：training/{arm}/。']
    (exp/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')


def validate_completion(exp):
    """Read-only integrity validation; no observation reader/model invocation."""
    exp = Path(exp).resolve()
    frozen = read(exp/'frozen_validation_identity.json')
    if identities(exp) != frozen:
        raise ValueError('Frozen completed evaluation identity changed')
    files = {}
    for arm in ARMS:
        hashes = _verify_completed_arm(exp, arm, frozen)
        for name, value in hashes.items():
            files[f'evaluation/val/{arm}/{name}'] = value
        name = f'evaluation/val/{arm}/completion_receipt.json'
        files[name] = sha(exp/name)
    for name in ('comparison.json', 'episode_differences.json', 'group_error_decomposition.json', 'strata.json', 'REPORT_CN.md', 'frozen_validation_identity.json'):
        files[name] = sha(exp/name)
    receipt = dict(status='complete', evidence_sha256=files,
        production_default_changed=False, test_observations_or_labels_read=False)
    existing = exp/'delivery_verification.json'
    if existing.exists() and read(existing) != receipt:
        raise ValueError('Final comparison/report evidence changed')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT/'results/roi_radar_20261125')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--phase', choices=('finalize', 'validate'), required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    exp = args.experiment.resolve()
    if args.phase == 'finalize':
        freeze(exp)
        for arm in ARMS:
            evaluate_method(exp, arm, args.device)
        compare(exp)
        value = validate_completion(exp)
        write(exp/'delivery_verification.json', value)
    else:
        value = validate_completion(exp)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return value


if __name__ == '__main__':
    main()
