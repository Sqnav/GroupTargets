"""Common offline evaluation for single-frame depth localizers.

Labels are used only after sensor-only prediction, for a fixed bearing/support
assignment and explicitly named oracle diagnostics. Every truth instance has
an output row, including misses, rejected frames and invalid predictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.geometry import freeze_correspondence, ray_truth_geometry
from depth_localization.evaluation import prediction_rejection
from rvwm.ray_scoring import observation_rejection_reason

THRESHOLDS = (1, 2, 3, 5)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def distribution(values):
    a = np.asarray([v for v in values if v is not None], float)
    if not len(a):
        return dict(count=0, mean=None, mae=None, rmse=None, p50_abs=None,
                    p95_abs=None, max_abs=None)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite errors must be recorded as failures, not averaged')
    return dict(count=len(a), mean=float(a.mean()), mae=float(np.abs(a).mean()),
                rmse=float(np.sqrt(np.mean(a*a))), p50_abs=float(np.quantile(np.abs(a), .5)),
                p95_abs=float(np.quantile(np.abs(a), .95)), max_abs=float(np.abs(a).max()))


def _spherical(point):
    point = np.asarray(point, float)
    return (float(np.linalg.norm(point)), float(np.arctan2(point[1], point[0])),
            float(np.arctan2(point[2], np.hypot(point[0], point[1]))))


def _wrapped_angle(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


@contextmanager
def capture_refinements(model):
    """Observe intermediate computations in memory, without changing sources.

    The wrapper returns the original tensors unchanged. It is removed before
    returning, including on exceptions. Captured tensors never use labels.
    """
    captures = []
    raw_global = []
    original = getattr(model, '_support_residual', None)
    had_instance = '_support_residual' in getattr(model, '__dict__', {})
    previous_instance = model.__dict__.get('_support_residual') if had_instance else None
    handle = None
    if original is not None:
        def observe(base, raw, bounds, *args, **kwargs):
            result = original(base, raw, bounds, *args, **kwargs)
            captures.append(dict(base=base.detach().clone(), result=result.detach().clone()))
            return result
        model._support_residual = observe
    if hasattr(model, 'global_head'):
        handle = model.global_head.register_forward_hook(
            lambda module, arguments, result: raw_global.append(result.detach().clone()))
    try:
        yield captures, raw_global
    finally:
        if handle is not None:
            handle.remove()
        if original is not None:
            if had_instance:
                model._support_residual = previous_instance
            else:
                delattr(model, '_support_residual')


def predict_stages(model, features, capture=True, allow_untrained=False):
    captures, raw_global = [], []
    if capture:
        with capture_refinements(model) as (captures, raw_global), torch.no_grad():
            scene = model.predict(features, allow_untrained_validation=allow_untrained)
    else:
        with torch.no_grad():
            scene = model.predict(features, allow_untrained_validation=allow_untrained)
    scene = serializable(scene)
    stages = {}
    if 'initial_depths' in scene:
        stages['initial'] = np.asarray(scene['initial_depths'], float)
    diagnostics = scene.get('sampling_diagnostics', {})
    if 'global_depths' in diagnostics:
        stages['global_proposal'] = np.asarray(diagnostics['global_depths'], float)
    elif raw_global and hasattr(model, '_bounded'):
        # The frozen v2 output omitted the global proposal, but its exact
        # sensor-only value is determined by the recorded global-head logits.
        with torch.no_grad():
            bounds = model._tensor(features['ray_depth_bounds'])
            prior = model.depth_prior_m.expand(len(bounds))
            proposal = model._bounded(prior, raw_global[-1][:, 0], bounds)
            stages['global_proposal'] = proposal.detach().cpu().numpy().astype(float)
    for i, row in enumerate(captures):
        prefix = 'first' if i == 0 else 'second' if i == 1 else f'round_{i+1}'
        stages[prefix+'_global_blend'] = row['base'].cpu().numpy().astype(float)
        stages[prefix+'_refined'] = row['result'].cpu().numpy().astype(float)
    for i, depth in enumerate(diagnostics.get('stage_depths_m', [])):
        prefix = 'first' if i == 0 else 'second' if i == 1 else f'round_{i+1}'
        stages[prefix+'_refined'] = np.asarray(depth, float)
    stages['final'] = np.asarray(scene['depths'], float)
    return scene, stages


def _geometric_gate(features, config):
    n = len(features['ray_directions'])
    if not n:
        return 'no_detections'
    if n > int(config.get('group_max_members', 24)):
        return 'unsupported_detected_count'
    if not np.asarray(features['ray_valid'], bool).any():
        return 'no_valid_rays'
    return None


def summarize(member_rows, frame_rows):
    """Summarize all declared truth members, with finite errors only in RMSE."""
    valid = [r for r in member_rows if r['status'] == 'localized']
    matched = [r for r in member_rows if r.get('matched', False)]
    n = len(member_rows)
    detected = sum(r['detection_count'] for r in frame_rows)
    error = distribution(r['error_3d_m'] for r in valid)
    parallel = distribution(r['parallel_error_m'] for r in valid)
    transverse = distribution(r['perpendicular_error_m'] for r in valid)
    out = dict(frames=len(frame_rows), truth_members=n, detection_instances=detected,
        matched_detection_instances=len(matched), localized_matched_members=len(valid),
        unmatched_truth_members=n-len(matched), unmatched_detection_instances=detected-len(matched),
        unlocalized_matched_members=len(matched)-len(valid),
        member_position_error_m=error,
        range_error_m=distribution(r['base_range_error_m'] for r in valid),
        azimuth_error_deg=distribution(r['base_azimuth_error_deg'] for r in valid),
        elevation_error_deg=distribution(r['base_elevation_error_deg'] for r in valid),
        projection_depth_error_m=distribution(r['depth_error_to_projection_m'] for r in valid),
        parallel_error_m=parallel, perpendicular_error_m=transverse,
        parallel_mse_fraction=(parallel['rmse']**2/error['rmse']**2) if error['rmse'] else None,
        perpendicular_mse_fraction=(transverse['rmse']**2/error['rmse']**2) if error['rmse'] else None,
        matched_detection_rate=len(matched)/n if n else None,
        localization_effective_rate=len(valid)/n if n else None,
        localized_given_matched_detection_rate=len(valid)/len(matched) if matched else None,
        failure_counts=dict(Counter(r['status'] for r in member_rows if r['status'] != 'localized')),
        rejected_frame_counts=dict(Counter(r['status'] for r in frame_rows if r['status'] != 'valid')),
        coverage_denominator='every declared GT member; misses and rejected/out-of-support predictions are failures',
        detection_rate_definition='fixed observed-bearing/support matching, not an IoU detector benchmark',
        error_scope='finite localized members under fixed observed-bearing correspondence',
        valid_mask_is_calibrated_confidence=False)
    for t in THRESHOLDS:
        count = sum(r['error_3d_m'] <= t for r in valid)
        out[f'success_count_{t}m'] = count
        out[f'success_ratio_{t}m'] = count/n if n else None
    oracle = {}
    for kind in ('line', 'support'):
        values = [r['oracle'][kind+'_error_m'] for r in matched if 'oracle' in r]
        row = dict(position_error_m=distribution(values))
        for t in THRESHOLDS:
            row[f'success_count_{t}m'] = sum(v <= t for v in values)
            row[f'success_ratio_all_truth_{t}m'] = sum(v <= t for v in values)/n if n else None
        oracle[kind] = row
    oracle.update(projection_outside_support_count=sum(not r['oracle']['projection_inside_support']
        for r in matched if 'oracle' in r), matched_members=len(matched),
        scope='GT-assisted geometry diagnostic only; not deployed performance or a sensor information bound',
        assignment_caveat='the fixed matcher itself applies depth-support feasibility; excluded truth members remain failures')
    out['fixed_ray_oracle_diagnostic'] = oracle
    stage_names = sorted({k for r in matched for k in r.get('stages', {})})
    out['stage_diagnostics'] = {}
    sequence = ('initial', 'first_global_blend', 'first_refined', 'second_global_blend', 'second_refined', 'final')
    for name in stage_names:
        rows = [r['stages'][name] for r in matched if name in r.get('stages', {})]
        out['stage_diagnostics'][name] = dict(depth_error_m=distribution(r['depth_error_m'] for r in rows),
            position_error_m=distribution(r['position_error_m'] for r in rows),
            scope='offline intermediate value; global_proposal is an alternative, not the deployed point')
    out['stage_update_diagnostics'] = []
    for before, after in zip(sequence, sequence[1:]):
        pairs = [(r['stages'][before]['position_error_m'], r['stages'][after]['position_error_m'])
            for r in matched if before in r.get('stages', {}) and after in r['stages']]
        if not pairs:
            continue
        a, b = np.asarray(pairs).T
        out['stage_update_diagnostics'].append(dict(before=before, after=after, count=len(pairs),
            improved_ratio=float(np.mean(b < a-1e-6)), worsened_ratio=float(np.mean(b > a+1e-6)),
            unchanged_ratio=float(np.mean(np.abs(b-a) <= 1e-6))))
    # Round-level effects are clearer than blend/refinement substeps alone.
    for before, after in (('initial','first_refined'), ('first_refined','final')):
        pairs = [(r['stages'][before]['position_error_m'], r['stages'][after]['position_error_m'])
            for r in matched if before in r.get('stages', {}) and after in r['stages']]
        if pairs:
            a,b=np.asarray(pairs).T
            out['stage_update_diagnostics'].append(dict(before=before,after=after,count=len(pairs),
                improved_ratio=float(np.mean(b < a-1e-6)),worsened_ratio=float(np.mean(b > a+1e-6)),
                unchanged_ratio=float(np.mean(np.abs(b-a)<=1e-6))))
    return out


def evaluate_samples(model, samples, manifest, config, output_dir=None, *,
                     radar_required=True, capture_stages=True, progress_callback=None,
                     run_info=None, observation_gate=None, allow_untrained=False):
    by_key = {(s['episode_id'], int(s['frame_idx'])):s for s in samples}
    keys = [(r['episode_id'], int(r['frame_idx'])) for r in manifest]
    if len(by_key) != len(samples) or len(set(keys)) != len(keys) or set(by_key)-set(keys):
        raise ValueError('Duplicate or undeclared evaluation frames')
    previous = model.training
    model.eval()
    members, frames, detections = [], [], []
    latency = []
    norms = []
    started = time.perf_counter()
    if next(model.parameters(), torch.tensor(0.)).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for index, (item, key) in enumerate(zip(manifest, keys)):
            sample = by_key.get(key)
            count = int(item.get('label_count', len(sample['positions']) if sample is not None else 0))
            frame = dict(episode_id=key[0], frame_idx=key[1], truth_count=count,
                         detection_count=int(item.get('detected_count', 0)), status='missing_features',
                         label_use='offline scoring only')
            if sample is None:
                for j in range(count):
                    members.append(dict(episode_id=key[0], frame_idx=key[1], truth_index=j,
                        matched=False, status='missing_features'))
                frames.append(frame)
                continue
            f, truth = sample['features'], np.asarray(sample['positions'], float)
            if count != len(truth):
                raise ValueError('Declared truth count differs from evaluation labels')
            origins = np.asarray(f['ray_origins'], float)
            directions = np.asarray(f['ray_directions'], float)
            allowed = np.asarray(f['ray_valid'], bool)
            ray_norm = np.linalg.norm(directions[allowed], axis=-1)
            norms.extend(ray_norm.tolist())
            if not np.allclose(ray_norm, 1., rtol=1e-6, atol=1e-6):
                raise ValueError('Evaluation requires the actual calibrated unit rays')
            correspondence = freeze_correspondence(f, truth, config)
            rows, cols = correspondence['rows'], correspondence['cols']
            by_truth = {int(c):int(r) for r,c in zip(rows,cols)}
            by_detection = {r:c for c,r in by_truth.items()}
            has_radar = radar_required and 'ra_power' in f and 're_power' in f
            gate = observation_gate or (observation_rejection_reason if has_radar else _geometric_gate)
            reason = gate(f, config)
            scene, stages = None, {}
            if reason is None:
                cuda = next(model.parameters(), torch.tensor(0.)).is_cuda
                if cuda:
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                scene, stages = predict_stages(model, f, capture=capture_stages,allow_untrained=allow_untrained)
                if cuda:
                    torch.cuda.synchronize()
                duration = time.perf_counter()-tick
                latency.append(duration)
                reason = prediction_rejection(f, scene)
                if reason is not None:
                    scene = None
            frame.update(detection_count=len(origins), matched_count=len(rows),
                status=reason or 'valid', correspondence_sha256=correspondence['correspondence_sha256'],
                input_feature_digest=item.get('depth_feature_digest', item.get('feature_digest')),
                feature_cache=item.get('cache'), ray_origin_base_m=origins.tolist(),
                ray_direction_base=directions.tolist(), ray_valid=allowed.tolist(),
                ray_depth_bounds_m=np.asarray(f['ray_depth_bounds']).tolist(),
                bbox_xyxy=serializable(f.get('bbox_xyxy')), observer_position_base_m=serializable(f.get('observer_position')),
                stage_depths_m={k:v.tolist() for k,v in stages.items()},
                predicted_positions_base_m=scene['member_positions'] if scene is not None else None,
                prediction_valid_mask=scene['valid_mask'] if scene is not None else None,
                truth_positions_base_m=truth.tolist())
            for j, point in enumerate(truth):
                member = dict(episode_id=key[0],frame_idx=key[1],truth_index=j,
                    truth_position_base_m=point.tolist(),matched=j in by_truth,
                    status='unmatched_truth',error_3d_m=None)
                if j in by_truth:
                    row = by_truth[j]
                    member['detection_index'] = row
                    oracle = ray_truth_geometry(origins[row], directions[row], point,
                                                np.asarray(f['ray_depth_bounds'])[row])
                    unit = directions[row]/np.linalg.norm(directions[row])
                    rho = oracle['projection_depth_m']
                    member['oracle'] = dict(projection_depth_m=rho,
                        true_camera_slant_m=oracle['true_camera_slant_distance_m'],
                        projection_inside_support=oracle['projection_inside_support'],
                        supported_depth_m=oracle['supported_optimal_depth_m'],
                        line_error_m=oracle['line_perpendicular_floor_m'],
                        support_error_m=oracle['supported_ray_floor_m'],
                        projection_position_base_m=(origins[row]+rho*unit).tolist(),
                        supported_position_base_m=(origins[row]+oracle['supported_optimal_depth_m']*unit).tolist())
                    member['stages'] = {}
                    for name, depth in stages.items():
                        if len(depth) != len(origins) or not np.isfinite(depth[row]):
                            continue
                        stage_point = origins[row]+depth[row]*directions[row]
                        member['stages'][name] = dict(depth_m=float(depth[row]),
                            depth_error_m=float(depth[row]-rho),
                            position_error_m=float(np.linalg.norm(stage_point-point)))
                    valid = (scene is not None and bool(scene['valid_mask'][row]))
                    member['status'] = 'localized' if valid else reason or 'invalid_prediction_mask'
                    if valid:
                        predicted = np.asarray(scene['member_positions'][row],float)
                        error = predicted-point
                        parallel = float(error@unit)
                        pr,pa,pe = _spherical(predicted)
                        tr,ta,te = _spherical(point)
                        member.update(predicted_position_base_m=predicted.tolist(),
                            predicted_depth_m=float(scene['depths'][row]),
                            error_3d_m=float(np.linalg.norm(error)),parallel_error_m=parallel,
                            perpendicular_error_m=float(np.linalg.norm(error-parallel*unit)),
                            depth_error_to_projection_m=float(scene['depths'][row]-rho),
                            base_range_error_m=pr-tr,
                            base_azimuth_error_deg=float(np.degrees(_wrapped_angle(pa-ta))),
                            base_elevation_error_deg=float(np.degrees(pe-te)))
                members.append(member)
            for j in range(len(origins)):
                detections.append(dict(episode_id=key[0],frame_idx=key[1],detection_index=j,
                    matched_truth_index=by_detection.get(j),observed_ray_valid=bool(allowed[j]),
                    predicted_valid=bool(scene is not None and scene['valid_mask'][j]),
                    predicted_depth_m=float(scene['depths'][j]) if scene is not None else None,
                    predicted_position_base_m=scene['member_positions'][j] if scene is not None else None))
            frames.append(frame)
            if progress_callback is not None and (index % 250 == 0 or index+1 == len(keys)):
                progress_callback(dict(frames=index+1,total_frames=len(keys),elapsed_s=time.perf_counter()-started))
    finally:
        model.train(previous)
    summary = summarize(members, frames)
    summary['ray_unit_norm_check'] = dict(count=len(norms),min=min(norms) if norms else None,
        max=max(norms) if norms else None,max_abs_deviation=max((abs(x-1) for x in norms),default=None))
    summary['inference_seconds_per_frame'] = distribution(latency)
    summary['latency_scope'] = 'sensor features already prepared; sensor-only prediction and intermediate capture; excludes data I/O and label scoring'
    summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if next(model.parameters(), torch.tensor(0.)).is_cuda else 0
    summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if next(model.parameters(), torch.tensor(0.)).is_cuda else 0
    summary['elapsed_evaluation_s'] = time.perf_counter()-started
    episodes = {}
    for episode in dict.fromkeys(r['episode_id'] for r in frames):
        episodes[episode] = summarize([r for r in members if r['episode_id']==episode],
                                      [r for r in frames if r['episode_id']==episode])
    result = dict(summary=summary,episodes=episodes,frames=frames,members=members,detections=detections,
                  run_info=run_info or {})
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError('Existing evaluation output is protected: '+str(out))
        for name in ('frames','members','detections'):
            with (out/(name+'.jsonl')).open('w') as stream:
                for row in result[name]:
                    stream.write(json.dumps(serializable(row),ensure_ascii=False,allow_nan=False)+'\n')
        for name,value in (('summary',summary),('episodes',episodes),('run_info',run_info or {})):
            (out/(name+'.json')).write_text(json.dumps(serializable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def evaluate(model, samples, manifest, config, allow_untrained=False):
    """Trainer-compatible view of the same member-level scoring kernel."""
    result = evaluate_samples(model,samples,manifest,config,capture_stages=False,
        radar_required=config.get('precision_mode')!='visual',allow_untrained=allow_untrained)
    s = dict(result['summary'])
    s.update(truth_member_instances=s['truth_members'],
        primary_matched_member_instances=s['localized_matched_members'],
        primary_member_rmse_m=s['member_position_error_m']['rmse'],
        primary_member_mean_error_m=s['member_position_error_m']['mean'],
        primary_member_p95_m=s['member_position_error_m']['p95_abs'],
        unmatched_truth_instances=s['unmatched_truth_members'],
        model_trained=bool(getattr(model,'trained',False)),
        allow_untrained_validation=bool(allow_untrained),configured_window=1,
        primary_success_denominator='all GT members; misses, rejects and unmatched members count as failures',
        error_scope='finite localized members under frozen observed-bearing/support assignment')
    for threshold in THRESHOLDS:
        s[f'primary_member_success_ratio_{threshold}m']=s[f'success_ratio_{threshold}m']
    by_key=defaultdict(list)
    for member in result['members']:
        by_key[(member['episode_id'],member['frame_idx'])].append(member)
    frame_rows=[];centers=[]
    for frame in result['frames']:
        members=by_key[(frame['episode_id'],frame['frame_idx'])]
        valid=[r for r in members if r['status']=='localized']
        complete=(len(valid)==frame['truth_count']==frame['detection_count'] and bool(len(valid)))
        center=None
        if complete:
            center=float(np.linalg.norm(np.mean([r['predicted_position_base_m'] for r in valid],axis=0)-
                                        np.mean([r['truth_position_base_m'] for r in valid],axis=0)))
            centers.append(center)
        frame_rows.append(dict(episode_id=frame['episode_id'],frame_idx=frame['frame_idx'],
            truth_member_count=frame['truth_count'],detection_count=frame['detection_count'],
            primary_member_errors_m=[r['error_3d_m'] for r in valid],
            primary_center_error_m=center,
            primary_whole_scene_success_5m=bool(complete and all(r['error_3d_m']<=5 for r in valid)),
            observation_status=frame['status'],predicted=frame['status']=='valid',
            unmatched_truth_count=sum(not r['matched'] for r in members),
            unmatched_detection_count=frame['detection_count']-sum(r['matched'] for r in members),
            correspondence_sha256=frame.get('correspondence_sha256')))
    s.update(primary_whole_scene_success_ratio_5m=sum(r['primary_whole_scene_success_5m'] for r in frame_rows)/max(len(frame_rows),1),
        primary_center_rmse_m=float(np.sqrt(np.mean(np.asarray(centers)**2))) if centers else None,
        predicted_frames=sum(r['predicted'] for r in frame_rows),complete_center_frames=len(centers))
    return dict(summary=s,frames=frame_rows)
