"""Label-only error decomposition for the frozen visual-ray localization model.

These functions are offline evaluators.  Ground truth is never added to a
normal deployment observation or used to choose the deployed primary scene.
Correspondence is fixed by the original observed bearings before predictions
are scored; changing a predicted depth cannot select a more convenient label.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from rvwm.group_model import match_ray_labels
from rvwm.group_observation import build_spectral_features


SCHEMA = 'fixed_observed_ray_error_decomposition_v1'
THRESHOLDS_M = (1., 2., 5.)


def _points(values, name):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        array = array.reshape(0, 3)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError('Expected finite ' + name + ' [N,3]')
    return array


def _sha_array(array):
    array = np.asarray(array, dtype=np.float64)
    return hashlib.sha256(array.shape.__repr__().encode() + array.tobytes()).hexdigest()


def _threshold_name(value):
    return format(value, 'g') + 'm'


def freeze_correspondence(features, truth_positions, config=None):
    """Freeze the deployment evaluator's bearing/support-gated GT assignment.

    This takes observed geometry and labels only, never a predicted scene.
    Its truth hash and cardinalities prevent a caller reusing another frame's
    assignment accidentally.  The returned arrays are private copies.
    """
    truth = _points(truth_positions, 'truth positions')
    matched = match_ray_labels(features, truth, config)
    rows, cols = matched['rows'].copy(), matched['cols'].copy()
    identity = dict(rows=rows.tolist(), cols=cols.tolist(), truth_sha256=_sha_array(truth),
                    ray_origins_sha256=_sha_array(features['ray_origins']),
                    ray_directions_sha256=_sha_array(features['ray_directions']))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(rows=rows, cols=cols, truth_sha256=identity['truth_sha256'],
                ray_count=len(features['ray_origins']), truth_count=len(truth),
                correspondence_sha256=digest, diagnostics=dict(matched['diagnostics']),
                policy='original observed-ray bearing/support gate; independent of predicted positions')


def _checked_correspondence(features, truth, correspondence, config=None):
    if correspondence is None:
        correspondence = freeze_correspondence(features, truth, config)
    rows = np.asarray(correspondence['rows'], dtype=np.int64)
    cols = np.asarray(correspondence['cols'], dtype=np.int64)
    n = len(features['ray_origins'])
    if (rows.ndim != 1 or cols.shape != rows.shape
            or len(set(rows.tolist())) != len(rows) or len(set(cols.tolist())) != len(cols)
            or np.any(rows < 0) or np.any(rows >= n)
            or np.any(cols < 0) or np.any(cols >= len(truth))
            or correspondence.get('ray_count', n) != n
            or correspondence.get('truth_count', len(truth)) != len(truth)
            or correspondence.get('truth_sha256', _sha_array(truth)) != _sha_array(truth)):
        raise ValueError('Fixed correspondence does not belong to these rays/truth labels')
    if not np.asarray(features['ray_valid'], bool)[rows].all():
        raise ValueError('Fixed correspondence cannot enable invalid observed rays')
    return correspondence, rows, cols


def ray_truth_geometry(origin, direction, truth_position, depth_bounds):
    """Exact line, positive-ray and support-clamped point-to-ray distances.

    ``projection_depth_m`` is a signed *projection* on the calibrated bearing,
    not camera optical-Z and not radar range.  The camera-to-GT Euclidean slant
    distance is reported separately.  Bounds define a closed-interval lower
    bound (or infimum when the deployment decoder never reaches endpoints).
    """
    origin = np.asarray(origin, float);direction = np.asarray(direction, float)
    truth = np.asarray(truth_position, float);bounds = np.asarray(depth_bounds, float)
    if (origin.shape != (3,) or direction.shape != (3,) or truth.shape != (3,)
            or bounds.shape != (2,) or not np.isfinite(np.r_[origin, direction, truth, bounds]).all()
            or bounds[0] < 0 or bounds[1] <= bounds[0]
            or not np.isclose(np.linalg.norm(direction), 1., atol=1e-6, rtol=1e-6)):
        raise ValueError('Finite origin/truth, unit bearing and positive depth interval required')
    direction = direction / np.linalg.norm(direction)
    delta = truth - origin
    projected = float(delta @ direction)
    perpendicular = delta - projected * direction
    line_floor = float(np.linalg.norm(perpendicular))
    positive_depth = max(projected, 0.)
    optimal = float(np.clip(projected, bounds[0], bounds[1]))
    slant = float(np.linalg.norm(delta))
    angle = float(np.degrees(np.arccos(np.clip(projected / slant, -1., 1.)))) if slant > 0 else None
    return dict(projection_depth_m=projected, true_camera_slant_distance_m=slant,
                observed_bearing_error_deg=angle, line_perpendicular_floor_m=line_floor,
                positive_ray_optimal_depth_m=positive_depth,
                positive_ray_floor_m=float(np.linalg.norm(origin + positive_depth * direction - truth)),
                supported_optimal_depth_m=optimal,
                support_clamp_distance_m=abs(optimal - projected),
                supported_ray_floor_m=float(np.linalg.norm(origin + optimal * direction - truth)),
                projection_inside_support=bool(bounds[0] <= projected <= bounds[1]))


def decompose_frame(features, truth_positions, predicted_scene=None, *, correspondence=None,
                    config=None, exact_ray_tolerance_m=1e-3):
    """Score one already selected primary scene under the frozen assignment.

    ``predicted_scene`` accepts the normal model dictionary with
    ``member_positions`` and ``valid_mask``; None denotes a rejected/missing
    primary and contributes failures to every truth-denominator threshold.
    No oracle candidate selection is performed here.
    """
    truth = _points(truth_positions, 'truth positions')
    correspondence, rows, cols = _checked_correspondence(features, truth, correspondence, config)
    origins = _points(features['ray_origins'], 'ray origins')
    directions = _points(features['ray_directions'], 'ray directions')
    n = len(origins);allowed = np.asarray(features['ray_valid'], bool)
    bounds = np.asarray(features['ray_depth_bounds'], float)
    if directions.shape != origins.shape or allowed.shape != (n,) or bounds.shape != (n, 2):
        raise ValueError('Observed ray geometry shapes differ')
    if not np.isfinite(exact_ray_tolerance_m) or exact_ray_tolerance_m <= 0:
        raise ValueError('Positive finite exact-ray tolerance required')
    positions = np.zeros((n, 3), float);predicted_valid = np.zeros(n, bool)
    if predicted_scene is not None:
        positions = np.asarray(predicted_scene['member_positions'], float)
        predicted_valid = np.asarray(predicted_scene.get('valid_mask', allowed))
        if positions.shape != (n, 3) or predicted_valid.shape != (n,) or predicted_valid.dtype != np.bool_:
            raise ValueError('Primary scene cardinality or valid-mask type differs from observed rays')
    members = []
    for detection, label in zip(rows.tolist(), cols.tolist()):
        geometry = ray_truth_geometry(origins[detection], directions[detection], truth[label], bounds[detection])
        member = dict(detection_index=detection, truth_index=label, **geometry,
                      prediction_usable=False, error_3d_m=None, parallel_error_signed_m=None,
                      parallel_error_abs_m=None, perpendicular_error_m=None,
                      predicted_camera_slant_depth_m=None, depth_error_to_projection_m=None,
                      depth_error_to_supported_oracle_m=None, predicted_off_ray_residual_m=None,
                      pythagorean_squared_residual_m2=None, exact_visual_ray=None)
        if predicted_valid[detection] and np.isfinite(positions[detection]).all():
            direction = directions[detection] / np.linalg.norm(directions[detection])
            point = positions[detection];error = point - truth[label]
            parallel = float(error @ direction)
            transverse = float(np.linalg.norm(error - parallel * direction))
            depth = float((point - origins[detection]) @ direction)
            residual = float(np.linalg.norm(point - (origins[detection] + depth * direction)))
            squared = float(error @ error)
            member.update(prediction_usable=True, error_3d_m=float(np.sqrt(squared)),
                parallel_error_signed_m=parallel, parallel_error_abs_m=abs(parallel),
                perpendicular_error_m=transverse, predicted_camera_slant_depth_m=depth,
                depth_error_to_projection_m=depth - geometry['projection_depth_m'],
                depth_error_to_supported_oracle_m=depth - geometry['supported_optimal_depth_m'],
                predicted_off_ray_residual_m=residual,
                pythagorean_squared_residual_m2=squared - parallel**2 - transverse**2,
                exact_visual_ray=residual <= exact_ray_tolerance_m)
        members.append(member)
    usable = [m for m in members if m['prediction_usable']]
    complete = len(rows) == len(truth) == n and len(usable) == len(truth) and bool(len(truth))
    result = dict(schema=SCHEMA, episode_id=features.get('metadata', {}).get('episode_id'),
        frame_idx=features.get('metadata', {}).get('frame_idx'),
        correspondence_sha256=correspondence.get('correspondence_sha256'),
        correspondence_policy=correspondence.get('policy', 'explicit fixed label-only assignment'),
        truth_count=len(truth), detection_count=n, valid_observed_ray_count=int(allowed.sum()),
        matched_count=len(rows), localized_matched_count=len(usable),
        unmatched_truth_count=len(truth)-len(rows), unmatched_valid_detection_count=int(allowed.sum())-len(rows),
        extra_or_unmatched_detection_count=n-len(rows), invalid_observed_ray_count=int((~allowed).sum()),
        invalid_ray_enabled_by_prediction_count=int((predicted_valid & ~allowed).sum()),
        unlocalized_matched_count=len(rows)-len(usable),
        published_valid_prediction_count=int((predicted_valid & np.isfinite(positions).all(1)).sum()),
        complete_cardinality_and_localization=complete, members=members,
        label_use='offline evaluation only; never deployment conditions or primary scene selection',
        floor_scope='matched labels under fixed observed-ray correspondence; not an information-theoretic sensor bound')
    for threshold in THRESHOLDS_M:
        tag = _threshold_name(threshold)
        result['success_count_'+tag] = sum(m['error_3d_m'] <= threshold for m in usable)
        result['oracle_support_floor_success_count_'+tag] = sum(m['supported_ray_floor_m'] <= threshold for m in members)
        result['whole_scene_success_'+tag] = bool(complete and all(m['error_3d_m'] <= threshold for m in usable))
    return result


def _distribution(values):
    array = np.asarray(list(values), float)
    if not len(array):return dict(count=0,mean=None,mae=None,rmse=None,p50_abs=None,p95_abs=None,max_abs=None)
    absolute = np.abs(array)
    return dict(count=len(array),mean=float(array.mean()),mae=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(array**2))),p50_abs=float(np.quantile(absolute,.5)),
                p95_abs=float(np.quantile(absolute,.95)),max_abs=float(absolute.max()))


def aggregate_frames(frames: Sequence[Mapping]):
    """Aggregate only matched finite errors; keep all GT in coverage denominators."""
    frames = list(frames)
    if any(row.get('schema') != SCHEMA for row in frames):raise ValueError('Unexpected decomposition frame schema')
    members = [m for row in frames for m in row['members']]
    usable = [m for m in members if m['prediction_usable']]
    truth_count = sum(row['truth_count'] for row in frames)
    error = _distribution(m['error_3d_m'] for m in usable)
    parallel = _distribution(m['parallel_error_signed_m'] for m in usable)
    perpendicular = _distribution(m['perpendicular_error_m'] for m in usable)
    pythagorean = _distribution(m['pythagorean_squared_residual_m2'] for m in usable)
    total_mse = error['rmse']**2 if error['rmse'] is not None else None
    result = dict(schema=SCHEMA,frames=len(frames),truth_member_instances=truth_count,
        detection_instances=sum(r['detection_count'] for r in frames),
        matched_member_instances=len(members),localized_matched_member_instances=len(usable),
        unmatched_truth_instances=sum(r['unmatched_truth_count'] for r in frames),
        unmatched_valid_detection_instances=sum(r['unmatched_valid_detection_count'] for r in frames),
        extra_or_unmatched_detection_instances=sum(r['extra_or_unmatched_detection_count'] for r in frames),
        invalid_observed_ray_instances=sum(r['invalid_observed_ray_count'] for r in frames),
        unlocalized_matched_instances=len(members)-len(usable),
        member_error=error,parallel_error=parallel,perpendicular_error=perpendicular,
        supported_ray_floor=_distribution(m['supported_ray_floor_m'] for m in members),
        line_perpendicular_floor=_distribution(m['line_perpendicular_floor_m'] for m in members),
        predicted_off_ray_residual=_distribution(m['predicted_off_ray_residual_m'] for m in usable),
        pythagorean_squared_residual=pythagorean,
        parallel_mse_fraction=parallel['rmse']**2/total_mse if total_mse else None,
        perpendicular_mse_fraction=perpendicular['rmse']**2/total_mse if total_mse else None,
        matched_predictions_off_exact_ray=sum(not m['exact_visual_ray'] for m in usable),
        coverage_denominator='all truth members, including unmatched or unlocalized members',
        rmse_scope='finite localized members under the fixed observed-ray assignment',
        floor_scope='label-assisted lower bound within the fixed bearing assignment and analytic support')
    for threshold in THRESHOLDS_M:
        tag = _threshold_name(threshold)
        successful = sum(r['success_count_'+tag] for r in frames)
        floor_success = sum(r['oracle_support_floor_success_count_'+tag] for r in frames)
        result['success_count_'+tag] = successful
        result['success_ratio_'+tag] = successful/truth_count if truth_count else None
        result['oracle_support_floor_coverage_ratio_'+tag] = floor_success/truth_count if truth_count else None
        result['whole_scene_success_ratio_'+tag] = sum(r['whole_scene_success_'+tag] for r in frames)/len(frames) if frames else None
    return result


def oracle_bearing_features(features, truth_positions, *, correspondence=None, config=None):
    """GT-bearing intervention for diagnosis only; never a deployable ablation.

    Detection slots, original label assignment, spectra and observer navigation
    are retained.  Only matched ray directions change, and public analytic
    depth supports are rebuilt rather than copied from the original bearing.
    Unmatched and invalid slots remain unchanged and cannot gain GT labels.
    """
    truth = _points(truth_positions, 'truth positions')
    correspondence, rows, cols = _checked_correspondence(features, truth, correspondence, config)
    directions = np.asarray(features['ray_directions'], float).copy()
    origins = np.asarray(features['ray_origins'], float)
    delta = truth[cols] - origins[rows]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 0):raise ValueError('A GT point at the camera cannot define an oracle bearing')
    directions[rows] = delta/distance[:,None]
    metadata = dict(features.get('metadata', {}))
    metadata['diagnostic_intervention'] = dict(kind='oracle_true_bearing',labels_used=True,evaluation_only=True,
        original_correspondence_sha256=correspondence.get('correspondence_sha256'),
        truth_depth_supplied_to_model=False)
    fields = ('observer_position','ra_power','re_power','range_m','azimuth_rad','elevation_rad',
              'range_scale_m','ray_valid','ra_valid','re_valid')
    return build_spectral_features(ray_origins=origins,ray_directions=directions,
        metadata=metadata,**{key:features[key] for key in fields})
