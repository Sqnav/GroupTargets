"""Post-inference group geometry evaluation and predeclared episode protocols.

This module is never called by the observation encoder or the localizer.
Member labels enter only through explicit evaluator/supervision arguments.
Spatial second moments describe the group, not uncertainty in its centre.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .coordinates import FixedStationFrame


FORMATION_TYPES = ('line', 'v', 'ring', 'grid', 'staggered_3d',
                   'cross', 'diamond', 'double_column', 'pyramid', 'helix')


def _spherical(point):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('A finite three-dimensional group centre is required')
    distance = float(np.linalg.norm(point))
    if distance <= 0:
        raise ValueError('Spherical group centre cannot be at the station origin')
    return np.array([distance, np.degrees(np.arctan2(point[1], point[0])),
                     np.degrees(np.arctan2(point[2], np.hypot(point[0], point[1])))])


def _wrapped_difference_degrees(a, b):
    return float((float(a)-float(b)+180.) % 360. - 180.)


def _statistics(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(count=0, mean=None, rmse=None, p50=None, p90=None, p95=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite group metric')
    return dict(count=len(values), mean=float(values.mean()), rmse=float(np.sqrt(np.mean(values**2))),
                p50=float(np.quantile(values, .5)), p90=float(np.quantile(values, .9)),
                p95=float(np.quantile(values, .95)), max=float(values.max()))


def shape_statistics(positions_base, *, relative_eigengap=1e-5):
    """Order-free extent and spatial second moment in the fixed base axes.

    ``extent_m`` is max-minus-min along base x/y/z. Principal RMS spreads are
    sqrt(eigenvalues) of 1/N sum((p-c)(p-c)^T), not bounding-box half lengths.
    Eigenvectors are columns, with descending spread; their sign is arbitrary.
    Repeated eigenvalues do not define a unique full orientation, so the full
    axes are null in that case. A unique leading axis may still be available.
    """
    points = np.asarray(positions_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite member positions [N,3]')
    if not len(points):
        return dict(extent_m=None, spatial_second_moment_base=None,
                    principal_rms_spread_m=None, principal_axes_base=None,
                    primary_axis_base=None, rms_radius_m=None,
                    orientation_status='no_members', is_localization_covariance=False)
    centred = points-points.mean(axis=0)
    second = centred.T @ centred / len(points)
    values, vectors = np.linalg.eigh((second+second.T)*.5)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0.), vectors[:, order]
    # Deterministic signs are a serialization choice, not an oriented heading.
    for i in range(3):
        if vectors[np.argmax(np.abs(vectors[:, i])), i] < 0:
            vectors[:, i] *= -1
    tolerance = relative_eigengap * max(float(values[0]), 1e-12)
    unique = bool(values[0] > 1e-12 and np.all(values[:-1]-values[1:] > tolerance))
    leading = bool(values[0] > 1e-12 and values[0]-values[1] > tolerance)
    return dict(extent_m=np.ptp(points, axis=0).tolist(),
                spatial_second_moment_base=second.tolist(),
                principal_rms_spread_m=np.sqrt(values).tolist(),
                principal_axes_base=vectors.tolist() if unique else None,
                primary_axis_base=vectors[:, 0].tolist() if leading else None,
                rms_radius_m=float(np.sqrt(values.sum())),
                orientation_status='axes_up_to_sign' if unique else
                    'leading_axis_only_up_to_sign' if leading else 'isotropic_or_repeated_eigenvalues',
                is_localization_covariance=False)


def training_formation(dataset, episode, allowed_episodes=None):
    """Explicit label helper for training/evaluation, never normal inference.

    Supplying allowed_episodes makes an accidental holdout-label request fail.
    The normal observation object never contains this metadata field.
    """
    if allowed_episodes is not None and episode not in set(allowed_episodes):
        raise ValueError('Formation supervision requested outside allowed episodes')
    if episode not in dataset.episodes:
        raise FileNotFoundError(episode)
    metadata = json.loads((Path(dataset.root)/'data'/episode/'metadata.json').read_text())
    label = metadata.get('group_structure', metadata.get('scenario_spec', {}).get('formation_type'))
    if label not in FORMATION_TYPES:
        raise ValueError('Unknown or missing formation supervision: '+str(label))
    return label


def _estimated_centre(group):
    value = group.get('center_base_m')
    spherical_keys = ('range_m', 'azimuth_deg', 'elevation_deg')
    provided = [group.get(key) for key in spherical_keys]
    if value is None and all(v is None for v in provided):
        return None
    if value is not None:
        centre = np.asarray(value, float)
        spherical = _spherical(centre)
        if any(v is not None for v in provided):
            if not all(v is not None for v in provided) or not np.isfinite(provided).all():
                raise ValueError('Incomplete spherical group centre')
            residual = np.asarray(provided, float)-spherical
            residual[1] = _wrapped_difference_degrees(provided[1], spherical[1])
            if not np.allclose(residual, 0., atol=1e-5, rtol=1e-7):
                raise ValueError('Cartesian and spherical group centres disagree')
        return centre
    if not all(v is not None for v in provided) or not np.isfinite(provided).all() or provided[0] <= 0:
        raise ValueError('Incomplete or invalid spherical group centre')
    radius, azimuth, elevation = provided
    azimuth, elevation = np.radians([azimuth, elevation])
    return radius*np.array([np.cos(elevation)*np.cos(azimuth),
                            np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])


def _detected_set_correspondence(result, observation, positions, angle_gate_deg):
    """Evaluation-only visual correspondence, independent of predicted depths.

    A direct spectral result without pixel observations cannot identify which
    labelled subset its centre represents. It is deliberately left unscored;
    nearest predicted XYZ must not choose a favourable subset after inference.
    """
    from .geometry import rays
    from .group_observation import _ray_intervals
    from .group_model import match_ray_labels

    detections = result.get('targets', [])
    count = int(result.get('detected_count', len(detections)))
    info = dict(method='evaluation_only_gated_visual_ray_assignment',
                uses_predicted_positions_for_matching=False,
                angle_gate_deg=float(angle_gate_deg), gate_calibrated=False,
                detected_count=count, matched_count=0,
                unmatched_detection_count=count, unmatched_label_count=len(positions),
                complete_detected_set=False, complete_full_group=False,
                status='correspondence_unavailable', matches=[])
    empty = np.empty(0, dtype=np.int64)
    if count != len(detections):
        info['status'] = 'detection_cardinality_mismatch'
        return empty, empty, info
    if not count:
        info['status'] = 'no_detections'
        return empty, empty, info
    if observation.calibration is not None and np.any(np.asarray(observation.calibration.distortion) != 0):
        info['status'] = 'distorted_pixels_require_explicit_undistortion'
        return empty, empty, info
    pixels = np.zeros((count, 2), dtype=float)
    available = np.zeros(count, dtype=bool)
    for i, detection in enumerate(detections):
        point = detection.get('center_px')
        if point is not None:
            point = np.asarray(point, dtype=float)
            if point.shape == (2,) and np.isfinite(point).all():
                pixels[i] = point
                available[i] = True
    if not available.any():
        info['status'] = 'pixel_correspondence_unavailable'
        return empty, empty, info
    base = FixedStationFrame.from_observation(observation).observations(observation)
    origins = np.broadcast_to(base.camera.position, (count, 3)).copy()
    directions = rays(pixels, base.camera)
    bounds, valid = _ray_intervals(origins, directions, available, base.radar.range_limit)
    matched = match_ray_labels(dict(ray_origins=origins, ray_directions=directions,
        ray_valid=valid, ray_depth_bounds=bounds, range_scale_m=base.radar.range_limit),
        positions, dict(group_supervision_max_angle_deg=angle_gate_deg))
    rows, cols = matched['rows'], matched['cols']
    info.update(matched_count=len(rows), unmatched_detection_count=count-len(rows),
                unmatched_label_count=len(positions)-len(cols),
                complete_detected_set=len(rows) == count,
                complete_full_group=len(rows) == count == len(positions),
                status='matched_detected_set' if len(rows) == count else 'unmatched_visual_detections',
                missing_pixel_count=int((~available).sum()),
                matches=[dict(detection_index=int(a), label_index=int(b)) for a, b in zip(rows, cols)])
    # Centre scope includes every current detection, so partial matching cannot
    # identify the truth subset for that centre. Shape may have a smaller,
    # explicitly localized scope and is matched separately below.
    centre_indices = cols if info['complete_detected_set'] else empty
    localized = []
    for i, detection in enumerate(detections):
        point = detection.get('position_base_m')
        if point is not None:
            point = np.asarray(point, float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Finite localized member XYZ required for scoped shape evaluation')
            localized.append(i)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    shape_indices = (np.array([mapping[i] for i in localized], dtype=np.int64)
                     if all(i in mapping for i in localized) else empty)
    info['localized_detection_count'] = len(localized)
    info['complete_localized_set'] = bool(localized) and len(shape_indices) == len(localized)
    return centre_indices, shape_indices, info


class GroupEvaluator:
    """Every truth-containing frame stays in centre-success denominators.

    Only the selected primary solution is scored as normal performance. The
    best alternative error is explicitly an oracle diagnostic, never used to
    replace that selected estimate. New detected-set outputs use evaluation-only
    visual matching; legacy whole-group outputs retain their original scope.
    """
    def __init__(self, success_gate_m=5., matching_angle_gate_deg=3.):
        if not np.isfinite(success_gate_m) or success_gate_m <= 0:
            raise ValueError('Positive finite group success gate required')
        self.success_gate_m = float(success_gate_m)
        if not np.isfinite(matching_angle_gate_deg) or not 0 < matching_angle_gate_deg < 90:
            raise ValueError('Positive finite visual matching angle gate required')
        self.matching_angle_gate_deg = float(matching_angle_gate_deg)
        self.frames = []

    def update(self, result, targets, observation, formation_label=None):
        if result.get('coordinate_frame', 'base_local') != 'base_local' or result.get('angle_unit', 'deg') != 'deg':
            raise ValueError('Group evaluator requires base_local coordinates and degree angles')
        if result.get('episode_id') != observation.episode_id or int(result.get('frame_idx', -1)) != observation.frame_idx:
            raise ValueError('Prediction and observation episode/frame differ')
        if formation_label is not None and formation_label not in FORMATION_TYPES:
            raise ValueError('Unknown truth formation label')
        frame = FixedStationFrame.from_observation(observation)
        positions = frame.to_local_positions(targets.positions)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError('Expected finite complete group supervision [N,3]')
        truth_present = bool(len(positions))
        group = result.get('group', {})
        scoped = (result.get('schema') == 'ra_re_member_localization_v1' or
                  group.get('scope') == 'current_visual_detection_set')
        if scoped:
            angle_gate = float(result.get('solver_diagnostics', {}).get(
                'evaluation_angle_gate_deg', self.matching_angle_gate_deg))
            if not np.isfinite(angle_gate) or not 0 < angle_gate < 90:
                raise ValueError('Positive finite saved visual matching angle gate required')
            centre_indices, shape_indices, correspondence = _detected_set_correspondence(
                result, observation, positions, angle_gate)
            centre_positions, shape_positions = positions[centre_indices], positions[shape_indices]
            complete_group = correspondence['complete_full_group']
        else:
            centre_positions = shape_positions = positions
            correspondence = None
            complete_group = True
        truth_centre = centre_positions.mean(axis=0) if len(centre_positions) else None
        centre = _estimated_centre(group)
        localized = centre is not None
        structure = result.get('structure', {})
        predicted_type = structure.get('type')
        if predicted_type not in FORMATION_TYPES:
            predicted_type = 'unknown'
        truth_shape = shape_statistics(shape_positions)
        row = dict(episode_id=observation.episode_id, frame_idx=observation.frame_idx,
            timestamp_s=float(observation.timestamp), truth_member_count=len(positions),
            truth_group_present=truth_present, detected_count=int(result.get('detected_count', 0)),
            localized=localized, localization_status=group.get('localization_status', 'unspecified'),
            centre_error_m=None, position_error_m=None, range_error_m=None,
            azimuth_wrapped_error_deg=None, elevation_error_deg=None,
            successful_5m=False, covariance_covered_95=None,
            covariance_status='absent', truth_formation=formation_label,
            predicted_formation=predicted_type,
            formation_correct=None if formation_label is None else predicted_type == formation_label,
            extent_error_l2_m=None, spatial_second_moment_error_fro_m2=None,
            primary_axis_error_deg=None, oracle_best_alternative_centre_error_m=None,
            alternatives_count=len(result.get('alternatives', group.get('alternatives', []))),
            seconds=result.get('solver_diagnostics', {}).get('elapsed_s'),
            uncertainty_calibrated=False)
        row.update(centre_metric_scope='current_visual_detection_set' if scoped else 'full_label_group',
            structure_metric_scope='current_localized_detection_set' if scoped else 'full_label_group',
            visual_correspondence=correspondence,
            complete_full_group_correspondence=bool(complete_group),
            full_group_localized=bool(localized and complete_group),
            full_group_center_error_m=None, full_group_successful_5m=False,
            detected_subset_center_error_m=None, detected_subset_successful_5m=None,
            centre_comparable=truth_centre is not None,
            shape_comparable=bool(len(shape_positions)))
        if scoped and not complete_group:
            row['formation_correct'] = None
        if truth_centre is not None and localized:
            truth_spherical, estimated_spherical = _spherical(truth_centre), _spherical(centre)
            error = estimated_spherical-truth_spherical
            error[1] = _wrapped_difference_degrees(estimated_spherical[1], truth_spherical[1])
            distance = float(np.linalg.norm(centre-truth_centre))
            row.update(centre_error_m=distance, position_error_m=distance,
                range_error_m=abs(float(error[0])), azimuth_wrapped_error_deg=abs(float(error[1])),
                elevation_error_deg=abs(float(error[2])),
                successful_5m=bool(complete_group and distance <= self.success_gate_m),
                full_group_center_error_m=distance if complete_group else None,
                full_group_successful_5m=bool(complete_group and distance <= self.success_gate_m),
                detected_subset_center_error_m=distance if scoped else None,
                detected_subset_successful_5m=bool(distance <= self.success_gate_m) if scoped else None)
            covariance = group.get('covariance_r_az_el')
            if covariance is not None:
                covariance = np.asarray(covariance, float)
                if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                        or not np.allclose(covariance, covariance.T, atol=1e-8, rtol=1e-7)):
                    row['covariance_status'] = 'invalid'
                elif np.linalg.eigvalsh(covariance).min() <= 0:
                    row['covariance_status'] = 'singular_or_nonpositive'
                else:
                    row['covariance_status'] = 'finite_positive_definite_uncalibrated'
                    row['covariance_covered_95'] = bool(error @ np.linalg.solve(covariance, error) <= 7.814727903)
        if len(shape_positions):
            extent = structure.get('extent_m')
            if extent is not None:
                extent = np.asarray(extent, float)
                if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent < 0):
                    raise ValueError('Expected nonnegative finite base-axis extent [3]')
                row['extent_error_l2_m'] = float(np.linalg.norm(extent-np.asarray(truth_shape['extent_m'])))
            second = structure.get('spatial_second_moment_base')
            if second is not None:
                second = np.asarray(second, float)
                if second.shape != (3, 3) or not np.isfinite(second).all():
                    raise ValueError('Expected finite spatial second moment [3,3]')
                row['spatial_second_moment_error_fro_m2'] = float(np.linalg.norm(second-np.asarray(truth_shape['spatial_second_moment_base'])))
            truth_axis = truth_shape['primary_axis_base']
            predicted_axis = structure.get('primary_axis_base')
            if predicted_axis is None and structure.get('principal_axes_base') is not None:
                predicted_axis = np.asarray(structure['principal_axes_base'], float)[:, 0]
            if truth_axis is not None and predicted_axis is not None:
                axis = np.asarray(predicted_axis, float)
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                    raise ValueError('Expected finite nonzero primary axis [3]')
                cosine = abs(float(axis @ np.asarray(truth_axis))/np.linalg.norm(axis))
                row['primary_axis_error_deg'] = float(np.degrees(np.arccos(np.clip(cosine, 0., 1.))))
        if truth_centre is not None:
            alternative_errors = []
            for alternative in result.get('alternatives', group.get('alternatives', [])):
                candidate = _estimated_centre(alternative.get('group', alternative))
                if candidate is not None:
                    alternative_errors.append(float(np.linalg.norm(candidate-truth_centre)))
            if alternative_errors:
                row['oracle_best_alternative_centre_error_m'] = min(alternative_errors)
        self.frames.append(row)
        return row

    def summary(self):
        rows = self.frames
        present = [r for r in rows if r['truth_group_present']]
        localized = [r for r in present if r.get('full_group_localized', r['localized'])]
        labelled = [r for r in rows if r['truth_formation'] is not None and r['formation_correct'] is not None]
        scoped = [r for r in rows if r.get('centre_metric_scope') == 'current_visual_detection_set']
        detected = [r for r in scoped if r['detected_count'] > 0]
        covered = [r['covariance_covered_95'] for r in localized if r['covariance_covered_95'] is not None]
        metrics = {key: _statistics([r[key] for r in present]) for key in
            ('centre_error_m', 'position_error_m', 'range_error_m', 'azimuth_wrapped_error_deg',
             'elevation_error_deg', 'extent_error_l2_m', 'spatial_second_moment_error_fro_m2', 'primary_axis_error_deg')}
        successful = sum(r['successful_5m'] for r in present)
        return dict(frames=len(rows), truth_group_frames=len(present), localized_group_frames=len(localized),
            unlocalized_group_frames=len(present)-len(localized), successful_group_frames=successful,
            group_localization_success_ratio=successful/len(present) if present else None,
            localization_success_ratio=successful/len(present) if present else None,
            localized_group_ratio=len(localized)/len(present) if present else None,
            success_gate_m=self.success_gate_m, **metrics,
            formation_labelled_frames=len(labelled), formation_correct_frames=sum(r['formation_correct'] for r in labelled),
            formation_unknown_frames=sum(r['predicted_formation'] == 'unknown' for r in labelled),
            formation_accuracy=sum(r['formation_correct'] for r in labelled)/len(labelled) if labelled else None,
            formation_confusion={truth: dict(Counter(r['predicted_formation'] for r in labelled if r['truth_formation'] == truth))
                                 for truth in FORMATION_TYPES},
            localization_status_counts=dict(Counter(r['localization_status'] for r in rows)),
            covariance_covered_instances=sum(covered), covariance_coverage_denominator=len(covered),
            uncertainty_coverage_95=float(np.mean(covered)) if covered else None,
            covariance_status_counts=dict(Counter(r['covariance_status'] for r in rows)),
            mean_frame_s=_statistics([r['seconds'] for r in rows])['mean'],
            p95_frame_s=_statistics([r['seconds'] for r in rows])['p95'],
            alternatives_frames=sum(r['alternatives_count'] > 0 for r in rows),
            uncertainty_calibrated=False,
            centre_definition=('per-frame declared output scope: visually matched detection subset for RA/RE; '
                               'ALL labelled members for legacy whole-group outputs; observer excluded') if scoped else
                              'arithmetic mean of ALL labelled member centres; observer excluded',
            centre_metric_scope_counts=dict(Counter(r.get('centre_metric_scope', 'full_label_group') for r in rows)),
            full_group_center_error_m=_statistics([r.get('full_group_center_error_m', r['centre_error_m']) for r in present]),
            full_group_metric_scope='complete visual correspondence required for detected-set outputs; all truth frames stay in success denominator',
            detected_subset_center_error_m=_statistics([r.get('detected_subset_center_error_m') for r in scoped]),
            detected_subset_frames=len(detected),
            detected_subset_comparable_frames=sum(r.get('centre_comparable', False) for r in detected),
            detected_subset_localized_frames=sum(r['localized'] for r in detected),
            detected_subset_success_ratio=(sum(r.get('detected_subset_successful_5m') is True for r in detected)/len(detected)
                                           if detected else None),
            detected_subset_success_denominator='all detected-set output frames with at least one detection; unmatched/unlocalized count as failures',
            visual_correspondence_status_counts=dict(Counter(r['visual_correspondence']['status'] for r in scoped)),
            error_scope='all finite primary group estimates; no 3D error gate or oracle-alternative replacement',
            success_denominator='all truth-containing evaluated frames, including unlocalized/no-detection frames',
            structure_extent_definition='max-minus-min member coordinates along fixed base x/y/z, in metres',
            spatial_second_moment_notice='group shape, not covariance of centre localization',
            covariance_coverage_scope='finite positive-definite spherical local covariances among localized primary estimates')








class MemberEvaluator:
    """Position-set evaluation after inference; no persistent identities.

    Hungarian matching is evaluation-only and ungated for error statistics.
    The 5m success denominator includes every labelled member, including all
    missing/unlocalized detections. Centre/formation metrics remain separate.
    """
    def __init__(self, success_gate_m=5.):
        self.success_gate_m=float(success_gate_m);self.frames=[]

    def update(self,result,targets,observation):
        from scipy.optimize import linear_sum_assignment
        if result.get('coordinate_frame')!='base_local':raise ValueError('Member evaluation requires base_local')
        if (result.get('episode_id'),result.get('frame_idx'))!=(observation.episode_id,observation.frame_idx):
            raise ValueError('Member prediction/frame mismatch')
        truth=FixedStationFrame.from_observation(observation).to_local_positions(targets.positions)
        detected=result.get('targets',[])
        indices=[i for i,t in enumerate(detected) if t.get('position_base_m') is not None]
        estimates=np.array([detected[i]['position_base_m'] for i in indices],float).reshape(-1,3)
        if not np.isfinite(estimates).all():raise ValueError('Finite published member positions required')
        matches=[]
        if len(estimates) and len(truth):
            distances=np.linalg.norm(estimates[:,None]-truth[None],axis=-1)
            rr,cc=linear_sum_assignment(distances)
            for a,b in zip(rr,cc):
                pred_sphere=_spherical(estimates[a]);true_sphere=_spherical(truth[b]);error=pred_sphere-true_sphere
                error[1]=_wrapped_difference_degrees(pred_sphere[1],true_sphere[1])
                record=dict(detection_id=detected[indices[a]]['detection_id'],truth_evaluation_index=int(b),
                    position_error_m=float(distances[a,b]),range_error_m=abs(float(error[0])),
                    azimuth_wrapped_error_deg=abs(float(error[1])),elevation_error_deg=abs(float(error[2])),
                    successful=bool(distances[a,b]<=self.success_gate_m),covariance_covered_95=None)
                cov=detected[indices[a]].get('covariance_r_az_el')
                if cov is not None:
                    cov=np.asarray(cov,float)
                    if cov.shape==(3,3) and np.isfinite(cov).all() and np.allclose(cov,cov.T) and np.linalg.eigvalsh(cov).min()>0:
                        record['covariance_covered_95']=bool(error@np.linalg.solve(cov,error)<=7.814727903)
                matches.append(record)
        row=dict(episode_id=observation.episode_id,frame_idx=observation.frame_idx,truth_count=len(truth),
            detected_count=len(detected),localized_count=len(indices),matched_count=len(matches),
            successful_count=sum(r['successful'] for r in matches),unlocalized_detections=len(detected)-len(indices),
            unmatched_estimates=len(indices)-len(matches),matches=matches,
            seconds=result.get('solver_diagnostics',{}).get('elapsed_s'),
            validation_only=result.get('solver_diagnostics',{}).get('validation_only',False))
        self.frames.append(row);return row

    def summary(self):
        matches=[m for row in self.frames for m in row['matches']]
        total=sum(r['truth_count'] for r in self.frames);success=sum(r['successful_count'] for r in self.frames)
        cov=[m['covariance_covered_95'] for m in matches if m['covariance_covered_95'] is not None]
        return dict(frames=len(self.frames),truth_member_instances=total,
            detected_instances=sum(r['detected_count'] for r in self.frames),
            localized_instances=sum(r['localized_count'] for r in self.frames),
            successful_instances=success,success_gate_m=self.success_gate_m,
            success_fraction_all_truth=success/total if total else None,
            **{key:_statistics([m[key] for m in matches]) for key in
               ('position_error_m','range_error_m','azimuth_wrapped_error_deg','elevation_error_deg')},
            covariance_coverage_95=sum(cov)/len(cov) if cov else None,covariance_evaluated_instances=len(cov),
            seconds=_statistics([r['seconds'] for r in self.frames]),
            matching='post-inference ungated 3D Hungarian; all-truth success denominator',
            uncertainty_calibrated=False)
