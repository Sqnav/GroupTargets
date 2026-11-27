"""Explicit TRAIN-only calibration of visual gates and classical filters.

This module accepts exported numeric records, not a Dataset or an inference
object. The caller matches normal predictions to separately loaded TRAIN
labels offline. No label-bearing record is returned to the normal pipeline.
Whole TRAIN episodes are assigned deterministically to fit/calibrate/audit;
audit episodes only produce diagnostics and never change the fitted values.

Running these functions is calibration work, not test-set evaluation. Merely
importing this module never reads files, data, checkpoints, or model weights.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import re

import numpy as np
from scipy.stats import chi2

SCHEMA = "rvloc.calibration.v1"
SOURCES = ("fft", "rgb", "single_frame")


class CalibrationError(ValueError):
    pass


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _real(value, name, nonnegative=False):
    if isinstance(value, (bool, str, np.bool_)) or not np.isscalar(value):
        raise CalibrationError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise CalibrationError(f"Invalid {name}")
    return value


def _vector(value, name):
    if np.iscomplexobj(value):
        raise CalibrationError(f"{name} must be real")
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise CalibrationError(f"{name} must be a finite [3] vector")
    return value


def partition_train_episodes(train_episode_ids: Sequence[str], seed=20261126):
    """Deterministic approximately 60/20/20 split, with at least one per part."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise CalibrationError("seed must be an integer")
    episodes = list(train_episode_ids)
    if (len(episodes) < 3 or any(not isinstance(ep, str) or not ep for ep in episodes)
            or len(set(episodes)) != len(episodes)):
        raise CalibrationError("At least three unique TRAIN episode IDs are required")
    ordered = sorted(episodes, key=lambda ep: (hashlib.sha256(f"{int(seed)}:{ep}".encode()).hexdigest(), ep))
    held_out = max(1, int(np.floor(len(ordered) * .2)))
    fit_end = len(ordered) - 2 * held_out
    return {"fit": sorted(ordered[:fit_end]), "calibrate": sorted(ordered[fit_end:fit_end + held_out]),
            "audit": sorted(ordered[fit_end + held_out:])}


def _checked_records(records, allowed, name):
    result = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CalibrationError(f"{name} records must be mappings")
        episode = record.get("episode_id")
        if episode not in allowed:
            raise CalibrationError(f"Non-TRAIN or unknown episode in {name}: {episode!r}")
        if record.get("split", "train") != "train":
            raise CalibrationError(f"Only TRAIN records are allowed in {name}")
        result.append(dict(record))
    return result


def _robust_covariance(values, floor_m2):
    """Huberized residual second moment, including the robust residual bias.

    Centering and a coordinate MAD make extreme outliers bounded. The robust
    bias outer product is retained because inference does not subtract a
    label-derived bias correction from its measured position.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 6:
        raise CalibrationError("Each source needs at least six residuals in fit")
    location = np.median(values, axis=0)
    deviations = values - location
    scale = np.maximum(1.4826 * np.median(np.abs(deviations), axis=0), np.sqrt(floor_m2))
    distances = np.linalg.norm(deviations / scale, axis=1)
    weights = np.minimum(1., 2.5 / np.maximum(distances, 1e-12))
    # Squared weights bound second moments of arbitrarily distant outliers.
    bounded = deviations * weights[:, None]
    covariance = bounded.T @ bounded / max(float(np.sum(weights ** 2)) - 1., 1.)
    covariance += np.outer(location, location)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2.)
    covariance = (vectors * np.maximum(eigenvalues, floor_m2)) @ vectors.T
    return covariance, location


def _coverage(values, covariance):
    values = np.asarray(values, float).reshape(-1, 3)
    if not len(values):
        return dict(count=0, mean_mahalanobis2=None, coverage_95=None)
    mahalanobis = np.einsum("ni,ij,nj->n", values, np.linalg.inv(covariance), values)
    return dict(count=len(values), mean_mahalanobis2=float(np.mean(mahalanobis)),
                coverage_95=float(np.mean(mahalanobis <= chi2.ppf(.95, 3))))


def _trajectory_statistics(records, fit_episodes, variance_floor):
    groups = defaultdict(list)
    for row in records:
        identity = row.get("target_id")
        if not isinstance(identity, (str, int)) or isinstance(identity, bool):
            raise CalibrationError("TRAIN trajectories require a scalar target_id")
        timestamp = _real(row.get("time_s"), "trajectory time_s")
        position = _vector(row.get("position_m"), "trajectory position_m")
        groups[row["episode_id"], identity].append((timestamp, position))
    velocity, acceleration = [], []
    for (episode, _), observations in groups.items():
        observations.sort(key=lambda item: item[0])
        times = np.asarray([item[0] for item in observations])
        positions = np.asarray([item[1] for item in observations])
        if np.any(np.diff(times) <= 0):
            raise CalibrationError("Duplicate/nonincreasing time within a TRAIN target trajectory")
        if episode not in fit_episodes or len(times) < 2:
            continue
        dt = np.diff(times)
        segment_velocity = np.diff(positions, axis=0) / dt[:, None]
        velocity.extend(segment_velocity)
        if len(times) >= 3:
            # Second differences account for actual physical times, not PRI.
            acceleration.extend(2 * np.diff(segment_velocity, axis=0) / (dt[:-1] + dt[1:])[:, None])
    if not acceleration:
        raise CalibrationError("Fit TRAIN trajectories need at least three timestamps for acceleration calibration")
    # Robust isotropic zero-mean uncertainty; 75th component-square quantile
    # keeps isolated annotation spikes from controlling an entire filter bank.
    accel_variance = max(float(np.quantile(np.asarray(acceleration) ** 2, .75)), variance_floor)
    initial_velocity_variance = max(float(np.quantile(np.asarray(velocity) ** 2, .75)), variance_floor)
    return dict(acceleration_variance=accel_variance,
                initial_velocity_variance=initial_velocity_variance,
                fit_velocity_segments=len(velocity), fit_acceleration_segments=len(acceleration),
                units=dict(acceleration_variance="m2/s4", initial_velocity_variance="m2/s2"))


def _fit_radar_association(residual_rows, trajectory_rows, partitions, variance_floor):
    """Freeze radar association bounds from TRAIN motion and FFT residuals.

    Position gates cover zero-initial-velocity displacement plus a two-error
    measurement bound. Velocity gates cover TRAIN speed plus that error bound
    per observed interval. This is a declared empirical conservative policy;
    it does not claim GT velocity is available during normal association.
    """
    membership = {episode: part for part, episodes in partitions.items() for episode in episodes}
    groups = defaultdict(list)
    for row in trajectory_rows:
        groups[row["episode_id"], row["target_id"]].append(
            (float(row["time_s"]), np.asarray(row["position_m"], float)))
    speeds = {part: [] for part in partitions}
    intervals = {part: [] for part in partitions}
    for (episode, _), values in groups.items():
        values.sort(key=lambda value: value[0])
        if len(values) < 2:
            continue
        times = np.asarray([value[0] for value in values])
        positions = np.asarray([value[1] for value in values])
        dt = np.diff(times)
        part = membership[episode]
        speeds[part].extend(np.linalg.norm(np.diff(positions, axis=0) / dt[:, None], axis=1))
        intervals[part].extend(dt)
    for part in ("fit", "calibrate"):
        if not speeds[part] or not residual_rows[part]:
            raise CalibrationError(f"Radar association requires TRAIN {part} trajectories and FFT residuals")
    fit_noise = np.linalg.norm(np.asarray(residual_rows["fit"]), axis=1)
    calibrate_noise = np.linalg.norm(np.asarray(residual_rows["calibrate"]), axis=1)
    typical_dt = float(np.median(intervals["fit"]))
    fit_noise_scale = max(float(np.quantile(fit_noise, .75)), np.sqrt(variance_floor))
    velocity_scale = max(float(np.quantile(speeds["fit"], .75)) + np.sqrt(2) * fit_noise_scale / typical_dt, .1)
    position_scale = max(velocity_scale * typical_dt + np.sqrt(2) * fit_noise_scale, .1)
    calibration_speed = float(np.quantile(speeds["calibrate"], .99))
    two_error_bound = 2. * float(np.quantile(calibrate_noise, .99))
    position_gate = max(calibration_speed * max(intervals["calibrate"]) + two_error_bound, position_scale)
    velocity_gate = max(calibration_speed + two_error_bound / min(intervals["calibrate"]), velocity_scale)
    velocity_weight = .25
    params = dict(position_scale_m=position_scale, velocity_scale_m_s=velocity_scale,
                  position_gate_m=position_gate, velocity_gate_m_s=velocity_gate,
                  velocity_weight=velocity_weight,
                  cost_gate=(1 - velocity_weight) * position_gate / position_scale
                            + velocity_weight * velocity_gate / velocity_scale,
                  ema_velocity=.5, max_missed=5)
    diagnostics = dict(
        speed_segment_counts={part: len(rows) for part, rows in speeds.items()},
        policy="fit scales from TRAIN speed/FFT error; calibrate 99% speed and two-error empirical gates",
        fixed_public_policy=dict(velocity_weight=.25, ema_velocity=.5, max_missed=5),
        gt_motion_used="offline TRAIN calibration only", audit_used_for_tuning=False,
    )
    return params, diagnostics


def visual_pair_components(previous_boxes, current_boxes, previous_roi, current_roi, *, predicted_boxes=None):
    """Compute matched-pair features using the exact visual tracker cost code.

    All arrays contain already paired visible detections in the same order.
    The offline caller may use TRAIN IDs to pair them. Provide online-style
    image-motion predicted boxes when available; no 3D prediction is accepted.
    """
    from ._legacy.visual_association.tracker import pairwise_cost_components
    previous_boxes = np.asarray(previous_boxes, float)
    current_boxes = np.asarray(current_boxes, float)
    if previous_boxes.shape != current_boxes.shape:
        raise CalibrationError("Paired previous/current boxes must have equal shapes")
    predicted = previous_boxes if predicted_boxes is None else np.asarray(predicted_boxes, float)
    if predicted.shape != previous_boxes.shape:
        raise CalibrationError("predicted_boxes must have the paired box shape")
    components = pairwise_cost_components(predicted, current_boxes, previous_roi, current_roi)
    return {name: np.diag(values).copy() for name, values in components.items()}


def _fit_visual(records, partitions, appearance_weight, quantile):
    names = ("center_residual", "appearance_distance", "size_residual")
    grouped = {part: [] for part in partitions}
    membership = {episode: part for part, episodes in partitions.items() for episode in episodes}
    for row in records:
        values = [_real(row.get(name), name, nonnegative=True) for name in names]
        if values[1] > 2. + 1e-8:
            raise CalibrationError("appearance_distance must be one minus cosine, between zero and two")
        grouped[membership[row["episode_id"]]].append(values)
    if len(grouped["fit"]) < 4 or len(grouped["calibrate"]) < 4 or not grouped["audit"]:
        raise CalibrationError("Visual calibration requires >=4 fit/calibrate positive pairs and >=1 audit pair")
    fit = np.asarray(grouped["fit"])
    calibrate = np.asarray(grouped["calibrate"])
    scales = np.maximum(np.quantile(fit, .75, axis=0), .05)
    weights = np.asarray([1. - appearance_weight, appearance_weight, .1])
    costs = (calibrate / scales * weights).sum(axis=1)
    params = dict(appearance_weight=appearance_weight, center_scale=float(scales[0]),
                  app_scale=float(scales[1]), size_scale=float(scales[2]),
                  cost_gate=max(float(np.quantile(costs, quantile)), .05),
                  center_gate=max(float(np.quantile(calibrate[:, 0], quantile)), .05),
                  ema_velocity=.5, ema_appearance=.2)
    audit = np.asarray(grouped["audit"])
    audit_cost = (audit / scales * weights).sum(axis=1)
    diagnostics = dict(
        counts={part: len(values) for part, values in grouped.items()},
        audit_positive_pair_gate_acceptance=float(np.mean((audit_cost <= params["cost_gate"])
                                                         & (audit[:, 0] <= params["center_gate"]))),
        policy="TRAIN positive matched visual pairs; fit robust scales, calibrate quantile gates, audit without tuning",
        appearance_weight_policy="fixed public mixture weight; no old-dataset thresholds reused",
        gate_quantile=quantile, uses_3d=False, uses_radar=False,
    )
    return params, diagnostics


def fit_calibration(residual_records, trajectory_records, visual_pair_records, *,
                    manifest_sha256, train_episode_ids, seed=20261126,
                    measurement_floor_m2=.01, process_variance_floor=1e-4,
                    visual_appearance_weight=.25, visual_gate_quantile=.99):
    """Fit a complete runtime calibration artifact from authorized TRAIN data.

    Residual row: ``episode_id, source, residual_m:[3]``. Trajectory row:
    ``episode_id, target_id, time_s, position_m:[3]``. Visual row:
    ``episode_id, center_residual, appearance_distance, size_residual``.
    An optional ``split`` field must equal ``train``. Each residual source
    needs >=6 fit, >=4 calibrate, and >=1 audit samples. Insufficient evidence
    raises CalibrationError; it is never serialized as a calibrated artifact.
    """
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_sha256):
        raise CalibrationError("manifest_sha256 must be a SHA256 hex digest")
    floor = _real(measurement_floor_m2, "measurement_floor_m2", nonnegative=True)
    process_floor = _real(process_variance_floor, "process_variance_floor", nonnegative=True)
    if min(floor, process_floor) <= 0:
        raise CalibrationError("Variance floors must be positive")
    appearance_weight = _real(visual_appearance_weight, "visual_appearance_weight", nonnegative=True)
    quantile = _real(visual_gate_quantile, "visual_gate_quantile")
    if appearance_weight > 1 or not .5 < quantile < 1:
        raise CalibrationError("Invalid visual mixture weight or gate quantile")
    partitions = partition_train_episodes(train_episode_ids, seed)
    authorized = set(train_episode_ids)
    residuals = _checked_records(residual_records, authorized, "residual")
    trajectories = _checked_records(trajectory_records, authorized, "trajectory")
    visual = _checked_records(visual_pair_records, authorized, "visual pair")
    membership = {episode: part for part, episodes in partitions.items() for episode in episodes}
    grouped = {source: {part: [] for part in partitions} for source in SOURCES}
    canonical_residuals = []
    for row in residuals:
        source = row.get("source")
        if source not in SOURCES:
            raise CalibrationError(f"Unknown normal inference source: {source!r}")
        residual = _vector(row.get("residual_m"), "residual_m")
        grouped[source][membership[row["episode_id"]]].append(residual)
        canonical_residuals.append(dict(episode_id=row["episode_id"], source=source, residual_m=residual.tolist()))
    motion = _trajectory_statistics(trajectories, set(partitions["fit"]), process_floor)
    kalman, source_diagnostics = {}, {}
    for source in SOURCES:
        rows = grouped[source]
        if len(rows["calibrate"]) < 4 or not rows["audit"]:
            raise CalibrationError(f"{source} requires >=4 calibrate and >=1 audit residuals")
        fitted, location = _robust_covariance(rows["fit"], floor)
        calibration_values = np.asarray(rows["calibrate"])
        squared = np.einsum("ni,ij,nj->n", calibration_values, np.linalg.inv(fitted), calibration_values)
        inflation = max(1., float(np.quantile(squared, .95)) / float(chi2.ppf(.95, 3)))
        covariance = fitted * inflation
        kalman[source] = dict(measurement_covariance=covariance.tolist(),
                              acceleration_variance=motion["acceleration_variance"],
                              initial_velocity_variance=motion["initial_velocity_variance"])
        source_diagnostics[source] = dict(
            fit_count=len(rows["fit"]), robust_residual_location_m=location.tolist(),
            covariance_inflation=inflation,
            calibrate=_coverage(rows["calibrate"], covariance), audit=_coverage(rows["audit"], covariance),
            covariance_policy="robust fit residual second moment, TRAIN calibration 95% quantile inflation",
        )
    visual_params, visual_diagnostics = _fit_visual(visual, partitions, appearance_weight, quantile)
    radar_association, radar_diagnostics = _fit_radar_association(grouped["fft"], trajectories, partitions, floor)
    # Canonical numeric inputs record exactly which exported evidence was used.
    canonical_trajectories = [dict(episode_id=row["episode_id"], target_id=row["target_id"],
                                   time_s=float(row["time_s"]), position_m=_vector(row["position_m"], "position_m").tolist())
                              for row in trajectories]
    canonical_visual = [dict(episode_id=row["episode_id"], **{
        key: float(row[key]) for key in ("center_residual", "appearance_distance", "size_residual")}) for row in visual]
    # Sorting row JSON strings keeps hashes stable when equivalent exports are
    # read in a different filesystem order.
    evidence = {"residuals": sorted(_digest(row) for row in canonical_residuals),
                "trajectories": sorted(_digest(row) for row in canonical_trajectories),
                "visual_pairs": sorted(_digest(row) for row in canonical_visual)}
    artifact = dict(
        schema=SCHEMA, status="frozen", split="train", manifest_sha256=manifest_sha256.lower(),
        neural_architecture="single_query_depth_v3",
        status_reason="All required sources and visual/radar association calibrated on TRAIN fit/calibrate episodes; audit episodes unused for tuning",
        seed=int(seed), train_episode_ids=sorted(authorized),
        train_episode_ids_sha256=_digest(sorted(authorized)),
        partitions={part: dict(episode_ids=episodes, episode_ids_sha256=_digest(episodes))
                    for part, episodes in partitions.items()},
        partition_sha256=_digest(partitions), input_records_sha256=_digest(evidence),
        visual_params=visual_params, kalman=kalman, radar_association=radar_association,
        actual_counts={part: dict(episodes=len(episodes),
                                  residuals={source: len(grouped[source][part]) for source in SOURCES},
                                  trajectory_observations=sum(row["episode_id"] in episodes for row in trajectories),
                                  visual_pairs=sum(row["episode_id"] in episodes for row in visual))
                       for part, episodes in partitions.items()},
        diagnostics=dict(kalman=source_diagnostics, visual=visual_diagnostics, motion=motion,
                         radar_association=radar_diagnostics),
        provenance=dict(labels_used="separately loaded TRAIN only, offline calibration",
                        predictions_used="exported normal inference; labels never enter normal inference",
                        dev_used=False, test_used=False, audit_used_for_tuning=False,
                        data_access="numeric records only; no dataset loading inside module",
                        calibration_is_accuracy_certification=False),
    )
    # Ensure callers can write strict JSON without NaN or hidden ndarray data.
    json.dumps(artifact, sort_keys=True, allow_nan=False)
    return artifact


def validate_calibration_artifact(artifact, *, manifest_sha256=None, train_episode_ids=None):
    """Fail closed on an uncalibrated, mismatched or malformed runtime file."""
    if not isinstance(artifact, Mapping):
        raise CalibrationError("Calibration artifact must be a mapping")
    if artifact.get('neural_architecture') != 'single_query_depth_v3':
        raise CalibrationError('Calibration was not produced for the single_query_depth_v3 architecture')
    if (artifact.get("schema") != SCHEMA or artifact.get("status") != "frozen"
            or artifact.get("split") != "train"):
        raise CalibrationError("A calibrated TRAIN rvloc.calibration.v1 artifact is required")
    if manifest_sha256 is not None and artifact.get("manifest_sha256") != str(manifest_sha256).lower():
        raise CalibrationError("Calibration manifest SHA256 mismatch")
    if train_episode_ids is not None and artifact.get("train_episode_ids_sha256") != _digest(sorted(train_episode_ids)):
        raise CalibrationError("Calibration TRAIN episode hash mismatch")
    episodes = artifact.get("train_episode_ids")
    if (not isinstance(episodes, list) or not episodes or len(set(episodes)) != len(episodes)
            or artifact.get("train_episode_ids_sha256") != _digest(sorted(episodes))):
        raise CalibrationError("Invalid calibration TRAIN episode provenance")
    partitions = artifact.get("partitions", {})
    if set(partitions) != {"fit", "calibrate", "audit"}:
        raise CalibrationError("Missing fit/calibrate/audit episode partitions")
    flattened = []
    partition_ids = {}
    for name, partition in partitions.items():
        ids = partition.get("episode_ids", [])
        if not ids or partition.get("episode_ids_sha256") != _digest(ids):
            raise CalibrationError("Invalid calibration partition hash")
        flattened.extend(ids)
        partition_ids[name] = ids
    if sorted(flattened) != sorted(episodes) or len(set(flattened)) != len(flattened):
        raise CalibrationError("Calibration episode partitions overlap or omit TRAIN episodes")
    if artifact.get("partition_sha256") != _digest(partition_ids):
        raise CalibrationError("Calibration partition provenance mismatch")
    from ._legacy.visual_association.tracker import VisualAssociationTracker
    try:
        VisualAssociationTracker(params=artifact.get("visual_params"), variant="visual_gated")
    except (ValueError, TypeError) as error:
        raise CalibrationError("Invalid calibrated visual parameters") from error
    if set(artifact.get("visual_params", {})) != {
            "appearance_weight", "center_scale", "app_scale", "size_scale", "cost_gate", "center_gate", "ema_velocity", "ema_appearance"}:
        raise CalibrationError("Complete explicit calibrated visual parameters are required")
    if set(artifact.get("kalman", {})) != set(SOURCES):
        raise CalibrationError("Calibration must contain all three normal measurement sources")
    for source, params in artifact["kalman"].items():
        covariance = np.asarray(params.get("measurement_covariance"), float)
        if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, atol=1e-9)
                or np.linalg.eigvalsh(covariance).min() <= 0):
            raise CalibrationError(f"Invalid positive-definite {source} measurement covariance")
        for name in ("acceleration_variance", "initial_velocity_variance"):
            if _real(params.get(name), name, nonnegative=True) <= 0:
                raise CalibrationError(f"{name} must be positive")
    from .tracking.radar import RadarAssociator
    if not isinstance(artifact.get("radar_association"), Mapping):
        raise CalibrationError("Explicit TRAIN-frozen radar association parameters are required")
    try:
        RadarAssociator(**artifact["radar_association"])
    except (TypeError, ValueError) as error:
        raise CalibrationError("Invalid calibrated radar association parameters") from error
    provenance = artifact.get("provenance", {})
    if any(provenance.get(name) is not False for name in ("dev_used", "test_used", "audit_used_for_tuning")):
        raise CalibrationError("Calibration must declare no DEV/TEST use or audit tuning")
    json.dumps(dict(artifact), allow_nan=False)
    return artifact
