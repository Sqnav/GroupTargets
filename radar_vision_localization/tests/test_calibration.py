"""Synthetic-only TRAIN-calibration provenance and numerical checks."""
import copy
import json

import numpy as np
import pytest

from rvloc.calibration import (CalibrationError, fit_calibration, partition_train_episodes,
                               validate_calibration_artifact, visual_pair_components)


TRAIN = [f"train_episode_{index:02d}" for index in range(10)]
MANIFEST = "f" * 64


def evidence():
    rng = np.random.default_rng(8731)
    residuals, trajectories, pairs = [], [], []
    for episode in TRAIN:
        for source, scale in (("fft", 2.), ("rgb", 5.), ("single_frame", 1.)):
            for row in rng.normal(size=(24, 3)) * [scale, scale * .5, scale * .25]:
                residuals.append(dict(episode_id=episode, source=source, residual_m=row.tolist(), split="train"))
        for index in range(5):
            trajectories.append(dict(episode_id=episode, target_id="memberA", time_s=float(index),
                                     position_m=[index * 2., -.5 * index + .1 * index ** 2, 3.]))
        for _ in range(12):
            pairs.append(dict(episode_id=episode, center_residual=float(rng.uniform(.05, .3)),
                              appearance_distance=float(rng.uniform(.01, .15)),
                              size_residual=float(rng.uniform(.02, .1))))
    return residuals, trajectories, pairs


def calibrate(rows=None):
    rows = evidence() if rows is None else rows
    return fit_calibration(*rows, manifest_sha256=MANIFEST, train_episode_ids=TRAIN)


def test_partition_is_whole_episode_deterministic_and_disjoint():
    first = partition_train_episodes(TRAIN)
    assert first == partition_train_episodes(TRAIN[::-1])
    assert all(first.values())
    assert set.union(*(set(value) for value in first.values())) == set(TRAIN)
    assert sum(map(len, first.values())) == len(TRAIN)
    assert first != partition_train_episodes(TRAIN, seed=3)


def test_complete_artifact_serializes_and_has_explicit_train_provenance():
    artifact = calibrate()
    json.dumps(artifact, allow_nan=False)
    assert artifact["status"] == "frozen" and artifact["split"] == "train"
    assert artifact["manifest_sha256"] == MANIFEST
    assert validate_calibration_artifact(artifact, manifest_sha256=MANIFEST, train_episode_ids=TRAIN) is artifact
    assert artifact["provenance"]["test_used"] is False
    for params in artifact["kalman"].values():
        covariance = np.asarray(params["measurement_covariance"])
        assert covariance.shape == (3, 3) and np.linalg.eigvalsh(covariance).min() > 0
        assert params["acceleration_variance"] > 0
        assert params["initial_velocity_variance"] > 0
    assert artifact["diagnostics"]["visual"]["uses_3d"] is False
    assert artifact["radar_association"]["position_gate_m"] > 0
    assert artifact["actual_counts"]["audit"]["episodes"] > 0


def test_audit_labels_never_change_calibrated_values():
    rows = evidence()
    original = calibrate(rows)
    audit_episodes = set(original["partitions"]["audit"]["episode_ids"])
    changed = copy.deepcopy(rows)
    for row in changed[0]:
        if row["episode_id"] in audit_episodes:
            row["residual_m"] = [1e6, -2e6, 3e6]
    for row in changed[1]:
        if row["episode_id"] in audit_episodes:
            row["position_m"] = (np.asarray(row["position_m"]) * 1000).tolist()
    for row in changed[2]:
        if row["episode_id"] in audit_episodes:
            row["center_residual"] = 1000.
    revised = calibrate(changed)
    assert revised["kalman"] == original["kalman"]
    assert revised["visual_params"] == original["visual_params"]
    assert revised["radar_association"] == original["radar_association"]
    assert revised["diagnostics"]["kalman"]["fft"]["audit"] != original["diagnostics"]["kalman"]["fft"]["audit"]
    assert revised["input_records_sha256"] != original["input_records_sha256"]


def test_non_train_records_and_insufficient_evidence_fail_closed():
    rows = evidence()
    rows[0][0]["episode_id"] = "DEV_episode"
    with pytest.raises(CalibrationError, match="Non-TRAIN"):
        calibrate(rows)
    rows = evidence()
    rows[2][0]["split"] = "test"
    with pytest.raises(CalibrationError, match="Only TRAIN"):
        calibrate(rows)
    with pytest.raises(CalibrationError, match="trajectories"):
        fit_calibration([], [], [], manifest_sha256=MANIFEST, train_episode_ids=TRAIN)
    with pytest.raises(CalibrationError, match="three"):
        partition_train_episodes(TRAIN[:2])


def test_runtime_validation_rejects_manifest_partition_and_uncalibrated_files():
    artifact = calibrate()
    with pytest.raises(CalibrationError, match="manifest"):
        validate_calibration_artifact(artifact, manifest_sha256="a" * 64)
    changed = copy.deepcopy(artifact)
    changed["partitions"]["audit"]["episode_ids"].append(TRAIN[0])
    with pytest.raises(CalibrationError, match="partition"):
        validate_calibration_artifact(changed)
    changed = copy.deepcopy(artifact)
    changed["status"] = "not_calibrated"
    with pytest.raises(CalibrationError, match="calibrated TRAIN"):
        validate_calibration_artifact(changed)
    changed = copy.deepcopy(artifact)
    del changed["visual_params"]["cost_gate"]
    with pytest.raises(CalibrationError, match="explicit"):
        validate_calibration_artifact(changed)


def test_calibration_is_order_independent_and_fits_full_anisotropic_covariance():
    rows = evidence()
    first = calibrate(rows)
    second = calibrate(tuple(list(reversed(values)) for values in rows))
    # Floating reductions are compared numerically; provenance is canonical.
    assert first["input_records_sha256"] == second["input_records_sha256"]
    assert first["partition_sha256"] == second["partition_sha256"]
    for source in first["kalman"]:
        a = np.asarray(first["kalman"][source]["measurement_covariance"])
        b = np.asarray(second["kalman"][source]["measurement_covariance"])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert a[0, 0] > a[1, 1] > a[2, 2]


def test_visual_pair_components_use_image_only_matched_diagonal():
    previous = np.array([[0., 0., 10., 10.], [30., 0., 40., 10.]])
    current = previous + [2., 0., 2., 0.]
    roi = np.zeros((2, 1024))
    roi[0, 0], roi[1, 1] = 1., 1.
    components = visual_pair_components(previous, current, roi, roi)
    np.testing.assert_allclose(components["center_residual"], 2. / np.sqrt(200.))
    np.testing.assert_array_equal(components["appearance_distance"], [0., 0.])
    predicted = previous + [2., 0., 2., 0.]
    motion_compensated = visual_pair_components(previous, current, roi, roi, predicted_boxes=predicted)
    np.testing.assert_array_equal(motion_compensated["center_residual"], [0., 0.])
