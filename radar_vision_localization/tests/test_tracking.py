"""Synthetic image/position CPU fixtures, with no dataset or weight access."""
import inspect

import numpy as np
import pytest

from rvloc.tracking import KalmanBank, RadarAssociator, VisualAssociator, cv_predict, position_update


def image_observations(centers, descriptor_axes=None):
    centers = np.asarray(centers, float).reshape(-1, 2)
    boxes = np.column_stack((centers - 5., centers + 5.))
    roi = np.zeros((len(centers), 1024))
    if descriptor_axes is None:
        descriptor_axes = range(len(centers))
    for row, axis in enumerate(descriptor_axes):
        roi[row, axis] = 1.
    return boxes, roi, np.ones(len(boxes))


def visual_step(tracker, index, centers, axes=None):
    return tracker.step(index, float(index), *image_observations(centers, axes))


def test_visual_adapter_matches_frozen_implementation_and_all_observations():
    from rvloc._legacy.visual_association.tracker import VisualAssociationTracker
    adapter, original = VisualAssociator(), VisualAssociationTracker(variant="visual_gated")
    sequence = [([(20, 20), (80, 20)], [0, 1]),
                ([(22, 20), (78, 20)], [0, 1]),
                ([(24, 20), (76, 20)], [0, 1]),
                ([(74, 20), (26, 20), (400, 400)], [1, 0, 2])]
    for index, (centers, axes) in enumerate(sequence):
        boxes, roi, scores = image_observations(centers, axes)
        actual = adapter.step(index, float(index), boxes, roi, scores)
        expected = original.step(dict(frame_idx=index, time=float(index), boxes=boxes, roi=roi, scores=scores))
        assert actual.tolist() == [row["track_id"] for row in expected["outputs"]]
        assert len(actual) == len(boxes)
    assert actual[-1] is None
    assert adapter.last_result["current_observation_coverage_complete"] is True


def test_g5_g6_g7_use_identical_visual_association_without_3d_boundary():
    associators = [VisualAssociator() for _ in range(3)]
    for index in range(5):
        ids = [visual_step(tracker, index, [(20 + index, 20), (80 - index, 20)])
               for tracker in associators]
        assert ids[0].tolist() == ids[1].tolist() == ids[2].tolist()
    assert set(inspect.signature(VisualAssociator.step).parameters) == {
        "self", "frame_idx", "time", "boxes", "roi", "scores"}
    boxes, roi, scores = image_observations([(30, 20)])
    with pytest.raises(TypeError):
        associators[0].step(6, 6., boxes, roi, scores, positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="strictly"):
        associators[0].step(4, 4., boxes, roi, scores)


def test_visual_crossing_reorder_and_reset_do_not_use_detection_indices_as_ids():
    tracker = VisualAssociator(params={"appearance_weight": .9})
    first = visual_step(tracker, 0, [(20, 20), (80, 20)], [0, 1])
    for index, centers in enumerate(([(30, 20), (70, 20)], [(40, 20), (60, 20)]), 1):
        visual_step(tracker, index, centers, [0, 1])
    switched = visual_step(tracker, 3, [(50, 20), (50, 21)], [1, 0])
    assert switched.tolist() == first[::-1].tolist()
    tracker.reset()
    assert visual_step(tracker, 0, [(20, 20), (80, 20)], [0, 1]).tolist() == first.tolist()


def test_radar_position_velocity_association_reorder_gap_and_outlier():
    tracker = RadarAssociator(position_gate_m=10., velocity_gate_m_s=10.)
    first = tracker.step(0., [[0., 0., 0.], [100., 0., 0.]])
    second = tracker.step(1., [[98., 0., 0.], [2., 0., 0.]])
    assert second.tolist() == first[::-1].tolist()
    assert tracker.step(2., np.empty((0, 3))).shape == (0,)
    returned = tracker.step(3., [[6., 0., 0.], [94., 0., 0.], [1000., 0., 0.]])
    assert returned[:2].tolist() == first.tolist()
    assert returned[2] not in first
    invalid = tracker.step(4., [[np.nan, 0., 0.], [8., 0., 0.]])
    assert invalid[0] is None and invalid[1] == first[0]
    assert set(inspect.signature(RadarAssociator.step).parameters) == {"self", "time", "points"}
    assert tracker.last_result["uses_rgb"] is False


def test_radar_unmatched_can_create_new_ids_and_expired_tracks_do_not_resurrect():
    tracker = RadarAssociator(max_missed=0)
    first = tracker.step(0., [[0., 0., 0.]])[0]
    tracker.step(1., [])
    assert tracker.step(2., [[0., 0., 0.]])[0] != first
    with pytest.raises(ValueError, match="strictly"):
        tracker.step(2., [])


def test_cv_prediction_has_physical_dt_and_six_state_order():
    state = np.array([1., 2., 3., 4., -2., .5])
    predicted, covariance = cv_predict(state, np.eye(6), dt=2., acceleration_variance=0.)
    np.testing.assert_array_equal(predicted, [9., -2., 4., 4., -2., .5])
    assert covariance[0, 3] == 2. and covariance[0, 0] == 5.
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_kalman_recovers_cv_velocity_and_preserves_unassociated_invalid_rows():
    bank = KalmanBank(acceleration_variance=.001, measurement_variance=.01,
                      initial_velocity_variance=100.)
    for time in range(8):
        point = np.array([[2. * time, -.5 * time, 1.]])
        filtered = bank.update(float(time), [7], point)
        assert filtered.shape == (1, 3)
    np.testing.assert_allclose(bank.states[7].x[3:], [2., -.5, 0.], atol=2e-3)
    result = bank.update(8., [7, None, 8], [[16., -4., 1.], [500., 0., 0.], [np.nan, 1., 2.]],
                         valid=np.array([True, True, False]))
    np.testing.assert_array_equal(result[1], [500., 0., 0.])
    assert np.isnan(result[2, 0]) and 8 not in bank.states
    assert len(result) == 3 and set(bank.states) == {7}
    covariance = bank.states[7].covariance
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_kalman_covariance_controls_gain_and_failed_calls_do_not_advance_time():
    x = np.zeros(6)
    p = np.eye(6)
    accurate, p_accurate = position_update(x, p, [1., 1., 1.], np.eye(3) * .01)
    uncertain, _ = position_update(x, p, [1., 1., 1.], np.eye(3) * 100.)
    assert np.linalg.norm(accurate[:3] - 1.) < np.linalg.norm(uncertain[:3] - 1.)
    assert np.linalg.eigvalsh(p_accurate).min() > 0
    bank = KalmanBank()
    bank.update(0., [1], [[0., 0., 0.]])
    with pytest.raises(ValueError, match="at most once"):
        bank.update(1., [1, 1], [[1., 0., 0.], [2., 0., 0.]])
    assert bank.time == 0.
    with pytest.raises(ValueError, match="covariance"):
        bank.update(1., [1], [[1., 0., 0.]], covariance=np.diag([-1., 1., 1.]))
    assert bank.time == 0.
    bank.update(1., [1], [[1., 0., 0.]])
    with pytest.raises(ValueError, match="strictly"):
        bank.update(1., [], [])


def test_kalman_no_prediction_only_rows_and_gap_reset():
    bank = KalmanBank(max_gap_s=1.)
    bank.update(0., [1], [[0., 0., 0.]])
    assert bank.update(.5, [], []).shape == (0, 3)
    assert bank.update(2., [], []).shape == (0, 3)
    assert bank.states == {}
