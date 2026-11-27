"""Generated-sensor checks for shared direct-depth support and supervision.

No real dataset, weights, or TEST split is opened by these fixtures.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from rvloc.contracts import ContractError, RFMetadata
from rvloc.data import (DIRECT_DEPTH_SCHEMA, DIRECT_FUSION_SCHEMA, load_manifest,
                        prepare_radar, sha256, validate_direct_fusion_features)
from rvloc.supervision import match_direct_ray_labels
from test_data import synthetic_manifest, write_json, _rewrite_normal


@pytest.fixture
def direct_pair(tmp_path):
    path, manifest, normal = synthetic_manifest(tmp_path)
    data = Path(manifest["data_root"])
    # Deliberately choose a normalization scale far below the measured RF
    # extent. Neither normalization nor measured RF defines optical support.
    manifest["visual_geometry"] = dict(depth_min_m=.2, depth_max_m=900., range_scale_m=17.)
    normal["camera"]["position_m"] = [-1000, 0, 2]
    normal["camera"]["rotation_frd_to_frame"] = np.diag([-1., -1., -1.]).tolist()
    normal["observer_position_m"] = [-1000, 0, 1]
    normal["observer_scatter_reference_position_m"] = [-1000, 0, 2]
    normal["navigation"]["position_m"] = [[-1001, 0, 1], [-1000, 0, 1]]
    rf = RFMetadata.from_mapping(manifest["rf"])
    prepare_radar(data / "iq.npy", data / "radar.npz", rf=rf, angle_bins=9, spatial_oversampling=1)
    with np.load(data / "radar.npz", allow_pickle=False) as file:
        maps = {key: file[key] for key in file.files}
    maps["ra_valid"][3, 2] = False
    np.savez_compressed(data / "radar.npz", **maps)
    normal["radar"] = dict(path="radar.npz", sha256=sha256(data / "radar.npz"))
    _rewrite_normal(path, manifest, normal)
    parsed = load_manifest(path)
    rgb, = parsed.observations("train", modality="rgb")
    fused, = parsed.observations("train", modality="rgb_radar")
    return rgb, fused, maps


def test_visual_and_fusion_share_optical_geometry_even_outside_rf(direct_pair):
    rgb, fused, maps = direct_pair
    shared = ("ray_origins", "ray_directions", "ray_valid", "ray_depth_bounds",
              "observer_position", "bbox_xyxy", "detection_scores", "visual_scale",
              "detector_roi_features", "camera_K", "camera_image_size_wh")
    for field in shared:
        np.testing.assert_array_equal(rgb.features[field], fused.features[field], err_msg=field)
    assert rgb.features["range_scale_m"] == fused.features["range_scale_m"] == 17.
    assert rgb.features["depth_schema"] == fused.features["depth_schema"] == DIRECT_DEPTH_SCHEMA
    assert fused.features["schema"] == DIRECT_FUSION_SCHEMA
    # This entire ray is behind the array and outside its unambiguous range.
    # It remains a valid visible object for all direct regressors.
    assert fused.features["ray_valid"].tolist() == [True]
    assert fused.features["ray_depth_bounds"].tolist() == [[.2, 900.]]
    assert fused.features["metadata"]["geometry_bounds_source"] == "public_visual_geometry"
    assert fused.features["metadata"]["radar_support_defines_output_bounds"] is False
    for key in ("range_m", "azimuth_rad", "elevation_rad", "ra_power", "re_power", "ra_valid", "re_valid"):
        np.testing.assert_array_equal(fused.features[key], maps[key], err_msg=key)
    assert fused.features["range_m"][-1] > fused.features["range_scale_m"]
    assert not fused.features["ra_valid"][3, 2]
    assert rgb.iq is None


def test_direct_validator_preserves_original_tensor_gradient(direct_pair):
    import torch
    features = copy.deepcopy(direct_pair[1].features)
    features["ra_power"] = torch.tensor(features["ra_power"], requires_grad=True)
    original = features["ra_power"]
    assert validate_direct_fusion_features(features) is features
    assert features["ra_power"] is original
    features["ra_power"].sum().backward()
    assert torch.equal(original.grad, torch.ones_like(original))


@pytest.mark.parametrize("mutation", ["legacy", "negative_power", "mask_type", "reversed_axis", "gt", "future"])
def test_direct_validator_rejects_malformed_or_hidden_inputs(direct_pair, mutation):
    features = copy.deepcopy(direct_pair[1].features)
    if mutation == "legacy":features["schema"] = "base_rays_ra_re_v1"
    elif mutation == "negative_power":features["ra_power"][0, 0] = -1
    elif mutation == "mask_type":features["re_valid"] = features["re_valid"].astype(int)
    elif mutation == "reversed_axis":features["range_m"] = features["range_m"][::-1]
    elif mutation == "gt":features["target_positions"] = np.ones((1, 3))
    else:features["metadata"]["future_read"] = True
    with pytest.raises((ContractError, ValueError)):
        validate_direct_fusion_features(features)


def geometric_features(origins, directions, bounds, *, scale=17., valid=None):
    directions = np.asarray(directions, float).reshape(-1, 3)
    count = len(directions)
    return dict(ray_origins=np.asarray(origins, float).reshape(count, 3), ray_directions=directions,
                ray_depth_bounds=np.asarray(bounds, float).reshape(count, 2),
                ray_valid=np.ones(count, bool) if valid is None else np.asarray(valid, bool), range_scale_m=scale)


def test_supervision_uses_camera_support_not_station_range_or_halfspace(direct_pair):
    rgb, fused, _ = direct_pair
    target = np.array([[-1100., 0., 2.]])
    assert np.linalg.norm(target[0]) > fused.rf.range_limit
    for observation in (rgb, fused):
        match = match_direct_ray_labels(observation.features, target)
        assert match["rows"].tolist() == [0] and match["cols"].tolist() == [0]
        np.testing.assert_allclose(match["depths_m"], [100.])
        assert match["diagnostics"]["supervision_domain"] == "public_camera_ray_support"
        assert match["diagnostics"]["station_radius_filter_applied"] is False
        assert match["diagnostics"]["station_front_halfspace_filter_applied"] is False


def test_supervision_assignment_is_invariant_to_normalization_scale():
    features = geometric_features([[1000, 0, 0], [-1000, 0, 0]], [[1, 0, 0], [-1, 0, 0]], [[.1, 200], [.1, 200]])
    targets = np.array([[1100., .1, 0.], [-1100., 0., .1]])
    first = match_direct_ray_labels(features, targets)
    features["range_scale_m"] = 5000.
    second = match_direct_ray_labels(features, targets)
    assert first["rows"].tolist() == second["rows"].tolist() == [0, 1]
    assert first["cols"].tolist() == second["cols"].tolist() == [0, 1]
    np.testing.assert_allclose(first["depths_m"], second["depths_m"])
    np.testing.assert_allclose(first["depths_m"], [100., 100.])


def test_supervision_rejects_off_ray_and_outside_public_depth_with_dummy_assignment():
    features = geometric_features([[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0]], [[.1, 20], [.1, 5]])
    match = match_direct_ray_labels(features, [[10, .1, 0], [0, 10, 0], [-10, 0, 0]])
    assert match["rows"].tolist() == [0] and match["cols"].tolist() == [0]
    # Use the projected along-ray target, without clipping or converting to
    # the Euclidean camera-target distance.
    np.testing.assert_allclose(match["depths_m"], [10.])
    assert match["diagnostics"]["unmatched_detection_count"] == 1
    assert match["diagnostics"]["unmatched_label_count"] == 2


def test_supervision_empty_labels_invalid_rays_and_empty_frames_are_explicit():
    features = geometric_features([[0, 0, 0]], [[1, 0, 0]], [[.1, 20]], valid=[False])
    no_valid = match_direct_ray_labels(features, [[10, 0, 0]])
    assert len(no_valid["rows"]) == 0 and no_valid["diagnostics"]["invalid_detection_count"] == 1
    features["ray_valid"][:] = True
    assert len(match_direct_ray_labels(features, [])['rows']) == 0
    empty = geometric_features([], [], [])
    match = match_direct_ray_labels(empty, [[10, 0, 0]])
    assert match["diagnostics"]["training_detected_count"] == 0
    assert match["diagnostics"]["unmatched_label_count"] == 1
