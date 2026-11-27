"""Contract tests use generated assets only; no experiment/dataset is opened."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rvloc.contracts import ContractError, PendingDataError, RFMetadata
from rvloc.data import DIRECT_DEPTH_SCHEMA, load_labels, load_manifest, prepare_radar, prepare_visual, sha256
from rvloc.data.prepare import validate_radar_arrays


def rf_mapping():
    return dict(carrier_hz=4.9e9, bandwidth_hz=32e6, pulse_interval_s=.0025,
                shape=[40, 128, 16, 16], spacing_y_m=299792458. / 4.9e9 / 2,
                spacing_z_m=299792458. / 4.9e9 / 2, frame_interval_s=1.,
                frequencies_hz=((np.arange(128) - 64) * 250000.).tolist(),
                frequency_grid="centered_baseband_uniform")


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return sha256(path)


def synthetic_manifest(tmp_path):
    """Small shared fixture suitable for every route's generated-data smoke."""
    data = tmp_path / "external_data"
    data.mkdir()
    Image.new("RGB", (1280, 720)).save(data / "rgb.png")
    prepare_visual(data / "rgb.png", data / "visual.npz", detector_sha256="a" * 64,
                   detector_callback=lambda rgb: dict(bbox_xyxy=np.array([[620., 340., 660., 380.]]),
                    detection_scores=np.array([.9]), detector_roi_features=np.ones((1, 1024), np.float32)))
    evidence = dict(schema="rvloc.center_evidence.v1", coordinate_frame="base_local",
        reference="observer_self_scatter_reference", co_location="camera_optical_center",
        camera_to_scatter_reference_offset_m=[0, 0, 0], source="synthetic capture installation", capture_revision="synthetic-v1")
    evidence_sha = write_json(data / "center.json", evidence)
    np.save(data / "iq.npy", np.zeros((40, 128, 16, 16), np.complex64))
    normal = dict(schema="rvloc.observation.v1", episode_id="generated_episode", frame_idx=0, physical_time_s=10.,
        camera=dict(K=[[640, 0, 640], [0, 640, 360], [0, 0, 1]], position_m=[5, 0, 2],
                    rotation_frd_to_frame=[[1, 0, 0], [0, 1, 0], [0, 0, -1]], image_size_wh=[1280, 720]),
        calibration=dict(station_position_m=[0, 0, 0], array_basis_fru_to_frame=np.eye(3).tolist()),
        observer_position_m=[5, 0, 1], observer_scatter_reference_position_m=[5, 0, 2],
        navigation=dict(physical_time_s=[9., 10.], position_m=[[4, 0, 1], [5, 0, 1]], velocity_mps=[[1, 0, 0], [1, 0, 0]]),
        visual=dict(path="visual.npz", sha256=sha256(data / "visual.npz")),
        iq=dict(path="iq.npy", sha256=sha256(data / "iq.npy")),
        radar=dict(path="radar_does_not_exist.npz", sha256="0" * 64))
    normal_sha = write_json(data / "normal.json", normal)
    labels = dict(schema="rvloc.labels.v1", episode_id="generated_episode", frame_idx=0, physical_time_s=10.,
                  positions_m=[[25, 0, 2]], member_ids=["offline-only-id"], velocities_mps=[[0, 0, 0]])
    label_sha = write_json(data / "labels.json", labels)
    manifest = dict(schema="rvloc.manifest.v1", status="ready", coordinate_frame="base_local", data_root=str(data),
        rf=rf_mapping(), visual_geometry=dict(depth_min_m=.1, depth_max_m=200., range_scale_m=200.),
        center_contract=dict(reference="observer_self_scatter_reference", co_location="camera_optical_center", status="verified",
                             evidence=dict(path="center.json", sha256=evidence_sha)),
        splits=dict(train=["generated_episode"], dev=[]),
        episodes=[dict(episode_id="generated_episode", frames=[dict(frame_idx=0, physical_time_s=10., normal_path="normal.json",
            normal_sha256=normal_sha, labels_path="labels.json", labels_sha256=label_sha)])])
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return path, manifest, normal


def _rewrite_normal(path, manifest, normal):
    manifest["episodes"][0]["frames"][0]["normal_sha256"] = write_json(Path(manifest["data_root"]) / "normal.json", normal)
    write_json(path, manifest)


def test_rgb_never_opens_iq_radar_or_labels(tmp_path, monkeypatch):
    path, _, _ = synthetic_manifest(tmp_path)
    manifest = load_manifest(path)
    original = type(manifest)._read
    opened = []
    def guarded(self, relative, expected):
        assert relative not in ("iq.npy", "labels.json", "radar_does_not_exist.npz")
        opened.append(relative)
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    obs, = manifest.observations("train", modality="rgb")
    assert obs.iq is None and not hasattr(obs, "member_ids")
    assert opened == ["center.json", "normal.json", "visual.npz"]
    assert obs.features["ray_depth_bounds"].tolist() == [[.1, 200.]]
    assert obs.features["detector_roi_features"].shape == (1, 1024)
    assert obs.camera.width == 1280 and obs.camera.height == 720
    assert obs.observer_scatter_reference_position.tolist() == [5, 0, 2]
    assert obs.features["observer_position"].tolist() == [5, 0, 2]
    assert obs.observer_position.tolist() == [5, 0, 1]


def test_radar_only_never_opens_visual_or_radar_cache(tmp_path, monkeypatch):
    path, _, _ = synthetic_manifest(tmp_path)
    manifest = load_manifest(path)
    original = type(manifest)._read
    opened = []
    def guarded(self, relative, expected):
        assert relative not in ("visual.npz", "labels.json", "radar_does_not_exist.npz")
        opened.append(relative)
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    obs, = manifest.observations("train", modality="radar")
    assert obs.features == {} and obs.iq.shape == (40, 128, 16, 16)
    assert opened == ["center.json", "normal.json", "iq.npy"]


def test_labels_are_an_explicit_independent_access(tmp_path):
    path, _, _ = synthetic_manifest(tmp_path)
    manifest = load_manifest(path)
    obs, = manifest.observations("train")
    labels = load_labels(manifest, obs)
    assert labels.positions.tolist() == [[25, 0, 2]]
    assert labels.member_ids == ("offline-only-id",)
    assert not any("target" in key or "member_id" in key for key in obs.features)


def test_pending_metadata_check_opens_no_data_assets(tmp_path, monkeypatch):
    path, manifest, _ = synthetic_manifest(tmp_path)
    manifest.update(status="pending_data_fix", episodes=[], splits=dict(train=[], dev=[]), data_root=str(tmp_path / "not_created"))
    manifest["center_contract"].update(status="pending", evidence=None)
    write_json(path, manifest)
    with pytest.raises(PendingDataError):
        load_manifest(path)
    parsed = load_manifest(path, allow_pending=True)
    monkeypatch.setattr(type(parsed), "_read", lambda *args: pytest.fail("Pending metadata check accessed data"))
    assert parsed.metadata_report()["status"] == "pending_data_fix"
    assert parsed.metadata_report()["frames"] == 0
    with pytest.raises(PendingDataError):
        list(parsed.observations("train"))


@pytest.mark.parametrize("mutation", ["future", "unknown_gt", "wrong_center", "sha"])
def test_sensor_boundary_rejects_violations(tmp_path, mutation):
    path, manifest, normal = synthetic_manifest(tmp_path)
    if mutation == "future":
        normal["navigation"]["physical_time_s"] = [10., 11.]
    elif mutation == "unknown_gt":
        normal["truth_count"] = 1
    elif mutation == "wrong_center":
        normal["observer_scatter_reference_position_m"] = [0, 0, 0]
    else:
        normal["visual"]["sha256"] = "f" * 64
    _rewrite_normal(path, manifest, normal)
    with pytest.raises(ContractError):
        list(load_manifest(path).observations("train"))


def test_split_chronology_paths_and_external_root_are_strict(tmp_path):
    path, manifest, _ = synthetic_manifest(tmp_path)
    for modify in (lambda x: x["splits"]["dev"].append("generated_episode"),
                   lambda x: x["episodes"][0]["frames"][0].update(normal_path="../outside.json"),
                   lambda x: x["episodes"][0]["frames"].append(dict(x["episodes"][0]["frames"][0], frame_idx=1, physical_time_s=9.)),
                   lambda x: x.update(unapproved_field="labels")):
        changed = copy.deepcopy(manifest)
        modify(changed)
        write_json(path, changed)
        with pytest.raises(ContractError):
            load_manifest(path)
    write_json(path, manifest)
    with pytest.raises(ContractError):
        load_manifest(path, code_root=tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    normal_path = Path(manifest["data_root"]) / "normal.json"
    normal_path.unlink()
    normal_path.symlink_to(outside)
    with pytest.raises(ContractError):
        list(load_manifest(path).observations("train"))


def test_rf_requires_explicit_native_grid():
    rf = RFMetadata.from_mapping(rf_mapping())
    assert rf.frequencies_hz[0] == -16e6 and rf.pulse_times_s[-1] == .0975
    changed = rf_mapping()
    del changed["frequencies_hz"]
    with pytest.raises(ContractError):
        RFMetadata.from_mapping(changed)


def test_raw_preparation_and_fused_features_use_native_rf(tmp_path):
    path, manifest, normal = synthetic_manifest(tmp_path)
    data = Path(manifest["data_root"])
    rf = RFMetadata.from_mapping(manifest["rf"])
    prepare_radar(data / "iq.npy", data / "radar.npz", rf=rf, angle_bins=9, spatial_oversampling=1)
    normal["radar"] = dict(path="radar.npz", sha256=sha256(data / "radar.npz"))
    _rewrite_normal(path, manifest, normal)
    obs, = load_manifest(path).observations("train", modality="rgb_radar")
    assert obs.features["ra_power"].shape == (128, 9)
    assert obs.features["depth_schema"] == DIRECT_DEPTH_SCHEMA
    assert obs.features["metadata"]["iq_read"] is True
    assert obs.features["detector_roi_features"].shape == (1, 1024)
    np.testing.assert_array_equal(obs.features["observer_position"],obs.camera.position)
    with np.load(data / "radar.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["range_m"] = arrays["range_m"] * 2
    with pytest.raises(ContractError):
        validate_radar_arrays(arrays, rf, normal["iq"]["sha256"])


@pytest.mark.parametrize("modality",["rgb","rgb_radar"])
def test_body_navigation_is_retained_for_audit_without_changing_network_inputs(tmp_path,modality):
    path,manifest,normal=synthetic_manifest(tmp_path)
    if modality=="rgb_radar":
        data=Path(manifest['data_root'])
        prepare_radar(data/'iq.npy',data/'radar.npz',rf=RFMetadata.from_mapping(manifest['rf']),
                      angle_bins=9,spatial_oversampling=1)
        normal['radar']=dict(path='radar.npz',sha256=sha256(data/'radar.npz'))
        _rewrite_normal(path,manifest,normal)
    before,=load_manifest(path).observations('train',modality=modality)
    normal['observer_position_m']=[50,3,8]
    normal['navigation']['position_m'][-1]=[50,3,8]
    normal['navigation']['velocity_mps'][-1]=[46,3,7]
    _rewrite_normal(path,manifest,normal)
    after,=load_manifest(path).observations('train',modality=modality)
    assert after.observer_position.tolist()==[50,3,8]
    assert before.features.keys()==after.features.keys()
    for key,value in before.features.items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,after.features[key])
        else:assert value==after.features[key]
    assert after.features['metadata']['observer_position_reference']=='camera_optical_center'
