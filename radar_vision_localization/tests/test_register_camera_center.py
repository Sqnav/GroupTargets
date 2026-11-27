"""Registration boundary and resumability tests use synthetic episodes only."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rvloc.contracts import ContractError
from rvloc.data import load_labels, load_manifest, sha256
import rvloc.data.register_camera_center as reg


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return sha256(path)


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return sha256(path)


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    workspace = tmp_path / "group"
    branch = workspace / "data/source_branch"
    results = workspace / "results/source_branch"
    episode = "episode_train_only"
    folder = branch / "data" / episode
    protocol = dict(selection=dict(train=[episode], val=[]))
    protocol_sha = _json(results / "protocol.json", protocol)
    captures, echoes, rows = [], [], []
    basis = np.diag([1., 1., -1.])
    for index in range(2):
        rgb = folder / f"rgb/frame_{index:05d}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"synthetic-RGB-identity-" + str(index).encode())
        iq = folder / f"echoes/iq/frame_{index:05d}.npy"
        iq.parent.mkdir(parents=True, exist_ok=True)
        np.save(iq, np.zeros((40, 128, 16, 16), np.complex64))
        captures.append(dict(frame_idx=index, sim_timestamp_ns=9e17+index*100,
            rgb_path=f"rgb/frame_{index:05d}.png", camera=dict(camera_position_ned=[index, 0., 1.],
            camera_orientation_wxyz=[1., 0., 0., 0.]),
            observer=dict(position_ned=[index-.46, 0., 1.]), group=dict(ids=["DO_NOT_EXPORT_GT"])))
        echoes.append(dict(frame_idx=index, physical_time_s=float(index), iq_path=f"iq/frame_{index:05d}.npy",
            iq_sha256=sha256(iq), observer_reference=dict(position_ned_m=[index, 0., 1.],
                velocity_ned_m_s=[12345., 0., 0.]),  # forbidden simulator-only velocity
            target_ids=["offline-A", "offline-B"], target_positions_ned_m=[[10., 0., 1.], [20., 0., 1.]],
            target_velocities_ned_m_s=[[1., 0., 0.], [2., 0., 0.]]))
    capture_sha = _jsonl(folder / "frames.jsonl", captures)
    capture_meta_sha = _json(folder / "metadata.json", dict(frame_count=2))
    echo_sha = _jsonl(folder / "echoes/frames.jsonl", echoes)
    metadata = dict(schema_version="group_plus_observer_camera_optical_center_v1",
        station_position_ned_m=[0., 0., 0.], array_basis_in_ned=basis.tolist(),
        provenance=dict(config=dict(carrier_hz=4.9e9, bandwidth_hz=32e6,
            pulse_interval_s=.0025, frame_interval_s=1., iq_shape=[40, 128, 16, 16])))
    echo_meta_sha = _json(folder / "echoes/metadata.json", metadata)
    generation = dict(identity=dict(protocol_sha256=protocol_sha, capture_sha256=capture_sha,
        capture_metadata_sha256=capture_meta_sha), frames=2, all_original_replays_exact=True,
        output_echo_frames_sha256=echo_sha, output_echo_metadata_sha256=echo_meta_sha)
    generation_sha = _json(results / "generation_receipts" / (episode+".json"), generation)
    for index in range(2):
        identity = dict(iq_sha256=echoes[index]["iq_sha256"], test="synthetic-cache")
        f = dict(camera_K=np.array([[640., 0., 640.], [0., 640., 360.], [0., 0., 1.]]),
            camera_image_size_wh=np.array([1280, 720]), bbox_xyxy=np.array([[620., 340., 660., 380.]]),
            detection_scores=np.array([.9]), detector_roi_features=np.ones((1, 1024), np.float32),
            ray_origins=np.array([[index, 0., -1.]]), ray_directions=np.array([[1., 0., 0.]]),
            ra_power=np.ones((128, 3), np.float32), re_power=np.ones((128, 3), np.float32),
            ra_valid=np.ones((128, 3), bool), re_valid=np.ones((128, 3), bool),
            range_m=np.arange(128)*299792458./(2*32e6), azimuth_rad=np.array([-.5, 0., .5]),
            elevation_rad=np.array([-.5, 0., .5]), metadata=dict(
                spectrum=dict(version="rectangular_native_ra_re_v1"), sensor_identity=dict(
                    rgb_sha256=sha256(folder/captures[index]["rgb_path"]),
                    detector_contract=dict(weights_sha256="a"*64)), camera_branch=dict(
                    iq_sha256=echoes[index]["iq_sha256"], echo_frames_sha256=echo_sha,
                    echo_metadata_sha256=echo_meta_sha)))
        cache = branch / "features" / episode / f"frame_{index:05d}.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(features=f, receipt=identity), cache)
        rows.append(dict(frame_idx=index, camera_feature_sha256=sha256(cache), branch_identity=identity,
            hashes=dict(rgb=sha256(folder/captures[index]["rgb_path"]), iq=echoes[index]["iq_sha256"],
                        source_trajectory=capture_sha, source_metadata=capture_meta_sha),
            detector_roi_cache=dict(path="historical-roi-cache", sha256="b"*64),
            source_feature_cache=dict(path="historical-feature-cache", sha256="c"*64)))
    _json(results / "feature_receipts" / (episode+".json"), dict(identity=dict(
        protocol_sha256=protocol_sha, generation_receipt_sha256=generation_sha), frames=rows))
    source = reg._Source(workspace, branch, results, expected_counts=(1, 0), protocol_sha256=protocol_sha)
    monkeypatch.setattr(reg, "_default_source", lambda: source)
    return source, workspace / "data/export"


def test_registration_sensor_label_separation_geometry_and_causal_velocity(synthetic_source):
    source, output = synthetic_source
    result = reg.register_camera_center(output, workers=2)
    manifest = load_manifest(result["manifest"])
    observations = list(manifest.observations("train", modality="rgb"))
    assert len(observations) == 2
    assert observations[0].features["ray_depth_bounds"].tolist() == [[.1, 600.]]
    assert observations[0].navigation.velocity_mps.tolist() == [[0., 0., 0.]]
    assert observations[1].navigation.velocity_mps.tolist() == [[0., 0., 0.], [1., 0., 0.]]
    assert observations[1].observer_position.tolist() == [.54, 0., -1.]
    assert observations[1].camera.position.tolist() == [1., 0., -1.]
    for frame in manifest.frames("train"):
        normal = (source.data_root/frame.normal_path).read_text()
        assert "offline-A" not in normal and "DO_NOT_EXPORT_GT" not in normal and "target" not in normal
    assert load_labels(manifest, observations[0]).member_ids == ("offline-A", "offline-B")
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["max_detected_members"] == 1  # there are two GT: never used for capacity
    assert receipt["frames"] == 2 and receipt["raw_iq_verified"]
    replay = json.loads((output/"prepared_replay_receipt.json").read_text())
    assert replay["sensor_export_labels_read"] is False
    assert set(replay["frames"][0]["iq_stat"]) == {"size", "mtime_ns", "ctime_ns", "device", "inode"}


def test_resume_authenticates_outputs_and_reuses_verified_iq(synthetic_source):
    _, output = synthetic_source
    first = reg.register_camera_center(output, workers=1)
    manifest_sha = sha256(first["manifest"])
    second = reg.register_camera_center(output, workers=2)
    assert sha256(second["manifest"]) == manifest_sha
    runtime = json.loads(Path(second["runtime_receipt"]).read_text())
    assert runtime["iq_sha_recomputed"] == 0 and runtime["iq_prior_verified_stat_reused"] == 2
    visual = output/"episode_train_only/frame_00000/visual.npz"
    visual.write_bytes(b"corrupt")
    with pytest.raises((ValueError, ContractError)):
        reg.register_camera_center(output, workers=1)


def test_source_iq_tampering_is_not_hidden_by_cached_proof(synthetic_source):
    source, output = synthetic_source
    reg.register_camera_center(output, workers=1)
    iq = source.branch/"data/episode_train_only/echoes/iq/frame_00000.npy"
    with iq.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    with pytest.raises(ContractError, match="Source SHA256 mismatch"):
        reg.register_camera_center(output, workers=1)


def test_receipt_only_mode_and_source_protection(synthetic_source):
    source, output = synthetic_source
    result = reg.register_camera_center(output, workers=1, verify_iq=False)
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["raw_iq_verified"] is False
    runtime = json.loads(Path(result["runtime_receipt"]).read_text())
    assert runtime["iq_legacy_receipt_only"] == 2 and runtime["iq_sha_recomputed"] == 0
    with pytest.raises(ContractError, match="source branch"):
        reg.register_camera_center(source.branch/"forbidden")
    with pytest.raises(ContractError, match="data_root"):
        reg.register_camera_center(source.workspace/"outside_data")
