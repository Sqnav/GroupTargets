"""Prepared replay security/identity tests use generated sensor assets only."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rvloc.contracts import ContractError
from rvloc.data import load_manifest, load_observation, prepare_radar, sha256
from test_data import synthetic_manifest, write_json


def stamp(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, device=value.st_dev, inode=value.st_ino)


@pytest.fixture
def replay(tmp_path):
    path, manifest, normal = synthetic_manifest(tmp_path)
    data = Path(manifest["data_root"])
    prepare_radar(data / "iq.npy", data / "radar.npz", rf=load_manifest(path).rf,
                  angle_bins=9, spatial_oversampling=1)
    normal["radar"] = dict(path="radar.npz", sha256=sha256(data / "radar.npz"))
    frame = manifest["episodes"][0]["frames"][0]
    frame["normal_sha256"] = write_json(data / "normal.json", normal)
    receipt = dict(schema="rvloc.prepared_replay.v1", raw_iq_verified=True,
        sensor_export_labels_read=False,
        fft_source=dict(kind="authenticated_camera_center_cache_replay",
                        kernel="rectangular_native_ra_re_v1", sources={"native.py": "b" * 64}),
        frames=[dict(episode_id="generated_episode", frame_idx=0,
            normal_sha256=frame["normal_sha256"], iq_path=normal["iq"]["path"],
            iq_sha256=normal["iq"]["sha256"], iq_stat=stamp(data / "iq.npy"),
            radar_sha256=normal["radar"]["sha256"], visual_sha256=normal["visual"]["sha256"])])
    manifest["prepared_replay"] = dict(path="replay.json", sha256=write_json(data / "replay.json", receipt))
    write_json(path, manifest)
    return path, manifest, normal, receipt


def rewrite_receipt(replay):
    path, manifest, _, receipt = replay
    data = Path(manifest["data_root"])
    manifest["prepared_replay"]["sha256"] = write_json(data / "replay.json", receipt)
    write_json(path, manifest)


def rewrite_normal(replay):
    path, manifest, normal, _ = replay
    manifest["episodes"][0]["frames"][0]["normal_sha256"] = write_json(Path(manifest["data_root"]) / "normal.json", normal)
    write_json(path, manifest)


def test_prepared_only_never_reads_iq_labels_or_runs_fft(replay, monkeypatch):
    path, _, _, _ = replay
    manifest = load_manifest(path)
    original = type(manifest)._read
    opened = []
    def guarded(self, relative, expected):
        assert relative not in ("iq.npy", "labels.json")
        opened.append(relative)
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    monkeypatch.setattr("rvloc.data.prepare.spectra_from_iq", lambda *a, **k: pytest.fail("Prepared replay recomputed FFT"))
    first, = manifest.observations("train", modality="rgb_radar", prepared_only=True)
    second, = manifest.observations("train", modality="rgb_radar", prepared_only=True)
    assert first.iq is None and second.iq is None
    assert first.features["metadata"]["iq_read"] is False
    assert first.features["metadata"]["prepared_replay_verified"] is True
    assert first.features["metadata"]["labels_read"] is False
    assert first.features["detector_roi_features"].shape == (1, 1024)
    assert first.source_hashes["prepared_replay"] == manifest.prepared_replay["sha256"]
    assert opened == ["center.json", "replay.json", "normal.json", "visual.npz", "radar.npz"] * 2
    for name in ("ra_power", "re_power", "ray_depth_bounds", "detector_roi_features"):
        np.testing.assert_array_equal(first.features[name], second.features[name])


def test_prepared_and_strict_inputs_are_numerically_identical(replay):
    manifest = load_manifest(replay[0])
    strict, = manifest.observations("train", modality="rgb_radar")
    prepared, = manifest.observations("train", modality="rgb_radar", prepared_only=True)
    assert strict.iq is not None and prepared.iq is None
    for key, value in strict.features.items():
        if key == "metadata":
            continue
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, prepared.features[key])
        else:
            assert value == prepared.features[key]


def test_default_still_reads_raw_iq_and_does_not_trust_replay_receipt(replay, monkeypatch):
    path, manifest_dict, _, _ = replay
    data = Path(manifest_dict["data_root"])
    (data / "replay.json").write_text("untrusted changed receipt")
    manifest = load_manifest(path)
    opened = []
    original = type(manifest)._read
    def guarded(self, relative, expected):
        opened.append(relative)
        assert relative != "replay.json"
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    obs, = manifest.observations("train", modality="rgb_radar")
    assert obs.iq is not None and "iq.npy" in opened


def test_fft_still_reads_iq_and_rgb_still_reads_no_radar_when_flag_set(replay, monkeypatch):
    manifest = load_manifest(replay[0])
    opened = []
    original = type(manifest)._read
    def guarded(self, relative, expected):
        opened.append(relative)
        assert relative not in ("replay.json", "labels.json")
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    radar, = manifest.observations("train", modality="radar", prepared_only=True)
    assert radar.iq is not None and opened == ["center.json", "normal.json", "iq.npy"]
    opened.clear()
    rgb, = manifest.observations("train", modality="rgb", prepared_only=True)
    assert rgb.iq is None and opened == ["center.json", "normal.json", "visual.npz"]


def test_missing_receipt_fails_closed_even_with_prepared_arrays(replay):
    path, manifest, _, _ = replay
    del manifest["prepared_replay"]
    write_json(path, manifest)
    parsed = load_manifest(path)
    with pytest.raises(ContractError, match="registered full-IQ"):
        list(parsed.observations("train", modality="rgb_radar", prepared_only=True))
    with pytest.raises(ContractError, match="registered full-IQ"):
        load_observation(parsed, next(parsed.frames("train")), modality="rgb_radar", prepared_only=True)


@pytest.mark.parametrize("mutation", ["unverified_iq", "labels_read", "missing_frame", "duplicate_frame", "extra_test_frame", "normal_sha", "bad_stat", "unknown_kernel"])
def test_invalid_authentication_or_incomplete_inventory_fails_closed(replay, mutation):
    _, _, _, receipt = replay
    if mutation == "unverified_iq":
        receipt["raw_iq_verified"] = False
    elif mutation == "labels_read":
        receipt["sensor_export_labels_read"] = True
    elif mutation == "missing_frame":
        receipt["frames"].clear()
    elif mutation == "duplicate_frame":
        receipt["frames"].append(copy.deepcopy(receipt["frames"][0]))
    elif mutation == "extra_test_frame":
        receipt["frames"].append(dict(receipt["frames"][0], episode_id="historical_test_not_in_manifest"))
    elif mutation == "normal_sha":
        receipt["frames"][0]["normal_sha256"] = "c" * 64
    elif mutation == "bad_stat":
        receipt["frames"][0]["iq_stat"]["size"] = True
    else:
        receipt["fft_source"]["kernel"] = "unverified_radar_model"
    rewrite_receipt(replay)
    with pytest.raises(ContractError):
        list(load_manifest(replay[0]).observations("train", modality="rgb_radar", prepared_only=True))


def test_authentication_must_cover_dev_as_well_as_train(replay):
    path, manifest, _, _ = replay
    episode = copy.deepcopy(manifest["episodes"][0])
    episode["episode_id"] = "new_dev_episode"
    manifest["episodes"].append(episode)
    manifest["splits"]["dev"].append("new_dev_episode")
    write_json(path, manifest)
    with pytest.raises(ContractError, match="every manifest frame"):
        list(load_manifest(path).observations("train", modality="rgb_radar", prepared_only=True))


@pytest.mark.parametrize("mutation", ["bytes", "replace_inode", "remove"])
def test_changed_raw_iq_requires_reauthentication_without_reading_it(replay, mutation, monkeypatch):
    path, manifest_dict, _, _ = replay
    source = Path(manifest_dict["data_root"]) / "iq.npy"
    if mutation == "bytes":
        old = source.stat()
        with source.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif mutation == "replace_inode":
        temporary = source.with_suffix(".replacement")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(source)
    else:
        source.unlink()
    manifest = load_manifest(path)
    original = type(manifest)._read
    def guarded(self, relative, expected):
        assert relative not in ("iq.npy", "labels.json")
        return original(self, relative, expected)
    monkeypatch.setattr(type(manifest), "_read", guarded)
    with pytest.raises(ContractError, match="authenticate and register"):
        list(manifest.observations("train", modality="rgb_radar", prepared_only=True))


@pytest.mark.parametrize("asset", ["normal.json", "visual.npz", "radar.npz", "replay.json"])
def test_small_asset_hashes_remain_authenticated(replay, asset):
    path, manifest, _, _ = replay
    source = Path(manifest["data_root"]) / asset
    source.write_bytes(source.read_bytes() + b"corruption")
    with pytest.raises(ContractError, match="SHA256 mismatch"):
        list(load_manifest(path).observations("train", modality="rgb_radar", prepared_only=True))


def test_registered_normal_cannot_change_its_iq_or_prepared_identity(replay):
    path, _, normal, receipt = replay
    normal["iq"]["sha256"] = "d" * 64
    rewrite_normal(replay)
    receipt["frames"][0]["normal_sha256"] = replay[1]["episodes"][0]["frames"][0]["normal_sha256"]
    rewrite_receipt(replay)
    with pytest.raises(ContractError, match="sensor identities differ"):
        list(load_manifest(path).observations("train", modality="rgb_radar", prepared_only=True))


def test_prepared_radar_must_bind_the_authenticated_raw_iq(replay):
    path, manifest, normal, receipt = replay
    source = Path(manifest["data_root"]) / "radar.npz"
    with np.load(source, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    arrays["source_iq_sha256"] = np.array("e" * 64)
    np.savez_compressed(source, **arrays)
    normal["radar"]["sha256"] = sha256(source)
    rewrite_normal(replay)
    receipt["frames"][0].update(normal_sha256=manifest["episodes"][0]["frames"][0]["normal_sha256"],
                                radar_sha256=normal["radar"]["sha256"])
    rewrite_receipt(replay)
    with pytest.raises(ContractError, match="current measured IQ"):
        list(load_manifest(path).observations("train", modality="rgb_radar", prepared_only=True))


def test_prepared_flag_and_receipt_path_are_strict(replay):
    path, manifest, _, _ = replay
    with pytest.raises(ContractError, match="boolean"):
        list(load_manifest(path).observations("train", modality="rgb_radar", prepared_only=1))
    manifest["prepared_replay"]["path"] = "../outside.json"
    write_json(path, manifest)
    with pytest.raises(ContractError):
        load_manifest(path)


def test_metadata_inspection_does_not_open_the_receipt(replay, monkeypatch):
    manifest = load_manifest(replay[0])
    monkeypatch.setattr(type(manifest), "_read", lambda *a, **k: pytest.fail("Metadata opened a data asset"))
    assert manifest.metadata_report()["data_files_opened"] is False
