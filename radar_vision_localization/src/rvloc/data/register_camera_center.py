"""Register the authenticated optical-centre branch without recomputing FFT.

Only the frozen TRAIN/DEV episode lists are traversed. Sensor exports and
supervision have separate files and builders; target data never enters the
normal observation or prepared features. All positions use the fixed RF frame.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np

from rvloc.contracts import Camera, ContractError, RFMetadata
from rvloc.data import load_manifest, sha256
from rvloc.data.prepare import (RADAR_ARRAY_FIELDS, RADAR_SCHEMA, VISUAL_SCHEMA,
                               validate_radar_arrays, validate_visual_arrays)

PROTOCOL_SHA256 = "c0e4228822a2ba7a9ab38a3995e4f84b64695f2afa8fb0bbae99a89be0f0cee9"
GEOMETRY = dict(depth_min_m=.1, depth_max_m=600., range_scale_m=600.)


@dataclass(frozen=True)
class _Source:
    workspace: Path
    branch: Path
    results: Path
    expected_counts: tuple[int, int] = (700, 150)
    protocol_sha256: str = PROTOCOL_SHA256

    @property
    def data_root(self):
        return self.workspace / "data"


def _default_source():
    workspace = Path(__file__).resolve().parents[5]
    suffix = "dataset_construction/camera_center_branch_20261126"
    return _Source(workspace, workspace / "data" / suffix, workspace / "results" / suffix)


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _check(condition, message):
    if not condition:
        raise ContractError(message)


def _verified(path, expected):
    _check(sha256(path) == expected, f"Source SHA256 mismatch: {path}")


def _stat(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def _publish(path, raw):
    """Existing outputs are immutable; resume authenticates their exact bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _check(path.read_bytes() == raw, f"Existing output differs; choose a fresh export directory: {path}")
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def _publish_json(path, value):
    return _publish(path, _json_bytes(value))


def _publish_npz(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            _check(set(previous.files) == set(arrays), f"Prepared fields changed: {path}")
            for key, value in arrays.items():
                _check(np.array_equal(previous[key], value), f"Prepared content changed: {path}/{key}")
        return sha256(path)
    encoded = BytesIO()
    np.savez_compressed(encoded, **arrays)
    return _publish(path, encoded.getvalue())


def _relative(path, source):
    resolved = Path(path).resolve()
    _check(resolved.is_relative_to(source.data_root.resolve()), "Export/reference must stay inside data_root")
    return resolved.relative_to(source.data_root.resolve()).as_posix()


def _rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    _check(q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-12,
           "Invalid public camera quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _rf(metadata):
    config = metadata["provenance"]["config"]
    wavelength = 299792458. / config["carrier_hz"]
    return RFMetadata.from_mapping(dict(carrier_hz=config["carrier_hz"], bandwidth_hz=config["bandwidth_hz"],
        pulse_interval_s=config["pulse_interval_s"], shape=config["iq_shape"],
        spacing_y_m=config.get("spacing_y_m") or wavelength/2,
        spacing_z_m=config.get("spacing_z_m") or wavelength/2,
        frequencies_hz=((np.arange(128)-64)*config["bandwidth_hz"]/128).tolist(),
        frequency_grid="centered_baseband_uniform", frame_interval_s=config["frame_interval_s"]))


def _public_frame(*, episode, capture, index, timestamp, features, station, basis, visual_ref, radar_ref, iq_ref):
    """Sensor-only builder: receives no echo row, labels, target IDs or counts."""
    current = capture[index]
    camera_pos = (np.asarray(current["camera"]["camera_position_ned"])-station) @ basis
    rotation = basis.T @ _rotation(current["camera"]["camera_orientation_wxyz"])
    camera = dict(K=np.asarray(features["camera_K"]).tolist(), position_m=camera_pos.tolist(),
                  rotation_frd_to_frame=rotation.tolist(),
                  image_size_wh=np.asarray(features["camera_image_size_wh"]).tolist())
    Camera.from_mapping(camera)
    body = (np.asarray([row["observer"]["position_ned"] for row in capture[:index+1]])-station) @ basis
    velocities = np.zeros_like(body)
    if index:
        velocities[1:] = np.diff(body, axis=0)  # fixed, independently validated 1 s physical interval
    times = (np.arange(index+1, dtype=float) + timestamp-index).tolist()
    boxes = np.asarray(features["bbox_xyxy"])
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    optical = np.column_stack((centers, np.ones(len(boxes)))) @ np.linalg.inv(np.asarray(camera["K"])).T
    rays = optical[:, [2, 0, 1]] @ rotation.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    _check(np.allclose(features["ray_origins"], camera_pos, rtol=0, atol=1e-8), "Cached optical origin disagrees with public pose")
    _check(np.allclose(features["ray_directions"], rays, rtol=0, atol=1e-8), "Cached ray disagrees with public camera calibration")
    return dict(schema="rvloc.observation.v1", episode_id=episode, frame_idx=index, physical_time_s=timestamp,
                camera=camera, observer_position_m=body[-1].tolist(),
                observer_scatter_reference_position_m=camera_pos.tolist(),
                calibration=dict(station_position_m=[0., 0., 0.], array_basis_fru_to_frame=np.eye(3).tolist()),
                navigation=dict(physical_time_s=times, position_m=body.tolist(), velocity_mps=velocities.tolist()),
                visual=visual_ref, radar=radar_ref, iq=iq_ref)


def _offline_labels(episode, index, timestamp, echo, station, basis):
    """The only export builder that accesses target supervision."""
    return dict(schema="rvloc.labels.v1", episode_id=episode, frame_idx=index, physical_time_s=timestamp,
                positions_m=((np.asarray(echo["target_positions_ned_m"])-station) @ basis).tolist(),
                velocities_mps=(np.asarray(echo["target_velocities_ned_m_s"]) @ basis).tolist(),
                member_ids=echo["target_ids"])


def _export_episode(episode, split, source, output, protocol_sha, verify_iq):
    import torch
    start = time.perf_counter()
    folder = source.branch / "data" / episode
    generation_path = source.results / "generation_receipts" / (episode + ".json")
    features_path = source.results / "feature_receipts" / (episode + ".json")
    generation, feature_receipt = _read(generation_path), _read(features_path)
    metadata_path, frames_path = folder / "echoes/metadata.json", folder / "echoes/frames.jsonl"
    metadata = _read(metadata_path)
    _check(metadata["schema_version"] == "group_plus_observer_camera_optical_center_v1", "Wrong echo centre schema")
    _check(generation["all_original_replays_exact"], "Original IQ replay was not certified")
    _check(generation["identity"]["protocol_sha256"] == protocol_sha, "Generation protocol differs")
    _check(feature_receipt["identity"]["protocol_sha256"] == protocol_sha, "Feature protocol differs")
    _verified(folder / "frames.jsonl", generation["identity"]["capture_sha256"])
    _verified(folder / "metadata.json", generation["identity"]["capture_metadata_sha256"])
    _verified(metadata_path, generation["output_echo_metadata_sha256"])
    _verified(frames_path, generation["output_echo_frames_sha256"])
    _verified(generation_path, feature_receipt["identity"]["generation_receipt_sha256"])
    capture, echoes = _rows(folder / "frames.jsonl"), _rows(frames_path)
    sources = feature_receipt["frames"]
    _check(len(capture) == len(echoes) == len(sources) == generation["frames"], "Frame inventory differs")
    rf = _rf(metadata)
    station = np.asarray(metadata["station_position_ned_m"], dtype=float)
    basis = np.asarray(metadata["array_basis_in_ned"], dtype=float)
    _check(basis.shape == (3, 3) and np.allclose(basis.T @ basis, np.eye(3), atol=1e-10), "Invalid public array calibration")
    manifest_frames, replay_frames, proof_frames = [], [], []
    stats = dict(iq_sha_recomputed=0, iq_prior_verified_stat_reused=0, iq_legacy_receipt_only=0, rgb_sha_recomputed=0, max_detected_members=0, empty_detection_frames=0)
    for index, (cap, echo, row) in enumerate(zip(capture, echoes, sources)):
        _check(cap["frame_idx"] == echo["frame_idx"] == row["frame_idx"] == index, "Non-consecutive native frames")
        timestamp = float(echo["physical_time_s"])
        _check(timestamp == float(index), "Unexpected physical time; simulator clock is never used")
        _check(cap["camera"]["camera_position_ned"] == echo["observer_reference"]["position_ned_m"], "Camera/scatter centre differs")
        _check(row["hashes"]["source_trajectory"] == generation["identity"]["capture_sha256"], "Feature capture identity differs")
        _check(row["hashes"]["source_metadata"] == generation["identity"]["capture_metadata_sha256"], "Feature capture metadata differs")
        cache = source.branch / "features" / episode / f"frame_{index:05d}.pt"
        _verified(cache, row["camera_feature_sha256"])
        stored = torch.load(cache, map_location="cpu", weights_only=False)
        _check(stored["receipt"] == row["branch_identity"], "Feature internal receipt differs")
        f = stored["features"]
        stats["max_detected_members"] = max(stats["max_detected_members"], len(f["bbox_xyxy"]))
        stats["empty_detection_frames"] += int(len(f["bbox_xyxy"]) == 0)
        branch = f["metadata"]["camera_branch"]
        iq_sha = echo["iq_sha256"]
        _check(iq_sha == row["hashes"]["iq"] == stored["receipt"]["iq_sha256"] == branch["iq_sha256"], "New IQ binding differs")
        _check(branch["echo_frames_sha256"] == generation["output_echo_frames_sha256"] and
               branch["echo_metadata_sha256"] == generation["output_echo_metadata_sha256"], "Cached echo provenance differs")
        _check(f["metadata"]["spectrum"]["version"] == "rectangular_native_ra_re_v1", "Uncertified RA/RE kernel")
        sensor = f["metadata"]["sensor_identity"]
        detector_sha = sensor["detector_contract"]["weights_sha256"]
        rgb = folder / cap["rgb_path"]
        _check(rgb.resolve().is_relative_to(folder.resolve()), "RGB path escaped episode")
        _verified(rgb, row["hashes"]["rgb"])
        stats["rgb_sha_recomputed"] += 1
        _check(sensor["rgb_sha256"] == row["hashes"]["rgb"], "Cached RGB identity differs")
        iq = folder / "echoes" / echo["iq_path"]
        _check(iq.resolve().is_relative_to((folder / "echoes").resolve()), "IQ path escaped episode")
        frame_dir = output / episode / f"frame_{index:05d}"
        proof_path = frame_dir / "source_proof.json"
        iq_stat = _stat(iq)
        old_proof = _read(proof_path) if proof_path.exists() else None
        reused = bool(old_proof and old_proof.get("raw_iq_verified") and old_proof.get("iq_stat") == iq_stat and old_proof.get("iq_sha256") == iq_sha)
        if verify_iq and not reused:
            _verified(iq, iq_sha)
            _check(_stat(iq) == iq_stat, "IQ changed during hash verification")
            native = np.load(iq, mmap_mode="r", allow_pickle=False)
            _check(native.shape == rf.shape and native.dtype == np.complex64, "Wrong native IQ dtype/shape")
            del native
            stats["iq_sha_recomputed"] += 1
        elif verify_iq:
            stats["iq_prior_verified_stat_reused"] += 1
        else:
            stats["iq_legacy_receipt_only"] += 1
        visual = dict(schema=np.array(VISUAL_SCHEMA), bbox_xyxy=f["bbox_xyxy"],
                      detection_scores=f["detection_scores"], detector_roi_features=f["detector_roi_features"],
                      image_size_wh=f["camera_image_size_wh"], source_rgb_sha256=np.array(row["hashes"]["rgb"]),
                      detector_sha256=np.array(detector_sha))
        visual = validate_visual_arrays(visual, f["camera_image_size_wh"])
        radar = {key: f[key] for key in RADAR_ARRAY_FIELDS}
        radar.update(schema=np.array(RADAR_SCHEMA), source_iq_sha256=np.array(iq_sha),
                     rf_json=np.array(json.dumps(rf.as_dict(), sort_keys=True, allow_nan=False)))
        radar = validate_radar_arrays(radar, rf, iq_sha)
        visual_path, radar_path = frame_dir / "visual.npz", frame_dir / "radar.npz"
        visual_sha, radar_sha = _publish_npz(visual_path, visual), _publish_npz(radar_path, radar)
        normal = _public_frame(episode=episode, capture=capture, index=index, timestamp=timestamp, features=f,
            station=station, basis=basis, visual_ref=dict(path=_relative(visual_path, source), sha256=visual_sha),
            radar_ref=dict(path=_relative(radar_path, source), sha256=radar_sha),
            iq_ref=dict(path=_relative(iq, source), sha256=iq_sha))
        normal_path, labels_path = frame_dir / "normal.json", frame_dir / "labels.json"
        normal_sha = _publish_json(normal_path, normal)
        labels_sha = _publish_json(labels_path, _offline_labels(episode, index, timestamp, echo, station, basis))
        proof = dict(schema="rvloc.camera_center_source_proof.v1", episode_id=episode, frame_idx=index,
            source_cache=dict(path=str(cache), sha256=row["camera_feature_sha256"]),
            source_rgb=dict(path=str(rgb), sha256=row["hashes"]["rgb"]),
            original_roi_receipt=row["detector_roi_cache"], original_source_feature=row["source_feature_cache"],
            cache_internal_receipt=stored["receipt"], iq_path=_relative(iq, source), iq_sha256=iq_sha,
            iq_stat=iq_stat, raw_iq_verified=bool(verify_iq or reused),
            visual_sha256=visual_sha, radar_sha256=radar_sha, normal_sha256=normal_sha, labels_sha256=labels_sha,
            sensor_export_labels_read=False, supervision_export_labels_read=True)
        # A prior weak export cannot be silently upgraded in its immutable directory.
        _publish_json(proof_path, proof)
        proof_frames.append(dict(path=_relative(proof_path, source), sha256=sha256(proof_path)))
        manifest_frames.append(dict(frame_idx=index, physical_time_s=timestamp,
            normal_path=_relative(normal_path, source), normal_sha256=normal_sha,
            labels_path=_relative(labels_path, source), labels_sha256=labels_sha))
        replay_frames.append({key: proof[key] for key in ("episode_id", "frame_idx", "normal_sha256", "iq_path", "iq_sha256", "iq_stat", "visual_sha256", "radar_sha256")})
    episode_proof = dict(schema="rvloc.camera_center_episode_export.v1", episode_id=episode, split=split,
        generation_receipt=dict(path=str(generation_path), sha256=sha256(generation_path), content=generation),
        feature_receipt=dict(path=str(features_path), sha256=sha256(features_path), content=feature_receipt),
        frames=proof_frames)
    _publish_json(output / episode / "export_receipt.json", episode_proof)
    print(json.dumps(dict(stage="register", episode_id=episode, split=split, frames=len(capture),
                         wall_s=round(time.perf_counter()-start, 3), **stats)), flush=True)
    return dict(episode_id=episode, frames=manifest_frames), replay_frames, rf.as_dict(), stats, f["metadata"]["spectrum"]["version"]


def register_camera_center(output_dir, *, verify_iq=True, workers=4):
    """Export a new strict manifest; return manifest/receipt/runtime_receipt paths.

    ``verify_iq=True`` hashes native IQ on first registration. Completed frame
    proofs may be reused only with matching SHA claims and unchanged full stat.
    ``False`` is explicitly receipt-only and cannot certify prepared-only replay.
    """
    start = time.perf_counter()
    _check(type(workers) is int and 1 <= workers <= 32, "workers must be in [1,32]")
    _check(type(verify_iq) is bool, "verify_iq must be boolean")
    source = _default_source()
    output = Path(output_dir).expanduser().resolve()
    _relative(output, source)
    _check(not output.is_relative_to(source.branch.resolve()), "Export must not modify the source branch")
    protocol_path = source.results / "protocol.json"
    _verified(protocol_path, source.protocol_sha256)
    protocol = _read(protocol_path)
    splits = dict(train=protocol["selection"]["train"], dev=protocol["selection"]["val"])
    _check(tuple(len(splits[k]) for k in ("train", "dev")) == source.expected_counts, "Unexpected frozen TRAIN/DEV size")
    _check(not set(splits["train"]) & set(splits["dev"]), "TRAIN/DEV overlap")
    _check(len(set(sum(splits.values(), []))) == sum(source.expected_counts), "Duplicate frozen episode")
    output.mkdir(parents=True, exist_ok=True)
    immutable_plan = dict(schema="rvloc.camera_center_registration_plan.v1", source_branch=str(source.branch),
        protocol_path=str(protocol_path), protocol_sha256=source.protocol_sha256,
        verify_iq=verify_iq, splits=splits, visual_geometry=GEOMETRY,
        navigation_velocity="causal backward difference of public body navigation; first sample zero",
        normal_inference_target_fields=False, historical_TEST_payload_read=False)
    _publish_json(output / "registration_plan.json", immutable_plan)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_export_episode, episode, split, source, output,
                                  source.protocol_sha256, verify_iq): episode
                   for split, ids in splits.items() for episode in ids}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    episodes, replay, totals = [], [], dict(iq_sha_recomputed=0, iq_prior_verified_stat_reused=0, iq_legacy_receipt_only=0, rgb_sha_recomputed=0, max_detected_members=0, empty_detection_frames=0)
    rf = next(iter(results.values()))[2]
    for episode in sum(splits.values(), []):
        record, rows, current_rf, stats, kernel = results[episode]
        _check(current_rf == rf and kernel == "rectangular_native_ra_re_v1", "RF/kernel changed across episodes")
        episodes.append(record)
        replay.extend(rows)
        for key in totals:
            totals[key] = max(totals[key], stats[key]) if key == "max_detected_members" else totals[key] + stats[key]
    evidence = dict(schema="rvloc.center_evidence.v1", coordinate_frame="base_local",
        reference="observer_self_scatter_reference", co_location="camera_optical_center",
        camera_to_scatter_reference_offset_m=[0., 0., 0.],
        source=f"All {len(replay)} frame poses verified against certified camera-centre echo schema; protocol {source.protocol_sha256}",
        capture_revision=source.protocol_sha256)
    evidence_path = output / "center_evidence.json"
    evidence_sha = _publish_json(evidence_path, evidence)
    replay_receipt = dict(schema="rvloc.prepared_replay.v1", raw_iq_verified=verify_iq,
        sensor_export_labels_read=False, frames=replay,
        fft_source=dict(kind="authenticated_camera_center_cache_replay", kernel="rectangular_native_ra_re_v1",
            sources={"protocol": source.protocol_sha256, "exporter": sha256(__file__),
                     "episode_exports": hashlib.sha256(_json_bytes({ep: sha256(output / ep / "export_receipt.json")
                         for ep in sorted(results)})).hexdigest()}))
    replay_path = output / "prepared_replay_receipt.json"
    replay_sha = _publish_json(replay_path, replay_receipt)
    manifest = dict(schema="rvloc.manifest.v1", status="ready", coordinate_frame="base_local", data_root=str(source.data_root),
        rf=rf, visual_geometry=GEOMETRY, center_contract=dict(reference="observer_self_scatter_reference",
        co_location="camera_optical_center", status="verified", evidence=dict(path=_relative(evidence_path, source), sha256=evidence_sha)),
        splits=splits, episodes=episodes, prepared_replay=dict(path=_relative(replay_path, source), sha256=replay_sha))
    manifest_path = output / "manifest.json"
    manifest_sha = _publish_json(manifest_path, manifest)
    # Parser validation remains metadata-only and never opens historical TEST.
    load_manifest(manifest_path)
    receipt_path = output / "registration_receipt.json"
    receipt = dict(schema="rvloc.camera_center_registration_receipt.v1", status="complete",
        manifest_path=str(manifest_path), manifest_sha256=manifest_sha, prepared_replay_sha256=replay_sha,
        source_plan=immutable_plan, episodes=len(episodes), frames=len(replay),
        split_frames={split: sum(len(results[ep][0]["frames"]) for ep in ids) for split, ids in splits.items()},
        max_detected_members=totals["max_detected_members"], empty_detection_frames=totals["empty_detection_frames"],
        all_camera_scatter_centers_equal=True, sensor_export_labels_read=False, supervision_export_labels_read=True,
        raw_iq_verified=verify_iq, raw_IQ_fft_recomputed=False, cached_RA_RE_replayed=True,
        default_model_or_weights_changed=False, historical_TEST_payload_read=False)
    _publish_json(receipt_path, receipt)
    runtime_path = output / f"registration_runtime_{time.time_ns()}.json"
    _publish_json(runtime_path, dict(wall_s=time.perf_counter()-start, workers=workers, **totals,
                                    receipt_sha256=sha256(receipt_path)))
    result = dict(manifest=str(manifest_path), receipt=str(receipt_path), runtime_receipt=str(runtime_path))
    print(json.dumps(dict(stage="registration_complete", **result, **totals)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--receipt-only-iq", action="store_true", help="Do not freshly authenticate IQ; cannot enable prepared-only replay")
    args = parser.parse_args()
    register_camera_center(args.output_dir, verify_iq=not args.receipt_only_iq, workers=args.workers)


if __name__ == "__main__":
    main()
