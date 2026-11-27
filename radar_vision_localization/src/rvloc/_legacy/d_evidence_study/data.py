"""Single-frame D evidence inputs and authenticated frozen-detector ROI cache.

All radar/geometry features are read from the completed 32 MHz BW control
cache. Cache misses fail; this module never regenerates that cache. Labels
are opt-in, separate, current-frame supervision. Final test access is disabled.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bandwidth_study import data as bw
from depth_localization.features import (drop_detection_features as drop_base_features,
    feature_digest, validate_depth_features)
from rvwm.data import Dataset, sha256
from rvwm.detector_contract import detector_contract
from rvwm.group_model import match_ray_labels
from rvwm.partitions import canonical_manifest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/bandwidth_double_20261124"
ARMS = ("control", "sequence", "aligned", "roi")
ROI_SPEC = dict(schema="frozen_dinov3_box_head_roi_v1", dimension=1024,
    source_layer="roi_heads.box_head", pooling="native MultiScaleRoIAlign then native TwoMLPHead",
    pool_output_size=[7, 7], pool_sampling_ratio=2, feature_dtype="float32", autocast=False,
    boxes="unchanged frozen detected XYXY boxes, original RGB pixels; resized only by native transform")
SENSOR_FIELDS = frozenset(("schema", "depth_schema", "ray_origins", "ray_directions", "ray_valid",
    "ray_depth_bounds", "observer_position", "range_scale_m", "ra_power", "re_power", "ra_valid", "re_valid",
    "range_m", "azimuth_rad", "elevation_rad", "metadata", "depth_metadata", "bbox_xyxy", "detection_scores",
    "camera_K", "camera_image_size_wh", "visual_scale"))
_FILE_HASHES = {}
_SOURCE_NAMES = None


def verified_file_sha256(path):
    """Reuse a verified immutable file digest until its full stat identity changes."""
    path = Path(path).resolve(); s = path.stat()
    identity = (str(path), s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev)
    if identity not in _FILE_HASHES:
        _FILE_HASHES[identity] = sha256(path)
    return _FILE_HASHES[identity]


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    global _SOURCE_NAMES
    if _SOURCE_NAMES is None:
        _SOURCE_NAMES = tuple("d_evidence_study/"+name for name in ("data.py", "roi_export.py")) + tuple(bw.source_hashes())
    return {name: verified_file_sha256(ROOT / name) for name in _SOURCE_NAMES}


def sensor_config(config):
    value = bw.arm_config(read(config.get("d_evidence_sensor_config_path", PARENT / "configs/baseline.json")))
    if value["bandwidth_arm"] != "baseline" or value["bandwidth_hz"] != 32e6:
        raise ValueError("This evidence study is fixed to the registered original 32 MHz observations")
    for key, original in value.items():
        if (key in {"dataset_root", "echo_subdir", "split_manifest", "split_manifest_sha256", "visual_split_manifest",
                    "track_detection_dir", "training_phases", "inference_phases", "bandwidth_hz", "bandwidth_frequency_points"}
                or key.startswith(("ra_re_", "detector_"))):
            if key in config and config[key] != original:
                raise ValueError("The evidence study changed its registered sensor recipe: " + key)
    if config.get("d_evidence_arm", "control") not in ARMS or config.get("window", 1) != 1:
        raise ValueError("Explicit evidence arm and a single current observation are required")
    if config.get("d_evidence_roi_dim", 1024) != 1024:
        raise ValueError("Frozen native box_head output has exactly 1024 channels")
    return value


def parent_preparation(config, base):
    path = Path(config.get("d_evidence_sensor_preparation", PARENT / "prepare/baseline")).resolve()
    receipt = bw.validate_preparation(base, path, verify_cache=False)
    if receipt.get("max_episodes") or receipt.get("frames_per_episode") or set(receipt["requested_splits"]) != {"train", "val"}:
        raise ValueError("Require the complete original train/val sensor preparation")
    return path, receipt


def _single(features):
    if set(features) - SENSOR_FIELDS:
        raise ValueError("Unexpected non-sensor, history, ROI or truth field in base cache")
    return validate_depth_features(features)


class SensorCache:
    """Read-only sensor lookup shared by the loader and the ROI exporter."""
    def __init__(self, config, split):
        if split not in ("train", "val"):
            raise ValueError("This experiment never opens final test observations or labels")
        self.config, self.split = sensor_config(config), split
        path, receipt = parent_preparation(config, self.config)
        self.reader = bw.Reader(self.config, split)
        rows = read(receipt["inventories"][split]["path"])["frames"]
        self.rows = {(r["episode_id"], r["frame_idx"]): r for r in rows}
        if len(rows) != len(self.rows):
            raise ValueError("Duplicate registered sensor frames")
        self.metadata = dict(self.reader.metadata, d_evidence_source_sha256=source_hashes(),
            original_sensor_preparation=str(path), original_sensor_preparation_sha256=sha256(path / "preparation_receipt.json"),
            sensor_cache_written=False, history_read=False, future_read=False, gt_identity_used=False)
        self.memo = {}

    def close(self):
        self.reader.close()

    def ids(self):
        return sorted(self.reader.visual.ids)

    def indices(self, episode, count=0):
        return self.reader.indices(episode, count)

    def release(self):
        self.memo.clear()

    def frame(self, episode, index):
        key = episode, index
        if key in self.memo:
            return self.memo[key]
        if key not in self.rows:
            raise ValueError("Frame lacks an existing sensor cache: " + str(key))
        row = self.rows[key]
        identity = self.reader.episode(episode)
        sensor = self.reader.measured.sensor_frame(episode, index)
        path = Path(row["cache"]["path"])
        if (not path.resolve().is_relative_to(Path(self.config["bandwidth_feature_cache"]))
                or not path.is_file() or sha256(path) != row["cache"]["sha256"]
                or bw._stamp(sensor["iq_path"]) != row["iq_stat"]
                or sensor["iq_sha256"] != row["hashes"]["iq"]
                or identity["echo_metadata_sha256"] != row["echo_metadata_sha256"]
                or identity["echo_frames_sha256"] != row["echo_frames_sha256"]):
            raise ValueError("Sensor cache/source changed; no rebuild is permitted: " + str(key))
        item = torch.load(path, map_location="cpu", weights_only=False)
        f, source = _single(item["features"]), item["receipt"]
        if (bw._digest(source) != path.stem or source["source_sha256"] != bw.source_hashes()
                or source["frame_idx"] != index or source["episode"]["episode_id"] != episode
                or feature_digest(f) != item["feature_digest"] or item["feature_digest"] != row["feature_digest"]):
            raise ValueError("Sensor feature identity changed: " + str(key))
        self.memo[key] = f, row, sensor
        return self.memo[key]


def roi_cache_root(config):
    if not config.get("d_evidence_roi_cache"):
        raise ValueError("An independent d_evidence_roi_cache is required for the ROI arm")
    path = Path(config["d_evidence_roi_cache"]).resolve()
    base = sensor_config(config)
    forbidden = (Path(base["dataset_root"]).resolve(), Path(base["bandwidth_feature_cache"]).resolve(),
                 PARENT.resolve(), (ROOT / "results/temporal_depth_20261124").resolve())
    if not path.is_relative_to(ROOT / "results") or any(path == p or path.is_relative_to(p) for p in forbidden):
        raise ValueError("ROI cache must be an independent result directory")
    return path


def roi_identity(config, features, row, sensor):
    base = sensor_config(config)
    return dict(spec=ROI_SPEC, source_sha256=source_hashes(),
        detector_contract=detector_contract(base), episode_id=row["episode_id"], frame_idx=row["frame_idx"],
        rgb_path=str(sensor["rgb_path"]), rgb_sha256=row["hashes"]["rgb"], rgb_stat=bw._stamp(sensor["rgb_path"]),
        image_size_wh=np.asarray(features["camera_image_size_wh"]).tolist(),
        bbox_xyxy=np.asarray(features["bbox_xyxy"]).tolist(),
        detection_scores=np.asarray(features["detection_scores"]).tolist(),
        bbox_order="exact existing frozen detector order; no GT matching or sorting", sensor_cache=row["cache"],
        feature_digest=row["feature_digest"], split_manifest_sha256=base["split_manifest_sha256"])


def roi_paths(config, episode, index):
    folder = roi_cache_root(config) / episode
    return folder / f"frame_{index:05d}.npz", folder / f"frame_{index:05d}.json"


def load_roi(config, features, row, sensor):
    path, receipt_path = roi_paths(config, row["episode_id"], row["frame_idx"])
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Frozen ROI cache missing; run the separate exporter first: " + str(path))
    receipt = read(receipt_path)
    if receipt.get("identity") != roi_identity(config, features, row, sensor) or receipt.get("payload_sha256") != sha256(path):
        raise ValueError("ROI cache inputs, order, detector or exporter changed")
    for source, expected in receipt.get("backend_source_sha256", {}).items():
        if verified_file_sha256(source) != expected:
            raise ValueError("Frozen native ROIAlign/transform implementation changed: " + source)
    if not receipt.get("backend_source_sha256") or receipt.get("eval_mode") is not True or receipt.get("inference_mode") is not True:
        raise ValueError("ROI cache lacks an authenticated frozen inference receipt")
    with np.load(path, allow_pickle=False) as stored:
        value = stored["features"]
        if (value.dtype != np.float32 or value.shape != (len(features["bbox_xyxy"]), 1024)
                or not np.isfinite(value).all() or not np.array_equal(stored["boxes"], features["bbox_xyxy"])
                or not np.array_equal(stored["scores"], features["detection_scores"])):
            raise ValueError("ROI features are not aligned to the unchanged detected boxes")
    if len(value) and np.any(np.linalg.norm(value, axis=1) == 0.):
        raise ValueError("Degenerate zero ROI vectors are not a valid replacement for detector features")
    return value, dict(path=str(path), sha256=receipt["payload_sha256"], receipt_path=str(receipt_path),
                       receipt_sha256=sha256(receipt_path), dimension=1024)


def drop_detection_features(features, keep):
    result = drop_base_features(features, keep)
    if "detector_roi_features" in features:
        roi = np.asarray(features["detector_roi_features"])
        if roi.shape != (len(keep), 1024) or not np.isfinite(roi).all():
            raise ValueError("Misaligned ROI features before detection dropout")
        result["detector_roi_features"] = roi[np.asarray(keep)].copy()
    return result


def _sample(config, reader, episode, index, include_labels, labels):
    features, original, sensor = reader.frame(episode, index)
    row = deepcopy(original)
    if config.get("d_evidence_arm", "control") == "roi":
        roi, receipt = load_roi(config, features, row, sensor)
        features = dict(features, detector_roi_features=roi)
        row["detector_roi_cache"] = receipt
    result = dict(features=features, episode_id=episode, frame_idx=index)
    if include_labels:
        truth = labels.targets(episode, index)
        positions = sensor["station_frame"].to_local_positions(truth.positions).astype(np.float32)
        correspondence = match_ray_labels(features, positions, config)
        if not correspondence["diagnostics"]["supervision_usable"]:
            raise ValueError("No usable current supervision; refusing to drop frame: " + episode + "/" + str(index))
        row.update(label_count=len(positions), supervision=correspondence["diagnostics"])
        result.update(positions=positions, formation_index=None,
            complete_detection_cardinality=len(features["ray_directions"]) == len(positions))
    return result, row


def load_samples(config, ids=None, split=None, count=0, include_labels=True, *, limit=None):
    """Trainer API; ``load_samples(config, 'train', limit=N)`` is also supported."""
    if isinstance(ids, str) and split is None:
        split, ids = ids, None
    if split not in ("train", "val"):
        raise ValueError("This development study never reads final test data")
    if type(include_labels) is not bool or type(count) is not int or count < 0 or (limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError("Explicit label access and valid nonnegative sample limits required")
    reader = SensorCache(config, split)
    dataset = Dataset(reader.config["dataset_root"], echo_subdir=reader.config.get("echo_subdir", "echoes")) if include_labels else None
    samples, rows = [], []
    try:
        ids = list(reader.ids() if ids is None else ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode IDs")
        for episode in ids:
            for index in reader.indices(episode, count):
                sample, row = _sample(config, reader, episode, index, include_labels, dataset)
                samples.append(sample); rows.append(row)
                if limit is not None and len(samples) >= limit:
                    return samples, rows, dict(reader.metadata, roi_enabled=config.get("d_evidence_arm") == "roi")
        return samples, rows, dict(reader.metadata, roi_enabled=config.get("d_evidence_arm") == "roi")
    finally:
        reader.close()


def load_observation_features(config, episode, frame_idx, split):
    if split not in ("train", "val"):
        raise ValueError("This study does not open final test observations")
    reader = SensorCache(config, split)
    try:
        if frame_idx not in reader.indices(episode):
            raise ValueError("Current frame is outside its fixed split/phase inventory")
        return _sample(config, reader, episode, frame_idx, False, None)[0]["features"]
    finally:
        reader.close()


def preparation_identity(config):
    base = sensor_config(config); path, _ = parent_preparation(config, base)
    roi = config.get("d_evidence_arm") == "roi"
    return dict(schema="d_evidence_preparation_v1", source_sha256=source_hashes(),
        sensor_configuration_sha256=digest(base), parent_preparation_sha256=sha256(path / "preparation_receipt.json"),
        split_manifest_sha256=base["split_manifest_sha256"], roi_enabled=roi,
        roi_cache=str(roi_cache_root(config)) if roi else None, mode_independent_sensor=True,
        labels_read=False, history_read=False, future_read=False)


def _prepare_shard(config, split, ids, count):
    torch.set_num_threads(1)
    reader = SensorCache(config, split); rows = []
    try:
        for episode in ids:
            for index in reader.indices(episode, count):
                rows.append(_sample(config, reader, episode, index, False, None)[1])
            reader.release()
        return split, rows
    finally:
        reader.close()


def validate_preparation(config, output, *, verify_cache=False):
    output = Path(output).resolve(); receipt = read(output / "preparation_receipt.json")
    expected = preparation_identity(config)
    if receipt.get("status") != "complete" or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Evidence preparation identity changed")
    base = sensor_config(config); manifest = canonical_manifest(base)[1]
    episodes = {r["episode"]: r for r in manifest["episodes"]}
    for split, spec in receipt["inventories"].items():
        path = Path(spec["path"])
        if split not in ("train", "val") or not path.resolve().is_relative_to(output) or sha256(path) != spec["sha256"]:
            raise ValueError("Evidence inventory changed")
        ids = sorted(manifest["splits"][split]["episode_ids"])
        if receipt["max_episodes"]:
            ids = ids[:receipt["max_episodes"]]
        expected_keys = set()
        for episode in ids:
            indices = episodes[episode]["selected_frame_indices"]
            count = receipt["frames_per_episode"]
            if count:
                indices = [indices[i] for i in np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)]
            expected_keys.update((episode, int(i)) for i in indices)
        rows = read(path)["frames"]
        keys = [(r["episode_id"], r["frame_idx"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys or len(rows) != spec["frames"]:
            raise ValueError("Evidence preparation dropped or duplicated anchors")
        if verify_cache:
            for row in rows:
                if sha256(row["cache"]["path"]) != row["cache"]["sha256"]:
                    raise ValueError("Prepared sensor cache changed")
                if receipt["roi_enabled"]:
                    r = row["detector_roi_cache"]
                    if sha256(r["path"]) != r["sha256"] or sha256(r["receipt_path"]) != r["receipt_sha256"]:
                        raise ValueError("Prepared ROI cache changed")
    return receipt


def prepare(config, output, *, splits=("train", "val"), workers=1, max_episodes=0, count=0):
    if not splits or any(s not in ("train", "val") for s in splits) or min(workers, max_episodes+1, count+1) < 1:
        raise ValueError("Only explicit train/val preparation with valid limits is supported")
    output = Path(output).resolve(); identity = preparation_identity(config)
    if (output / "preparation_receipt.json").exists():
        prior = validate_preparation(config, output)
        if prior["requested_splits"] != list(splits) or prior["max_episodes"] != max_episodes or prior["frames_per_episode"] != count:
            raise ValueError("Existing preparation used another sample scope")
        return prior
    started = time.monotonic(); inventories = {}
    manifest = canonical_manifest(sensor_config(config))[1]
    jobs = []
    for split in splits:
        ids = sorted(manifest["splits"][split]["episode_ids"])
        if max_episodes:
            ids = ids[:max_episodes]
        jobs.extend((split, ids[i::workers]) for i in range(min(workers, len(ids))))
    grouped = {split: [] for split in splits}; done = 0
    def record(value):
        nonlocal done
        split, rows = value; grouped[split].extend(rows); done += 1
        bw.atomic_json(output / "progress.json", dict(stage="preparing", completed_shards=done,
            total_shards=len(jobs), frames=sum(map(len, grouped.values())), elapsed_s=time.monotonic()-started))
    if workers == 1:
        for split, ids in jobs:
            record(_prepare_shard(config, split, ids, count))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(_prepare_shard, config, split, ids, count) for split, ids in jobs]
            for future in as_completed(pending):
                record(future.result())
    for split, rows in grouped.items():
        rows.sort(key=lambda row: (row["episode_id"], row["frame_idx"]))
        path = output / "inventories" / (split + ".json")
        bw.atomic_json(path, dict(frames=rows, labels_read=False))
        inventories[split] = dict(path=str(path), sha256=sha256(path), frames=len(rows), episodes=len({row["episode_id"] for row in rows}))
    receipt = dict(identity, status="complete", requested_splits=list(splits), max_episodes=max_episodes,
        frames_per_episode=count, inventories=inventories, elapsed_s=time.monotonic()-started,
        workers_requested=workers, workers_used=workers)
    bw.atomic_json(output / "preparation_receipt.json", receipt)
    validate_preparation(config, output)
    bw.atomic_json(output / "progress.json", dict(stage="complete", **receipt))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    p.add_argument("--workers", type=int, default=1); p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--frames-per-episode", type=int, default=0)
    a = p.parse_args(argv)
    print(json.dumps(prepare(read(a.config), a.output, splits=tuple(a.splits), workers=a.workers,
                            max_episodes=a.max_episodes, count=a.frames_per_episode), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
