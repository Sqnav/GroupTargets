"""Export real frozen DINOv3/FPN/Faster-RCNN features at existing detections.

Run with uav_detection/.venv_deimv2/bin/python. No new detection, RPN, NMS,
ground-truth crop, augmentation, detector fine-tuning or IQ processing occurs.
The stored vector is the native box_head output after repooling the unchanged
final detected boxes, rather than a claim to recover old RPN proposal tensors.
"""
from __future__ import annotations

import argparse
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from .data import (ROI_SPEC, ROOT, SensorCache, digest, load_roi, read, roi_cache_root,
                   roi_identity, roi_paths, sensor_config, sha256, source_hashes, canonical_manifest)
from bandwidth_study.data import atomic_json


class Extractor:
    """Frozen native transform/backbone/ROIAlign/TwoMLPHead, without detection."""
    def __init__(self, config, device="cpu", threads=2):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from uav_detection.dinov3_detector import DINOv3Detector
        import torchvision
        from torchvision.models.detection import transform, roi_heads
        from torchvision.ops import poolers
        torch.set_num_threads(max(1, int(threads)))
        base = sensor_config(config)
        self.device = torch.device(device)
        self.detector = DINOv3Detector(base["detector_weights"], device=device,
                                       confidence=base["detector_confidence"])
        self.model = self.detector.model.float().eval().requires_grad_(False)
        self.resize_boxes = transform.resize_boxes
        actual = self.model.transform
        expected = config.get("detector_min_size", 720), config.get("detector_max_size", 1280)
        if (tuple(actual.min_size) != (expected[0],) or actual.max_size != expected[1]
                or actual.size_divisible != 32
                or list(actual.image_mean) != [.485, .456, .406]
                or list(actual.image_std) != [.229, .224, .225]
                or tuple(self.model.roi_heads.box_roi_pool.output_size) != (7, 7)
                or self.model.roi_heads.box_roi_pool.sampling_ratio != 2):
            raise ValueError("Native frozen detector preprocessing/ROI pooling contract changed")
        sources = [Path(inspect.getfile(module)).resolve() for module in (transform, roi_heads, poolers)]
        extension = Path(torchvision.__file__).parent / "_C.so"
        if not extension.is_file():
            raise FileNotFoundError("The native torchvision ROIAlign extension is required")
        sources.append(extension)
        self.backend_sources = {str(path): sha256(path) for path in sources}
        self.backend = dict(torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
            device=str(self.device), dtype="float32", autocast=False, frozen=True,
            native_transform=dict(min_size=list(actual.min_size), max_size=actual.max_size,
                size_divisible=actual.size_divisible, image_mean=list(actual.image_mean), image_std=list(actual.image_std)))

    @torch.inference_mode()
    def extract(self, rgb, boxes):
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or np.asarray(boxes).ndim != 2 or np.asarray(boxes).shape[1:] != (4,)):
            raise ValueError("Expected an actual uint8 RGB frame and detected XYXY boxes")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("The native detector must remain frozen and in eval mode")
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).to(self.device)
        original_size = tuple(rgb.shape[:2])
        started = time.perf_counter()
        images, _ = self.model.transform([tensor], None)
        resized = self.resize_boxes(torch.as_tensor(np.asarray(boxes), dtype=torch.float32, device=self.device),
                                    original_size, images.image_sizes[0])
        maps = self.model.backbone(images.tensors)
        pooled = self.model.roi_heads.box_roi_pool(maps, [resized], images.image_sizes)
        vectors = self.model.roi_heads.box_head(pooled)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter()-started
        value = vectors.float().cpu().numpy().copy()
        if value.shape != (len(boxes), 1024) or not np.isfinite(value).all():
            raise ValueError("Unexpected native frozen box_head features")
        if len(value) and np.any(np.linalg.norm(value, axis=1) == 0.):
            raise ValueError("The native detector produced degenerate zero ROI vectors")
        return value, dict(extraction_s=elapsed, original_size_hw=list(original_size),
            transformed_size_hw=list(images.image_sizes[0]), padded_size_hw=list(images.tensors.shape[-2:]),
            pooled_shape=list(pooled.shape), output_shape=list(value.shape),
            resized_bbox_xyxy=resized.cpu().tolist(), eval_mode=True, inference_mode=True,
            feature_l2_min=float(np.linalg.norm(value, axis=1).min()) if len(value) else None)


def _persist(path, vectors, boxes, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, features=vectors, boxes=np.asarray(boxes), scores=np.asarray(scores))
    raw = buffer.getvalue()
    checksum = __import__("hashlib").sha256(raw).hexdigest()
    if path.exists():
        if sha256(path) != checksum:
            raise ValueError("Conflicting orphan ROI payload retained: " + str(path))
        return checksum
    fd, name = tempfile.mkstemp(prefix=".roi_", suffix=".npz.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return checksum


def _selected_keys(config, splits, max_episodes, count):
    manifest = canonical_manifest(sensor_config(config))[1]
    inventory = {r["episode"]: r for r in manifest["episodes"]}
    keys = []
    for split in splits:
        ids = sorted(manifest["splits"][split]["episode_ids"])
        if max_episodes:
            ids = ids[:max_episodes]
        for ep in ids:
            indices = inventory[ep]["selected_frame_indices"]
            if count:
                indices = [indices[i] for i in np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)]
            keys.extend((split, ep, int(i)) for i in indices)
    return keys


def validate_export(config, output, *, verify_cache=False):
    """Root-venv validation: torchvision/native model imports stay deferred."""
    output = Path(output).resolve()
    config = dict(config, d_evidence_roi_cache=str(output / "cache"))
    receipt = read(output / "export_receipt.json")
    if (receipt.get("status") != "complete" or receipt.get("source_sha256") != source_hashes()
            or receipt.get("spec") != ROI_SPEC or receipt.get("sensor_configuration_sha256") != digest(sensor_config(config))
            or receipt.get("cache") != str(roi_cache_root(config))):
        raise ValueError("Frozen ROI export identity changed")
    path = Path(receipt["inventory"]["path"])
    if not path.resolve().is_relative_to(output) or sha256(path) != receipt["inventory"]["sha256"]:
        raise ValueError("ROI export inventory changed")
    rows = read(path)["frames"]
    expected = _selected_keys(config, receipt["requested_splits"], receipt["max_episodes"], receipt["frames_per_episode"])
    keys = [(r["split"], r["episode_id"], r["frame_idx"]) for r in rows]
    if keys != expected or len(keys) != receipt["frames"]:
        raise ValueError("ROI export dropped, duplicated or reordered frozen frames")
    for row in rows:
        payload, meta = roi_paths(config, row["episode_id"], row["frame_idx"])
        if str(payload) != row["path"] or str(meta) != row["receipt_path"] or sha256(meta) != row["receipt_sha256"]:
            raise ValueError("ROI frame receipt changed")
        frame = read(meta)
        if frame["identity"]["source_sha256"] != source_hashes() or frame["payload_sha256"] != row["sha256"]:
            raise ValueError("ROI frame source identity changed")
        if verify_cache and sha256(payload) != row["sha256"]:
            raise ValueError("ROI payload changed")
    return receipt


def export(config, output, *, splits=("train", "val"), device="cpu", threads=2,
           max_episodes=0, count=0):
    if not splits or any(s not in ("train", "val") for s in splits) or len(set(splits)) != len(splits):
        raise ValueError("Only explicit train/val ROI export is allowed")
    if min(threads, max_episodes+1, count+1) < 1:
        raise ValueError("Invalid export limits")
    output = Path(output).resolve()
    # Explicit --output isolates a smoke export from the configured formal
    # cache. The ROI identity is independent of its destination path.
    config = dict(config, d_evidence_roi_cache=str(output / "cache"))
    cache = roi_cache_root(config)
    recipe = dict(schema=ROI_SPEC["schema"], spec=ROI_SPEC, source_sha256=source_hashes(),
        sensor_configuration_sha256=digest(sensor_config(config)), requested_splits=list(splits),
        max_episodes=max_episodes, frames_per_episode=count, cache=str(cache),
        labels_read=False, detected_boxes_changed=False, trainable_tokens_cached=False)
    recipe_path = output / "export_identity.json"
    if recipe_path.exists() and read(recipe_path) != recipe:
        raise ValueError("Existing ROI export has another source/input scope; preserve it")
    atomic_json(recipe_path, recipe)
    expected_keys = _selected_keys(config, splits, max_episodes, count)
    if (output / "export_receipt.json").exists():
        return validate_export(config, output, verify_cache=True)
    extractor = Extractor(config, device, threads)
    if extractor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(extractor.device)
    started = time.perf_counter(); entries = []; created = reused = 0; extraction_s = 0.
    try:
        for split in splits:
            reader = SensorCache(config, split)
            try:
                ids = reader.ids()[:max_episodes] if max_episodes else reader.ids()
                for episode in ids:
                    for index in reader.indices(episode, count):
                        features, row, sensor = reader.frame(episode, index)
                        payload, receipt_path = roi_paths(config, episode, index)
                        if receipt_path.exists():
                            _, receipt = load_roi(config, features, row, sensor)
                            reused += 1
                        else:
                            raw = Path(sensor["rgb_path"]).read_bytes()
                            if __import__("hashlib").sha256(raw).hexdigest() != row["hashes"]["rgb"]:
                                raise ValueError("Actual RGB changed before ROI extraction")
                            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                            if list(rgb.shape[1::-1]) != np.asarray(features["camera_image_size_wh"]).tolist():
                                raise ValueError("RGB dimensions differ from the frozen boxes")
                            identity = roi_identity(config, features, row, sensor)
                            vectors, timings = extractor.extract(rgb, features["bbox_xyxy"])
                            if roi_identity(config, features, row, sensor) != identity:
                                raise ValueError("Source/box/detector changed during ROI extraction")
                            extraction_s += timings["extraction_s"]
                            checksum = _persist(payload, vectors, features["bbox_xyxy"], features["detection_scores"])
                            atomic_json(receipt_path, dict(identity=identity,
                                payload_sha256=checksum, backend=extractor.backend,
                                backend_source_sha256=extractor.backend_sources, timings=timings,
                                eval_mode=True, inference_mode=True, labels_read=False,
                                rpn_or_detection_rerun=False, no_placeholder_features=True))
                            _, receipt = load_roi(config, features, row, sensor)
                            created += 1
                        entries.append(dict(split=split, episode_id=episode, frame_idx=index,
                                            detected_members=len(features["bbox_xyxy"]), **receipt))
                        wall = time.perf_counter()-started
                        progress = dict(stage="exporting", frames=len(entries), frames_total=len(expected_keys), created=created, reused=reused,
                            split=split, episode=episode, frame_idx=index, elapsed_s=time.perf_counter()-started,
                            eta_s=wall*(len(expected_keys)-len(entries))/len(entries),
                            mean_wall_seconds_per_frame=wall/len(entries), device=str(extractor.device))
                        atomic_json(output / "progress.json", progress)
                        if len(entries) % 50 == 0:
                            print(json.dumps(progress), flush=True)
                    reader.release()
            finally:
                reader.close()
        inventory = output / "inventory.json"
        if [(r["split"], r["episode_id"], r["frame_idx"]) for r in entries] != expected_keys:
            raise ValueError("ROI export did not cover the complete selected inventory")
        atomic_json(inventory, dict(frames=entries, source_sha256=source_hashes()))
        elapsed = time.perf_counter()-started
        result = dict(recipe, status="complete", frames=len(entries), created=created, reused=reused,
            elapsed_s=elapsed, mean_wall_seconds_per_frame=elapsed/max(1, len(entries)),
            extraction_seconds=extraction_s, mean_new_extraction_seconds=extraction_s/max(1, created),
            device=str(extractor.device), backend=extractor.backend,
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(extractor.device)) if extractor.device.type == "cuda" else 0,
            inventory=dict(path=str(inventory), sha256=sha256(inventory)),
            preprocessing_and_backbone_included=True, detector_training=False, final_test_opened=False)
        atomic_json(output / "export_receipt.json", result)
        atomic_json(output / "progress.json", dict(stage="complete", **result))
        return result
    except BaseException as exc:
        atomic_json(output / "progress.json", dict(stage="failed", error=repr(exc), frames=len(entries), elapsed_s=time.perf_counter()-started))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    p.add_argument("--device", default="cpu"); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-episodes", type=int, default=0); p.add_argument("--frames-per-episode", type=int, default=0)
    a = p.parse_args(argv)
    result = export(read(a.config), a.output, splits=tuple(a.splits), device=a.device, threads=a.threads,
                    max_episodes=a.max_episodes, count=a.frames_per_episode)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
