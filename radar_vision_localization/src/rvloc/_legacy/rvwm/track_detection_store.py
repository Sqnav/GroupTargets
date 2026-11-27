"""Read a frozen RGB-only export for explicitly bound capture phases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from .phase_selection import validate_phases,inventory_phases,phase_label


class FrameNotExported(KeyError):
    """This frame has no row in the verified tracking-phase episode shard."""


class TrackDetectionStore:
    """Return original-pixel XYXY boxes and scores, retaining genuine empty rows.

    An instance reads one explicit split (train by default). RGB hashes are the
    SHA256 of the encoded source file. Optional pixel and trajectory hashes
    bind cached results to an in-memory observation.
    """

    def __init__(self, directory, *, dataset_root, split_manifest, weights_sha256,
                 split="train", confidence=0.45, imgsz=None, phases=("approach",),
                 backend="dinov3_fpn", detector_contract=None):
        self.directory = Path(directory).expanduser().resolve()
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        split_bytes = Path(split_manifest).read_bytes()
        self.phases=validate_phases(phases)
        if self.phases!=inventory_phases(json.loads(split_bytes)):
            raise ValueError('Selected detection phases differ from frozen source inventory')
        expected = {
            "schema": "rvwm_track_rgb_detections_v1" if self.phases==("track",) else "rvwm_selected_phase_rgb_detections_v2",
            "phase": phase_label(self.phases),
            "weights_sha256": weights_sha256,
            "confidence": confidence,
            "imgsz": imgsz,
            "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "classes": {"0": "drone"},
        }
        if backend != "dinov3_fpn":
            raise ValueError("Only the frozen DINOv3 detector is supported")
        if manifest.get("detector_backend") != backend:
            raise ValueError("Tracking detection manifest mismatch: detector_backend")
        required = {"detector_backend", "architecture", "weights_sha256", "confidence",
                    "imgsz", "preprocessing", "adapter_sha256", "loader_sha256",
                    "detector_split_manifest_sha256", "run_manifest_sha256", "training_config_sha256"}
        if not isinstance(detector_contract, dict) or not required <= detector_contract.keys():
            raise ValueError("Complete DINO detector identity required for exported replay")
        if (detector_contract["detector_backend"] != backend
                or detector_contract["weights_sha256"] != weights_sha256
                or detector_contract["confidence"] != confidence
                or detector_contract["imgsz"] is not None or imgsz is not None):
            raise ValueError("DINO export arguments disagree with the frozen detector identity")
        expected.update(detector_contract)
        if self.phases!=("track",):expected["selected_phases"]=list(self.phases)
        if manifest.get("complete") is not True:
            raise ValueError("Tracking detection export is incomplete")
        if Path(manifest["dataset_root"]).resolve() != Path(dataset_root).resolve():
            raise ValueError("Tracking detection dataset_root mismatch")
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Tracking detection manifest mismatch: {key}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        partition = json.loads(split_bytes)
        self.split = split
        self.backend = backend
        self.detector_contract = dict(detector_contract or {})
        self.manifest = manifest
        self.confidence = confidence
        self.episodes = frozenset(partition["splits"][split]["episode_ids"])
        self._inventory = {e["episode"]: e for e in partition["episodes"] if e["episode"] in self.episodes}
        if self.episodes != self._inventory.keys():
            raise ValueError("Frozen split is missing an episode inventory")
        self._episode = None
        self._rows = {}

    def _load(self, episode):
        if episode not in self.episodes:
            raise ValueError(f"{episode} does not belong to the {self.split} split")
        receipt = json.loads((self.directory / "receipts" / f"{episode}.json").read_bytes())
        raw = (self.directory / self.split / f"{episode}.jsonl").read_bytes()
        source = self._inventory[episode]
        if (receipt["episode_id"] != episode or receipt["split"] != self.split
                or receipt["prediction_sha256"] != hashlib.sha256(raw).hexdigest()
                or receipt["source_frames_sha256"] != source["frames_sha256"]):
            raise ValueError(f"Tracking detection receipt mismatch for {episode}")
        rows = {}
        for line in raw.splitlines():
            row = json.loads(line)
            idx = row["frame_idx"]
            if (type(idx) is not int or idx < 0 or idx in rows
                    or row["episode_id"] != episode or row["split"] != self.split
                    or row["phase"] not in self.phases or row["source_frames_sha256"] != source["frames_sha256"]):
                raise ValueError(f"Invalid tracking detection row for {episode}/{idx}")
            rows[idx] = row
        if len(rows) != source["frame_count"] or len(rows) != receipt["frames"]:
            raise ValueError(f"Tracking detection frame count mismatch for {episode}")
        if "selected_frame_indices" in source and sorted(rows)!=source["selected_frame_indices"]:
            raise ValueError(f"Selected detection frame indices mismatch for {episode}")
        self._episode, self._rows = episode, rows

    def get(self, episode_id, frame_idx, rgb_sha256, *, rgb_content_sha256=None,
            source_frames_sha256=None, image_size_wh=None):
        if episode_id != self._episode:
            self._load(episode_id)
        if frame_idx not in self._rows:
            raise FrameNotExported((episode_id, frame_idx))
        row = self._rows[frame_idx]
        if not rgb_sha256 or row["rgb_sha256"] != rgb_sha256:
            raise ValueError(f"Source RGB SHA256 mismatch for {episode_id}/{frame_idx}")
        if rgb_content_sha256 is not None and row["rgb_content_sha256"] != rgb_content_sha256:
            raise ValueError(f"Decoded RGB SHA256 mismatch for {episode_id}/{frame_idx}")
        if source_frames_sha256 is not None and row["source_frames_sha256"] != source_frames_sha256:
            raise ValueError(f"Source trajectory SHA256 mismatch for {episode_id}/{frame_idx}")
        width, height = row["image_size_wh"]
        if image_size_wh is not None and tuple(image_size_wh) != (width, height):
            raise ValueError(f"RGB dimensions mismatch for {episode_id}/{frame_idx}")
        detections = row["detections"]
        if any(d["class_id"] != 0 or d["detection_id"] != i for i, d in enumerate(detections)):
            raise ValueError(f"Invalid detection class or index for {episode_id}/{frame_idx}")
        boxes = np.asarray([d["bbox_xyxy"] for d in detections], dtype=float).reshape(len(detections), 4)
        scores = np.asarray([d["score"] for d in detections], dtype=float)
        if not (np.isfinite(boxes).all() and np.isfinite(scores).all()
                and (boxes[:, :2] >= 0).all() and (boxes[:, 2:] <= (width, height)).all()
                and (boxes[:, 2:] > boxes[:, :2]).all()
                and (scores >= self.confidence).all() and (scores <= 1).all()):
            raise ValueError(f"Invalid detection geometry or score for {episode_id}/{frame_idx}")
        return boxes, scores
