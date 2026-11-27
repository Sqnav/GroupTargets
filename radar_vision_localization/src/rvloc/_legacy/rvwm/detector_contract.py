"""Detector identity and preprocessing, independent of sensor/target labels.

The DINOv3 checkpoint supplies its validation-selected operating threshold and
native aspect-ratio transform. Only the trained DINOv3 frontend is supported.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DINO_BACKEND = 'dinov3_fpn'
DINO_ARCHITECTURE = 'dinov3_vits16_fpn_fasterrcnn_v1'
DINO_WEIGHTS = WORKSPACE / 'uav_detection/outputs/training/dinov3_fpn_uav/best.pt'


def _identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError('Detector contract needs a file: ' + str(path))
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=128)
def _digest(path, size, modified):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    return _digest(*_identity(path))


def verify_trained_detector_contract(config, frontend_metadata):
    """Keep a trained fusion head bound to its DINO detector and transform."""
    if config.get('detector_backend', DINO_BACKEND) != DINO_BACKEND:
        raise ValueError('Only the frozen DINOv3 detector is supported')
    current = detector_contract(config)
    if (frontend_metadata.get('detector_backend') != DINO_BACKEND
            or frontend_metadata.get('detector_contract') != current):
        raise ValueError('Weights belong to another RGB frontend: DINOv3 detector contract differs')


@lru_cache(maxsize=8)
def _checkpoint_metadata(path, size, modified):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != DINO_ARCHITECTURE or int(checkpoint.get('steps', 0)) < 1:
        raise ValueError('Expected a trained DINOv3 FPN Faster R-CNN checkpoint')
    # Do not retain the 132 MB tensor state in the provenance cache.
    return {name: checkpoint.get(name) for name in
            ('architecture', 'steps', 'epoch', 'config', 'validation', 'split_manifest_sha256')}


def detector_contract(config, weights=None):
    """Return JSON-safe model/transform identity without reading annotations."""
    if config.get('detector_backend', DINO_BACKEND) != DINO_BACKEND:
        raise ValueError('Only the frozen DINOv3 detector is supported')
    path = Path(weights or config.get('detector_weights') or DINO_WEIGHTS).expanduser().resolve()
    metadata = _checkpoint_metadata(*_identity(path))
    training = Path(config.get('dinov3_training_config', path.parent / 'config.json')).expanduser().resolve()
    run_path = Path(config.get('dinov3_run_manifest', path.parent / 'run_manifest.json')).expanduser().resolve()
    split_path = Path(config.get('dinov3_split_manifest', path.parent / 'split_manifest.json')).expanduser().resolve()
    train_config = json.loads(training.read_text())
    run = json.loads(run_path.read_text())
    split_sha = file_sha256(split_path)
    if (train_config.get('architecture') != DINO_ARCHITECTURE
            or run.get('architecture') != DINO_ARCHITECTURE
            or metadata['split_manifest_sha256'] != split_sha
            or run.get('split_manifest_sha256') != split_sha):
        raise ValueError('DINOv3 architecture/training split provenance mismatch')
    checkpoint_config = metadata['config']
    if not isinstance(checkpoint_config, dict) or checkpoint_config != train_config:
        raise ValueError('DINOv3 checkpoint and frozen training configuration differ')
    configured_split = config.get('visual_split_manifest')
    if configured_split and file_sha256(configured_split) != split_sha:
        raise ValueError('DINOv3 visual split differs from its frozen training split')
    if config.get('dataset_root') and Path(config['dataset_root']).resolve() != Path(train_config['dataset_root']).resolve():
        raise ValueError('DINOv3 detector dataset root differs from the configured source')
    for name, expected in run.get('source_sha256', {}).items():
        source = (WORKSPACE / name).resolve()
        if not source.is_relative_to(WORKSPACE) or file_sha256(source) != expected:
            raise ValueError('DINOv3 trained detector source changed: ' + name)
    weight_sha = file_sha256(path)
    if config.get('detector_weights_sha256') not in (None, weight_sha):
        raise ValueError('Configured DINOv3 weight SHA256 mismatch')
    validation = metadata.get('validation') or {}
    selected = validation.get('operating_point', {}).get('threshold')
    confidence = config.get('detector_confidence')
    confidence = selected if confidence is None else confidence
    if confidence is None or not math.isfinite(float(confidence)) or not 0. <= float(confidence) <= 1.:
        raise ValueError('DINOv3 needs a finite validation-selected or explicit score threshold')
    minimum, maximum = int(checkpoint_config.get('min_size', 720)), int(checkpoint_config.get('max_size', 1280))
    for name, expected in (('detector_min_size', minimum), ('detector_max_size', maximum)):
        if name in config and config[name] != expected:
            raise ValueError('DINOv3 uses its native checkpoint transform: ' + name)
    preprocessing = dict(resize='aspect_ratio_preserving_short_side_with_max_long_side',
        min_size=minimum, max_size=maximum, size_divisible=32,
        input='uint8 RGB HWC', tensor_scale='float RGB / 255',
        image_mean=[.485, .456, .406], image_std=[.229, .224, .225],
        box_coordinates='original RGB pixels', nms_iou=.5, max_detections=100)
    return dict(detector_backend=DINO_BACKEND, architecture=DINO_ARCHITECTURE,
        weights=str(path), weights_sha256=weight_sha, confidence=float(confidence), imgsz=None,
        threshold_source='best_checkpoint_validation' if confidence == selected else 'explicit_configuration',
        validation_threshold=selected, checkpoint_epoch=metadata['epoch'], checkpoint_steps=metadata['steps'],
        preprocessing=preprocessing, training_config=str(training), training_config_sha256=file_sha256(training),
        run_manifest=str(run_path), run_manifest_sha256=file_sha256(run_path),
        detector_split_manifest=str(split_path), detector_split_manifest_sha256=split_sha,
        adapter_sha256=file_sha256(WORKSPACE / 'uav_detection/dinov3_detector.py'),
        loader_sha256=file_sha256(Path(__file__).with_name('dinov3_worker.py')),
        inference='DINOv3Detector.detect; RGB-only; FP32 CPU / BF16 CUDA; native Faster R-CNN transform',
        classes={'0': 'drone'}, frozen=True, score_calibrated=False,
        training_annotation_files_opened=False)
