"""Persistent RGB-only DINOv3/FPN detector in its compatible environment.

The protocol accepts project-owned temporary uint8 RGB arrays. It never opens
target annotations or changes source images, detector weights or training data.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CACHE = PROJECT / 'cache' / 'dinov3_worker'


def emit(value):
    sys.stdout.write(json.dumps(value, allow_nan=False) + '\n')
    sys.stdout.flush()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def detector_metadata(detector, weights):
    """Use the strictly loaded model's actual native transform and receipt."""
    from uav_detection import dinov3_detector
    weights = Path(weights).expanduser().resolve()
    transform = detector.model.transform
    minimum = list(transform.min_size)
    if len(minimum) != 1:
        raise ValueError('Deterministic single native minimum image size required')
    result = dict(detector.metadata,
        detector_backend='dinov3_fpn', detector='dinov3_fpn_uav_finetuned',
        weights=str(weights),
        preprocessing=dict(resize='aspect_ratio_short_edge', min_size=int(minimum[0]),
            max_size=int(transform.max_size), image_mean=list(transform.image_mean),
            image_std=list(transform.image_std), size_divisible=int(transform.size_divisible),
            input='RGB uint8 HWC', output_boxes='original_image_xyxy', square_resize=False),
        detector_source_sha256=_sha(dinov3_detector.__file__),
        loader_sha256=_sha(dinov3_detector.__file__), worker_sha256=_sha(__file__),
        frozen=True, eval_mode=not detector.model.training, inference_mode=True,
        online_inputs=['RGB pixels'], oracle_inputs_used=False,
        training_annotation_files_opened=False)
    for name, field in (('config.json', 'training_config'), ('run_manifest.json', 'run_manifest')):
        path = weights.parent / name
        if path.is_file():
            result[field] = str(path)
            result[field + '_sha256'] = _sha(path)
    split = weights.parent / 'split_manifest.json'
    if not split.is_file() or _sha(split) != result['split_manifest_sha256']:
        raise ValueError('DINOv3 training split receipt is missing or changed')
    training_split = json.loads(split.read_text())
    result.update(detector_split_manifest=str(split), detector_split_manifest_sha256=_sha(split),
        detector_training_episodes=training_split.get('splits', {}).get('train', {}).get('episode_ids', []))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Default: validation operating point embedded in these exact weights')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(WORKSPACE))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from uav_detection.dinov3_detector import DINOv3Detector
            torch.set_num_threads(max(1, args.threads))
            detector = DINOv3Detector(args.weights, device=args.device, confidence=args.confidence)
            detector.model.requires_grad_(False).eval()
            metadata = detector_metadata(detector, args.weights)
        emit(dict(ok=True, metadata=metadata))
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit(dict(ok=False, error=str(error)))
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'close':
                return 0
            if request.get('command') != 'detect':
                raise ValueError('Expected detect or close')
            path = Path(request['rgb_npy']).resolve()
            if not path.is_relative_to(CACHE.resolve()):
                raise ValueError('Worker accepts only project cache RGB arrays')
            rgb = np.load(path, allow_pickle=False)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError('Expected RGB uint8 HWC')
            with contextlib.redirect_stdout(sys.stderr):
                rows = detector.detect(rgb)
            emit(dict(ok=True, boxes=[list(row.bbox_xyxy) for row in rows],
                      scores=[float(row.score) for row in rows]))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(dict(ok=False, error=str(error)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
