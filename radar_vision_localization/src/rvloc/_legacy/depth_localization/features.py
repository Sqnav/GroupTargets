"""Single-frame sensor features retaining observed bounding-box scale.

The original RA/RE schema stays intact for the shared analytic geometry.  This
module adds a separately versioned visual contract; it never changes a source
feature dictionary or writes the original feature cache.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

from rvwm.group_observation import feature_digest as base_feature_digest
from rvwm.group_observation import validate_features as validate_base_features

DEPTH_SCHEMA = 'bbox_scale_base_rays_ra_re_v2'
VISUAL_SCALE_NAMES = (
    'width_over_fx', 'height_over_fy', 'sqrt_normalized_area',
    'log_width_over_fx', 'log_height_over_fy', 'log_pixel_aspect_ratio', 'score',
)
DETECTION_FIELDS = (
    'ray_origins', 'ray_directions', 'ray_valid', 'ray_depth_bounds',
    'bbox_xyxy', 'detection_scores', 'visual_scale',
)
EXTRA_ARRAY_FIELDS = ('bbox_xyxy', 'detection_scores', 'camera_K',
                      'camera_image_size_wh', 'visual_scale')


def _visual_scale(boxes, scores, intrinsic):
    width, height = (boxes[:, 2:] - boxes[:, :2]).T
    nx, ny = width / intrinsic[0, 0], height / intrinsic[1, 1]
    return np.column_stack((nx, ny, np.sqrt(nx * ny), np.log(nx),
                            np.log(ny), np.log(width / height), scores))


def _camera(intrinsic, image_size):
    k = np.asarray(intrinsic)
    size = np.asarray(image_size)
    if (k.shape != (3, 3) or np.iscomplexobj(k) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0., 0., 1.], rtol=0, atol=1e-9)
            or np.linalg.matrix_rank(k) != 3):
        raise ValueError('camera_K must be a finite pinhole calibration in original RGB pixels')
    if (size.shape != (2,) or np.iscomplexobj(size) or size.dtype == np.bool_
            or not np.isfinite(size).all() or np.any(size < 1)
            or np.any(size != np.floor(size))):
        raise ValueError('camera_image_size_wh must be positive integer [width,height]')
    if not (0 <= k[0, 2] <= size[0] and 0 <= k[1, 2] <= size[1]):
        raise ValueError('Pinhole principal point must lie within the calibrated RGB image')
    return k.astype(float), size.astype(np.int64)


def validate_depth_features(features):
    """Validate nonempty and genuinely empty observed-detection sets equally."""
    validate_base_features(features)
    if features.get('depth_schema') != DEPTH_SCHEMA:
        raise ValueError('Unsupported depth-localization feature schema')
    n = len(features['ray_directions'])
    for key, shape in (('bbox_xyxy', (n, 4)), ('detection_scores', (n,)),
                       ('visual_scale', (n, 7))):
        value = np.asarray(features.get(key))
        if value.shape != shape or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError('Expected finite detection-aligned feature: ' + key)
    intrinsic, size = _camera(features.get('camera_K'), features.get('camera_image_size_wh'))
    boxes = np.asarray(features['bbox_xyxy'])
    scores = np.asarray(features['detection_scores'])
    if (np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(boxes[:, :2] < 0)
            or np.any(boxes[:, 2:] > size[None, :] + 1e-4)):
        raise ValueError('bbox_xyxy must be positive-area original RGB pixel boxes inside the image')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Detection scores must be in [0,1]')
    expected = _visual_scale(boxes, scores, intrinsic)
    if not np.allclose(features['visual_scale'], expected, rtol=1e-6, atol=1e-7):
        raise ValueError('visual_scale differs from observed boxes, scores or original calibration')
    contract = features.get('depth_metadata')
    required = dict(box_coordinate_frame='original_rgb_pixels', camera_K_unit='pixels',
                    image_size_order='width_height', visual_scale_unit='dimensionless',
                    visual_scale_names=list(VISUAL_SCALE_NAMES), window=1,
                    uses_target_labels=False, uses_history=False, uses_future=False)
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit sensor-only, single-frame, original-pixel feature contract required')
    # Reject ambiguous booleans/integers rather than accepting True == 1.
    if type(contract['window']) is not int or any(contract[k] is not False for k in
            ('uses_target_labels', 'uses_history', 'uses_future')):
        raise ValueError('Depth inputs must contain exactly one current observation and no labels')
    json.dumps(contract, sort_keys=True, allow_nan=False)
    return features


validate_features = validate_depth_features


def enrich_features(base_features, detections, camera_K, image_size_wh, provenance=None):
    """Append actual frozen detector boxes/scores, without changing base inputs.

    Accept ``rvwm.types.Detection`` objects or dictionaries with explicit
    ``bbox_xyxy`` and ``score`` keys. COCO ``bbox`` dictionaries are deliberately
    rejected because their XYWH convention is different.
    """
    validate_base_features(base_features)
    intrinsic, size = _camera(camera_K, image_size_wh)
    detections = list(detections)
    if len(detections) != len(base_features['ray_directions']):
        raise ValueError('Each observed ray requires exactly one original detector box')
    boxes, scores = [], []
    for detection in detections:
        if isinstance(detection, dict):
            if 'bbox_xyxy' not in detection or 'score' not in detection:
                raise ValueError('Dictionary detections require explicit bbox_xyxy and score')
            box, score = detection['bbox_xyxy'], detection['score']
            uv = detection.get('center_px')
        else:
            box, score, uv = detection.bbox, detection.score, detection.uv
        box = np.asarray(box, float)
        if box.shape != (4,):
            raise ValueError('Each original pixel box must have four XYXY coordinates')
        if uv is not None and not np.allclose(np.asarray(uv), (box[:2] + box[2:]) / 2,
                                              rtol=0, atol=1e-6):
            raise ValueError('Observed rays must use the actual detection bounding-box midpoint')
        boxes.append(box); scores.append(score)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    # Validate before logarithms, including malformed input scores/boxes.
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Finite positive-area observed boxes and finite scores required')
    result = deepcopy(base_features)
    result.update(depth_schema=DEPTH_SCHEMA, bbox_xyxy=boxes.copy(),
                  detection_scores=scores.copy(), camera_K=intrinsic.copy(),
                  camera_image_size_wh=size.copy(),
                  visual_scale=_visual_scale(boxes, scores, intrinsic))
    result['depth_metadata'] = dict(
        box_coordinate_frame='original_rgb_pixels', camera_K_unit='pixels',
        image_size_order='width_height', visual_scale_unit='dimensionless',
        visual_scale_names=list(VISUAL_SCALE_NAMES), window=1,
        uses_target_labels=False, uses_history=False, uses_future=False,
        provenance=deepcopy(provenance or {}))
    return validate_depth_features(result)


def drop_detection_features(features, retained_bool):
    """Apply one observed-detection mask to all detection-aligned fields."""
    validate_depth_features(features)
    keep = np.asarray(retained_bool)
    if keep.shape != (len(features['ray_directions']),) or keep.dtype != np.bool_:
        raise ValueError('retained_bool must be a boolean mask matching observed detections')
    result = deepcopy(features)
    for key in DETECTION_FIELDS:
        result[key] = np.asarray(features[key])[keep].copy()
    return validate_depth_features(result)


def feature_digest(features):
    """Bind both original authenticated sensor features and the new inputs."""
    validate_depth_features(features)
    digest = hashlib.sha256()
    digest.update(base_feature_digest(features).encode())
    digest.update(DEPTH_SCHEMA.encode())
    for key in EXTRA_ARRAY_FIELDS:
        value = np.ascontiguousarray(features[key])
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode()); digest.update(value.tobytes())
    digest.update(json.dumps(features['depth_metadata'], sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()
