"""Causal association of image observations with a fixed, visually estimated roster.

Only ``frame_idx, time, boxes, roi, scores`` may cross the input boundary. Boxes
are XYXY image pixels, time is in physical seconds, and ROI vectors have 1024
components. No camera, navigation, range, validity mask, or annotation is used.

The first three *observed* frames allow initialization candidates. The median
of their box-NMS counts estimates the roster, which is then fixed. These are
delayed observation initializations, not claims about physical target births.
Previously returned results are never amended, and absent members never produce
an output: every output corresponds to exactly one current detection.

Defaults below are placeholders; experiment scales and rejection thresholds
must be replaced by the frozen TRAIN calibration before evaluating a holdout.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import time as _time

import numpy as np
from scipy.optimize import linear_sum_assignment


INPUT_FIELDS = frozenset(("frame_idx", "time", "boxes", "roi", "scores"))
VARIANTS = frozenset(("appearance_only", "visual_forced", "visual_gated", "image_motion_only"))
ROI_DIM = 1024
DEFAULT_PARAMS = {
    "appearance_weight": 0.25,
    "center_scale": 1.0,
    "app_scale": 1.0,
    "size_scale": 1.0,
    "cost_gate": 3.0,
    "center_gate": 3.0,
    "ema_velocity": 0.5,
    "ema_appearance": 0.2,
}
DEFAULT_ROSTER_CONFIG = {"frames": 3, "iou_threshold": 0.3, "statistic": "median"}


def _boxes(value, name="boxes"):
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), in XYXY image pixels")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if np.any(array[:, 2:] <= array[:, :2]):
        raise ValueError(f"{name} must have positive width and height")
    return array.copy()


def _roi(value, count, name="roi"):
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == 0 and count == 0:
        array = array.reshape(0, ROI_DIM)
    if array.shape != (count, ROI_DIM):
        raise ValueError(f"{name} must have shape (N, {ROI_DIM})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _unit_rows(array):
    # Scale first to avoid overflow for finite, large feature values. A zero
    # descriptor remains zero and has cosine distance one to every descriptor.
    if len(array) == 0:
        return array.copy()
    maximum = np.max(np.abs(array), axis=1, keepdims=True)
    scaled = np.divide(array, maximum, out=np.zeros_like(array), where=maximum > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(array), where=norm > 0)


def overlap_matrix(a, b):
    """Pairwise image-box IoU; no distance or depth is involved."""
    a = _boxes(a, "a")
    b = _boxes(b, "b")
    intersection = np.prod(
        np.maximum(0.0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                   - np.maximum(a[:, None, :2], b[None, :, :2])), axis=-1
    )
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=-1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def nms_indices(boxes, scores, threshold=0.3):
    """Stable score-ordered NMS used only to estimate/select the initial roster."""
    boxes = _boxes(boxes)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector with one value per box")
    order = np.lexsort((np.arange(len(scores)), -scores))
    kept = []
    while len(order):
        index = int(order[0])
        kept.append(index)
        rest = order[1:]
        order = rest[overlap_matrix(boxes[index:index + 1], boxes[rest])[0] <= threshold]
    return np.asarray(kept, dtype=int)


def pairwise_cost_components(predicted_boxes, boxes, track_roi=None, roi=None):
    """Return three unscaled, purely visual (track, detection) cost matrices.

    ``predicted_boxes`` translate the last observed boxes using image velocity;
    their sizes are unchanged. Center residual is center distance divided by
    the mean of the two box diagonals. Size residual is
    ``L2(log(current_width_height / previous_width_height))``. Appearance is
    ``1 - cosine``. Passing neither ROI array skips all appearance computation
    and returns zeros for that component (the image-motion ablation).
    """
    predicted_boxes = _boxes(predicted_boxes, "predicted_boxes")
    boxes = _boxes(boxes)
    old_wh = predicted_boxes[:, 2:] - predicted_boxes[:, :2]
    new_wh = boxes[:, 2:] - boxes[:, :2]
    old_center = predicted_boxes[:, :2] + 0.5 * old_wh
    new_center = boxes[:, :2] + 0.5 * new_wh
    diagonals = 0.5 * (np.linalg.norm(old_wh, axis=1)[:, None]
                       + np.linalg.norm(new_wh, axis=1)[None, :])
    center = np.linalg.norm(old_center[:, None] - new_center[None, :], axis=-1)
    center /= np.maximum(diagonals, 1e-12)
    size = np.linalg.norm(np.log(new_wh[None, :, :] / old_wh[:, None, :]), axis=-1)
    if track_roi is None and roi is None:
        appearance = np.zeros_like(center)
    elif track_roi is None or roi is None:
        raise ValueError("track_roi and roi must either both be supplied or both be omitted")
    else:
        old_roi = _unit_rows(_roi(track_roi, len(predicted_boxes), "track_roi"))
        new_roi = _unit_rows(_roi(roi, len(boxes)))
        appearance = np.clip(1.0 - old_roi @ new_roi.T, 0.0, 2.0)
    return {"center_residual": center, "appearance_distance": appearance, "size_residual": size}


@dataclass
class _Track:
    identity: int
    box: np.ndarray
    roi: np.ndarray | None
    velocity_px_s: np.ndarray
    last_time: float
    last_frame_idx: int
    last_detection_index: int
    birth_frame: int
    hits: int = 1
    missed: int = 0
    score_sum: float = 0.0
    score_count: int = 1


class VisualAssociationTracker:
    """Episode-local tracker. Construct/reset once for each episode.

    ``step(frame)`` (also ``update``) returns serializable current-observation
    records. The mixed visual cost is
    ``(1-w)*center/center_scale + w*appearance/app_scale + .1*size/size_scale``.
    Appearance-only uses ``appearance/app_scale`` and is forced. Image-motion
    uses ``w=0`` and does not retain or compare ROI descriptors. Forced matching
    removes rejection gates, while preserving one-to-one rectangular matching.
    A finite dummy column per member permits unmatched members/detections with
    one Hungarian solve. Gated matching applies both raw-center and total-cost
    thresholds before that solve.
    """

    def __init__(self, params=None, variant="visual_gated", roster_config=None, **overrides):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        supplied = dict(params or {})
        supplied.update(overrides)
        extra = set(supplied) - (set(DEFAULT_PARAMS) | {"forced"})
        if extra:
            raise ValueError(f"Unknown tracker parameters: {sorted(extra)}")
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(supplied)
        default_forced = variant in ("appearance_only", "visual_forced")
        self.forced = self.params.pop("forced", default_forced)
        if not isinstance(self.forced, (bool, np.bool_)):
            raise ValueError("forced must be boolean")
        self.forced = bool(self.forced)
        for name, value in list(self.params.items()):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite number")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("appearance_weight", "ema_velocity", "ema_appearance"):
                if not 0 <= value <= 1:
                    raise ValueError(f"{name} must be between zero and one")
            elif name in ("cost_gate", "center_gate"):
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
            self.params[name] = value
        self.variant = variant
        if variant == "image_motion_only":
            self.params["appearance_weight"] = 0.0
        config = dict(DEFAULT_ROSTER_CONFIG)
        config.update(roster_config or {})
        if config != DEFAULT_ROSTER_CONFIG:
            raise ValueError("The fixed protocol requires three frames, IoU 0.3 and median roster count")
        self.roster_config = config
        self.reset()

    def reset(self):
        """Forget the entire previous episode, including identity numbering."""
        self.tracks = {}
        self.next_id = 1
        self.frame_number = 0
        self.previous_time = None
        self.previous_frame_idx = None
        self.grouped_counts = []
        self.roster_frozen = False
        self.roster_ids = set()
        self.estimated_member_count = None

    def _validate_frame(self, frame):
        if not isinstance(frame, Mapping):
            raise ValueError("frame must be a mapping containing only the visual input fields")
        extra = set(frame) - INPUT_FIELDS
        missing = INPUT_FIELDS - set(frame)
        if extra:
            raise ValueError(f"Forbidden/unknown frame fields: {sorted(map(str, extra))}")
        if missing:
            raise ValueError(f"Missing visual frame fields: {sorted(missing)}")
        index = frame["frame_idx"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("frame_idx must be an integer")
        index = int(index)
        timestamp = frame["time"]
        if isinstance(timestamp, (bool, np.bool_)) or not np.isscalar(timestamp):
            raise ValueError("time must be a finite scalar in physical seconds")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("time must be finite")
        if self.previous_frame_idx is not None and index <= self.previous_frame_idx:
            raise ValueError("frame_idx must increase strictly within an episode")
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError("time must increase strictly in physical seconds within an episode")
        boxes = _boxes(frame["boxes"])
        roi = _roi(frame["roi"], len(boxes))
        scores = np.asarray(frame["scores"], dtype=float)
        if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
            raise ValueError("scores must be a finite vector with one value per box")
        return index, timestamp, boxes, roi, scores.copy()

    def _components_and_cost(self, tracks, timestamp, boxes, roi):
        predicted = np.asarray([
            track.box + np.tile(track.velocity_px_s * (timestamp - track.last_time), 2)
            for track in tracks
        ], dtype=float).reshape(-1, 4)
        use_appearance = self.variant == "appearance_only" or self.params["appearance_weight"] > 0
        if use_appearance:
            track_roi = np.stack([track.roi for track in tracks])
            components = pairwise_cost_components(predicted, boxes, track_roi, roi)
        else:
            components = pairwise_cost_components(predicted, boxes)
        if self.variant == "appearance_only":
            cost = components["appearance_distance"] / self.params["app_scale"]
        else:
            weight = self.params["appearance_weight"]
            cost = ((1.0 - weight) * components["center_residual"] / self.params["center_scale"]
                    + weight * components["appearance_distance"] / self.params["app_scale"]
                    + 0.1 * components["size_residual"] / self.params["size_scale"])
        if not np.isfinite(cost).all():
            raise ValueError("Visual cost overflow; check image coordinates and calibrated scales")
        return components, cost

    def _assign(self, tracks, timestamp, boxes, roi):
        if not tracks or not len(boxes):
            return []
        components, cost = self._components_and_cost(tracks, timestamp, boxes, roi)
        if self.forced:
            eligible = np.ones(cost.shape, dtype=bool)
            # This penalty enforces maximal real-pair cardinality even when a
            # complete reassignment raises the sum of existing match costs.
            dummy_cost = (float(np.max(cost)) + 1.0) * (len(tracks) + len(boxes) + 1)
        else:
            eligible = ((cost <= self.params["cost_gate"])
                        & (components["center_residual"] <= self.params["center_gate"]))
            dummy_cost = self.params["cost_gate"] + max(1.0, self.params["cost_gate"]) * 1e-9
        forbidden = (dummy_cost + float(np.max(cost)) + 1.0) * (len(tracks) + len(boxes) + 1)
        if not np.isfinite(forbidden):
            raise ValueError("Assignment penalty overflow; check calibrated scales")
        matrix = np.full((len(tracks), len(boxes) + len(tracks)), forbidden)
        matrix[:, :len(boxes)] = np.where(eligible, cost, forbidden)
        matrix[np.arange(len(tracks)), len(boxes) + np.arange(len(tracks))] = dummy_cost
        rows, columns = linear_sum_assignment(matrix)
        matches = []
        for row, column in zip(rows, columns):
            if column >= len(boxes) or not eligible[row, column]:
                continue
            row_alternatives = cost[row, eligible[row]].tolist()
            row_alternatives.remove(float(cost[row, column]))
            column_alternatives = cost[eligible[:, column], column].tolist()
            column_alternatives.remove(float(cost[row, column]))
            alternatives = row_alternatives + column_alternatives
            if not self.forced:
                alternatives.append(dummy_cost)
            margin = min(alternatives) - float(cost[row, column]) if alternatives else None
            association = {
                "previous_frame_idx": tracks[row].last_frame_idx,
                "previous_detection_index": tracks[row].last_detection_index,
                "cost": float(cost[row, column]),
                "margin": float(margin) if margin is not None else None,
                **{key: float(value[row, column]) for key, value in components.items()},
            }
            matches.append((tracks[row], int(column), association))
        return matches

    def _observe(self, track, detection_index, index, timestamp, boxes, descriptors, scores):
        box = boxes[detection_index]
        old_center = 0.5 * (track.box[:2] + track.box[2:])
        new_center = 0.5 * (box[:2] + box[2:])
        measured_velocity = (new_center - old_center) / (timestamp - track.last_time)
        if track.hits == 1:
            track.velocity_px_s = measured_velocity
        else:
            alpha = self.params["ema_velocity"]
            track.velocity_px_s = ((1.0 - alpha) * track.velocity_px_s + alpha * measured_velocity)
        track.box = box.copy()
        if track.roi is not None:
            alpha = self.params["ema_appearance"]
            combined = ((1.0 - alpha) * track.roi + alpha * descriptors[detection_index])
            track.roi = _unit_rows(combined[None, :])[0]
        track.last_time = timestamp
        track.last_frame_idx = index
        track.last_detection_index = detection_index
        track.hits += 1
        track.missed = 0
        track.score_sum += float(scores[detection_index])
        track.score_count += 1

    def _freeze_roster(self, outputs, boxes):
        requested = int(np.floor(float(np.median(self.grouped_counts)) + 0.5))
        current_boxes = {output["track_id"]: boxes[output["detection_index"]]
                         for output in outputs if output["track_id"] is not None}
        ranking = sorted(self.tracks.values(), key=lambda track: (
            -track.hits, track.missed, -track.score_sum / track.score_count,
            track.birth_frame, track.identity,
        ))
        selected, selected_boxes = [], []
        for track in ranking:
            if len(selected) >= requested:
                break
            box = current_boxes.get(track.identity)
            if box is not None and selected_boxes and np.any(overlap_matrix([box], selected_boxes)[0] > 0.3):
                continue
            selected.append(track)
            if box is not None:
                selected_boxes.append(box)
        self.roster_ids = {track.identity for track in selected}
        dropped = sorted(set(self.tracks) - self.roster_ids)
        self.tracks = {track.identity: track for track in selected}
        self.roster_frozen = True
        self.estimated_member_count = len(selected)
        for output in outputs:
            if output["track_id"] not in self.roster_ids:
                output.update(track_id=None, birth=False, association=None)
        return requested, dropped

    def step(self, frame):
        started = _time.perf_counter()
        index, timestamp, boxes, roi, scores = self._validate_frame(frame)
        self.frame_number += 1
        tracks = [self.tracks[identity] for identity in sorted(self.tracks)]
        outputs = [{"detection_index": detection_index, "track_id": None,
                    "birth": False, "association": None}
                   for detection_index in range(len(boxes))]
        matches = self._assign(tracks, timestamp, boxes, roi)
        for track in tracks:
            track.missed += 1
        use_appearance = self.variant == "appearance_only" or self.params["appearance_weight"] > 0
        descriptors = _unit_rows(roi) if use_appearance else None
        for track, detection_index, association in matches:
            outputs[detection_index].update(track_id=track.identity, association=association)
            self._observe(track, detection_index, index, timestamp, boxes, descriptors, scores)
        if not self.roster_frozen:
            for output in outputs:
                if output["track_id"] is not None:
                    continue
                detection_index = output["detection_index"]
                identity = self.next_id
                self.next_id += 1
                self.tracks[identity] = _Track(
                    identity=identity, box=boxes[detection_index].copy(),
                    roi=descriptors[detection_index].copy() if use_appearance else None,
                    velocity_px_s=np.zeros(2), last_time=timestamp,
                    last_frame_idx=index, last_detection_index=detection_index,
                    birth_frame=self.frame_number, score_sum=float(scores[detection_index]),
                )
                output.update(track_id=identity, birth=True)
            self.grouped_counts.append(len(nms_indices(boxes, scores, 0.3)))
        requested = None
        dropped = []
        if not self.roster_frozen and self.frame_number == 3:
            requested, dropped = self._freeze_roster(outputs, boxes)
        self.previous_time = timestamp
        self.previous_frame_idx = index
        current_ids = [output["track_id"] for output in outputs if output["track_id"] is not None]
        assert len(current_ids) == len(set(current_ids)), "A member cannot consume two detections"
        return {
            "frame_idx": index,
            "time": timestamp,
            "outputs": outputs,
            "wall_s": _time.perf_counter() - started,
            "roster_frozen": self.roster_frozen,
            "estimated_member_count": self.estimated_member_count,
            "requested_member_count": requested,
            "unfilled_initialization_slots": (requested - len(self.tracks)) if requested is not None else 0,
            "initialization_candidates_dropped": dropped,
            "grouped_initialization_counts": self.grouped_counts.copy(),
            "alive_tracks": len(self.tracks),
            "current_observations": len(boxes),
            "assigned_current_observations": len(current_ids),
            "unassigned_detection_indices": [output["detection_index"] for output in outputs
                                             if output["track_id"] is None],
            "current_observation_coverage_complete": True,
        }

    update = step
