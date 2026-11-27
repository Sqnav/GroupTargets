"""Build offline TRAIN calibration records from already persisted predictions.

Normal inference never calls this module. Ground truth is opened only after
all three prediction files have passed sensor-only provenance and complete
TRAIN coverage checks. GT identities pair calibration samples only; neither
identities nor positions are attached to observations or prediction inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import CalibrationError, visual_pair_components
from .contracts import finite_array, finite_float, strict_fields
from .data import _json_bytes, load_labels
from .evaluation import match_positions
from .runtime import file_sha256

_SOURCES = {1: "fft", 2: "rgb", 3: "single_frame"}
_PREDICTION_REQUIRED = {"schema", "method_id", "split", "manifest_sha256", "normal_inference", "labels_opened", "frames", "evaluation_contract", "neural_architecture"}
_PREDICTION_OPTIONAL = {"method", "config_sha256", "source", "checkpoint_sha256", "calibration_sha256", "coordinate_frame", "cost"}
_FRAME_REQUIRED = {"episode_id", "frame_idx", "physical_time_s", "positions", "valid"}
_FRAME_OPTIONAL = {"raw_positions", "member_ids", "depths", "scores", "radar_quality", "velocities", "runtime_s", "source_hashes", "bbox_xyxy", "association", "observer_near", "radar", "architecture", "visual_depth_mean_m", "visual_depth_std_m", "ray_origins", "ray_directions", "ray_depth_bounds", "range_scale_m", "radar_query"}


def _prediction_inputs(prediction_paths):
    if isinstance(prediction_paths, Mapping):
        if set(prediction_paths) != set(_SOURCES.values()):
            raise CalibrationError("Exactly fft/rgb/single_frame prediction paths are required")
        return [(key, path) for key, path in prediction_paths.items()]
    if isinstance(prediction_paths, Sequence) and not isinstance(prediction_paths, (str, bytes)):
        if len(prediction_paths) == 3:
            return [(None, path) for path in prediction_paths]
    raise CalibrationError("Provide the three persisted G1/G2/G3 TRAIN prediction paths")


def _read_prediction(path, declared_source, manifest, expected, maximum=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    value = _json_bytes(raw)
    strict_fields(value, _PREDICTION_REQUIRED, _PREDICTION_OPTIONAL, name="calibration prediction")
    method = value["method_id"]
    if type(method) is not int or method not in _SOURCES:
        raise CalibrationError("Calibration requires unfiltered G1/G2/G3 predictions")
    source = _SOURCES[method]
    if value['neural_architecture'] != (None if method == 1 else 'single_query_depth_v3'):
        raise CalibrationError('Calibration must use the current direct-depth neural predictions')
    if value['evaluation_contract'] != dict(matching='same_per_frame_one_to_one_euclidean', match_max_distance_m=maximum):
        raise CalibrationError('Calibration matching must equal the contract frozen before inference')
    if declared_source is not None and declared_source != source:
        raise CalibrationError("Prediction source key differs from file method_id")
    if (value["schema"] != "rvloc.predictions.v1" or value["split"] != "train"
            or value["manifest_sha256"] != manifest.sha256 or value["normal_inference"] is not True
            or value["labels_opened"] is not False or value.get("coordinate_frame", "base_local") != "base_local"):
        raise CalibrationError("Only persisted sensor-only TRAIN predictions for this exact manifest are valid")
    if not isinstance(value["frames"], list):
        raise CalibrationError("Prediction frames must be a list")
    frames = {}
    for row in value["frames"]:
        strict_fields(row, _FRAME_REQUIRED, _FRAME_OPTIONAL, name="calibration prediction frame")
        expected_architecture = 'fft_power_peaks_v1' if method == 1 else 'single_query_depth_v3'
        if 'architecture' in row and row['architecture'] != expected_architecture:
            raise CalibrationError('Prediction frame architecture differs from its direct-depth source contract')
        if not isinstance(row["episode_id"], str) or type(row["frame_idx"]) is not int:
            raise CalibrationError("Prediction frame identity must be explicit")
        key = row["episode_id"], row["frame_idx"]
        if key in frames or key not in expected:
            raise CalibrationError("Duplicate, DEV, or unknown predicted frame in TRAIN calibration")
        timestamp = finite_float(row["physical_time_s"], "prediction physical_time_s")
        if timestamp != expected[key].physical_time_s:
            raise CalibrationError("Prediction time differs from TRAIN manifest")
        points = np.asarray(row["positions"])
        if points.shape == (0,):
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise CalibrationError("Predicted positions require [N,3]")
        # JSON nulls for invalid output rows are allowed, but can never match.
        try:
            points = points.astype(float)
        except (TypeError, ValueError) as error:
            raise CalibrationError("Predicted positions must be numeric/null") from error
        valid = finite_array(row["valid"], (len(points),), "prediction validity", boolean=True)
        if not np.isfinite(points[valid]).all():
            raise CalibrationError("Valid prediction rows must have finite positions")
        if ('visual_depth_mean_m' in row) != ('visual_depth_std_m' in row):
            raise CalibrationError('Visual prior mean/std must be exported together')
        if 'visual_depth_mean_m' in row:
            mean=np.asarray(row['visual_depth_mean_m'],float);std=np.asarray(row['visual_depth_std_m'],float)
            if mean.shape!=(len(points),) or std.shape!=mean.shape or not np.isfinite(mean[valid]).all() or not np.isfinite(std[valid]).all() or np.any(std[valid]<=0):
                raise CalibrationError('Exported visual prior must have finite aligned mean and positive metre std')
        frames[key] = dict(row, positions=points, valid=valid)
    if set(frames) != set(expected):
        raise CalibrationError("Each source must cover every TRAIN frame exactly once")
    import hashlib
    return source, frames, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "neural_architecture":value['neural_architecture'],
                            "checkpoint_sha256":value.get('checkpoint_sha256')}


def _projected_visual_assignment(observation, labels):
    """Partial one-to-one box/GT projection assignment, independent of depth."""
    boxes = np.asarray(observation.features["bbox_xyxy"], float)
    n, m = len(boxes), len(labels.positions)
    if not n or not m:
        return {}
    camera = observation.camera
    local = (labels.positions - camera.position) @ camera.rotation
    optical = local[:, [1, 2, 0]]
    pixels = optical @ camera.K.T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:]
    visible = ((local[:, 0] > 0) & np.isfinite(uv).all(axis=1)
               & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
               & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
    inside = (visible[None] & np.asarray(observation.features["ray_valid"])[:, None]
              & (uv[None, :, 0] >= boxes[:, None, 0]) & (uv[None, :, 0] <= boxes[:, None, 2])
              & (uv[None, :, 1] >= boxes[:, None, 1]) & (uv[None, :, 1] <= boxes[:, None, 3]))
    center, size = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]
    cost = np.linalg.norm((uv[None] - center[:, None]) / size[:, None], axis=-1)
    unmatched = float(n + 1)
    augmented = np.full((n, m + n), unmatched)
    augmented[:, :m] = np.where(inside, cost, unmatched * (n + 2))
    rows, cols = linear_sum_assignment(augmented)
    return {labels.member_ids[int(col)]: int(row) for row, col in zip(rows, cols)
            if col < m and inside[row, col]}


def collect_calibration_inputs(manifest, prediction_paths, model_config=None):
    """Return three record lists plus a provenance ``receipt``.

    ``prediction_paths`` is a list of three files or a mapping with exactly
    ``fft``, ``rgb``, ``single_frame`` keys. ``model_config`` accepts only
    ``match_max_distance_m`` (default None) for the same offline 3D Hungarian
    matching convention used by evaluation. Models are never executed here.

    Visual pairs require the same GT projection inside detected boxes in
    three consecutive TRAIN frames. The two earlier boxes define a pixel
    constant-velocity prediction; only boxes/ROI enter visual cost helpers.
    At least one usable visual pair and one residual per source are required;
    ``fit_calibration`` then checks the stronger fit/calibrate/audit counts.
    """
    options = dict(model_config or {})
    strict_fields(options, set(), {"match_max_distance_m"}, name="calibration input configuration")
    maximum = options.get("match_max_distance_m")
    if maximum is not None:
        maximum = finite_float(maximum, "match_max_distance_m", positive=True)
    if manifest.metadata_report()["status"] != "ready":
        raise CalibrationError("Corrected, verified TRAIN data is required before calibration")
    records = list(manifest.frames("train"))
    expected = {(row.episode_id, row.frame_idx): row for row in records}
    if not expected or len(expected) != len(records):
        raise CalibrationError("Calibration requires nonempty unique TRAIN frames")
    predictions, source_receipts = {}, {}
    for declared, path in _prediction_inputs(prediction_paths):
        source, frames, receipt = _read_prediction(path, declared, manifest, expected, maximum)
        if source in predictions:
            raise CalibrationError("Duplicate prediction source")
        predictions[source], source_receipts[source] = frames, receipt
    if set(predictions) != set(_SOURCES.values()):
        raise CalibrationError("All three G1/G2/G3 prediction sources are required")
    # Every source is validated before any labels or visual assets are opened.
    residuals, trajectories, visual_pairs = [], [], []
    history, previous_episode = {}, None
    source_counts = {source: 0 for source in predictions}
    skipped_invalid_predictions = 0
    observation_count = 0
    label_receipts = {}
    normal_receipts = {}
    for observation in manifest.observations("train", modality="rgb"):
        observation_count += 1
        key = observation.episode_id, observation.frame_idx
        if observation.episode_id != previous_episode:
            history, previous_episode = {}, observation.episode_id
        labels = load_labels(manifest, observation)
        record = expected[key]
        label_receipts[f"{key[0]}/{key[1]}"] = record.labels_sha256
        normal_receipts[f"{key[0]}/{key[1]}"] = observation.source_hashes
        for identity, point in zip(labels.member_ids, labels.positions):
            trajectories.append(dict(episode_id=key[0], split="train", target_id=identity,
                                     time_s=observation.physical_time_s, position_m=point.tolist()))
        for source, frames in predictions.items():
            frame = frames[key]
            for index, target, _ in match_positions(frame["positions"], labels.positions, frame["valid"], maximum):
                residuals.append(dict(episode_id=key[0], split="train", source=source,
                                      residual_m=(frame["positions"][index] - labels.positions[target]).tolist()))
                source_counts[source] += 1
        if len(predictions["rgb"][key]["positions"]) != len(observation.features["bbox_xyxy"]):
            raise CalibrationError("G2 predictions must preserve every current RGB detection row")
        assignment = _projected_visual_assignment(observation, labels)
        boxes = observation.features["bbox_xyxy"]
        roi = observation.features["detector_roi_features"]
        for identity, index in assignment.items():
            previous = history.get(identity, [])
            if (len(previous) >= 2 and previous[-1][0] == key[1] - 1 and previous[-2][0] == key[1] - 2):
                earlier, last = previous[-2], previous[-1]
                dt = observation.physical_time_s - last[1]
                previous_dt = last[1] - earlier[1]
                predicted = last[2] + (last[2] - earlier[2]) * (dt / previous_dt)
                if np.isfinite(predicted).all() and np.all(predicted[2:] > predicted[:2]):
                    values = visual_pair_components(last[2][None], boxes[index][None], last[3][None], roi[index][None],
                                                    predicted_boxes=predicted[None])
                    visual_pairs.append(dict(episode_id=key[0], split="train",
                        **{name: float(value[0]) for name, value in values.items()}))
                else:
                    skipped_invalid_predictions += 1
            history[identity] = (previous + [(key[1], observation.physical_time_s, boxes[index].copy(), roi[index].copy())])[-2:]
    if observation_count != len(expected):
        raise CalibrationError("RGB observations do not cover exactly the declared TRAIN frames")
    if not trajectories or any(count == 0 for count in source_counts.values()):
        raise CalibrationError("Insufficient TRAIN labels or matched residuals for one or more G1/G2/G3 sources")
    if not visual_pairs:
        raise CalibrationError("Insufficient visual pairs: require a matched GT projection inside RGB boxes in at least three consecutive TRAIN frames")
    return dict(residual_records=residuals, trajectory_records=trajectories, visual_pair_records=visual_pairs,
        receipt=dict(schema="rvloc.calibration_inputs.v2", split="train", manifest_sha256=manifest.sha256,
            neural_architecture='single_query_depth_v3',
            episode_ids=list(manifest.splits["train"]), prediction_files=source_receipts,
            label_sha256=label_receipts, observation_hashes=normal_receipts,
            residual_counts=source_counts, trajectory_count=len(trajectories), visual_pair_count=len(visual_pairs),
            match_max_distance_m=maximum, residual_matching="same per-frame 3D Hungarian as evaluation",
            visual_assignment="TRAIN GT camera projection inside original RGB bbox; one-to-one maximum cardinality, minimum normalized pixel-centre error; independent of predicted depth",
            visual_motion="constant pixel box velocity from two preceding consecutive TRAIN frames; current ROI versus previous ROI",
            skipped_nonpositive_predicted_boxes=skipped_invalid_predictions,
            supervision_access="offline calibration only; never attached to normal features", models_executed=False))


__all__ = ["collect_calibration_inputs"]
