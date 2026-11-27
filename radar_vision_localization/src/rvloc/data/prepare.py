"""Prepare public features from RGB/detector callbacks and measured native IQ.

Prepared files contain a strict allowlist of numeric sensor arrays and source
hashes. No target counts, identities, motion, risk, or future fields are valid.
Preparation is explicit and never runs while checking a pending manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from rvloc.contracts import ContractError, RFMetadata, finite_array, strict_fields

VISUAL_SCHEMA = "rvloc.prepared_visual.v1"
RADAR_SCHEMA = "rvloc.prepared_radar.v1"
VISUAL_FIELDS = {"schema", "bbox_xyxy", "detection_scores", "detector_roi_features", "image_size_wh", "source_rgb_sha256", "detector_sha256"}
RADAR_ARRAY_FIELDS = {"ra_power", "re_power", "ra_valid", "re_valid", "range_m", "azimuth_rad", "elevation_rad"}
RADAR_FIELDS = RADAR_ARRAY_FIELDS | {"schema", "source_iq_sha256", "rf_json"}


def _scalar_string(array, name):
    array = np.asarray(array)
    if array.shape != () or array.dtype.kind not in "US":
        raise ContractError(f"{name} must be a scalar string")
    return str(array.item())


def _sha_string(array, name):
    from . import _digest
    return _digest(_scalar_string(array, name), name)


def _destination(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    code_root = Path(__file__).resolve().parents[3]
    if path.is_relative_to(code_root):
        raise ContractError("Prepared data must be written outside code_root")
    if path.suffix != ".npz":
        raise ContractError("Prepared artifacts require an explicit .npz filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_visual_arrays(arrays: dict, image_size_wh) -> dict:
    if set(arrays) != VISUAL_FIELDS or _scalar_string(arrays["schema"], "visual schema") != VISUAL_SCHEMA:
        raise ContractError("Unknown or incomplete prepared visual contract")
    boxes = np.asarray(arrays["bbox_xyxy"])
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ContractError("Observed boxes must have shape [N,4]")
    n = len(boxes)
    boxes = finite_array(boxes, (n, 4), "bbox_xyxy").astype(float)
    scores = finite_array(arrays["detection_scores"], (n,), "detection_scores").astype(float)
    roi = finite_array(arrays["detector_roi_features"], (n, 1024), "detector_roi_features").astype(np.float32)
    size = finite_array(arrays["image_size_wh"], (2,), "image_size_wh")
    if not np.array_equal(size, image_size_wh) or np.any(size < 1) or np.any(size != np.floor(size)):
        raise ContractError("Prepared RGB and current camera image dimensions differ")
    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > size) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ContractError("Observed XYXY boxes must have positive area inside original RGB")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ContractError("Observed detector scores must lie in [0,1]")
    _sha_string(arrays["source_rgb_sha256"], "source RGB SHA")
    _sha_string(arrays["detector_sha256"], "detector SHA")
    return dict(arrays, bbox_xyxy=boxes, detection_scores=scores, detector_roi_features=roi)


def prepare_visual(rgb_path: str | Path, output_path: str | Path, *,
                   detector_callback: Callable[[np.ndarray], dict], detector_sha256: str) -> dict:
    """Run an RGB-only callback and save [N,1024] ROI tokens plus boxes/scores.

    The callback receives the RGB image as an HxWx3 uint8 array and must return
    exactly ``bbox_xyxy``, ``detection_scores``, ``detector_roi_features``. It
    receives no manifest, labels, IQ, navigation, or future frames.
    ``detector_sha256`` identifies the frozen detector checkpoint.
    """
    from io import BytesIO
    from PIL import Image
    from . import _digest
    _digest(detector_sha256, "detector SHA")
    destination = _destination(output_path)
    raw = Path(rgb_path).expanduser().resolve().read_bytes()
    with Image.open(BytesIO(raw)) as source:
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
    detected = detector_callback(rgb)
    strict_fields(detected, {"bbox_xyxy", "detection_scores", "detector_roi_features"}, name="RGB detector output")
    arrays = dict(detected, schema=np.array(VISUAL_SCHEMA), image_size_wh=np.array([rgb.shape[1], rgb.shape[0]]),
                  source_rgb_sha256=np.array(hashlib.sha256(raw).hexdigest()), detector_sha256=np.array(detector_sha256))
    arrays = validate_visual_arrays(arrays, [rgb.shape[1], rgb.shape[0]])
    np.savez_compressed(destination, **arrays)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "schema": VISUAL_SCHEMA, "detections": len(arrays["bbox_xyxy"])}


def validate_iq(iq, rf: RFMetadata) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.shape != rf.shape or iq.dtype.kind != "c" or not np.isfinite(iq).all():
        raise ContractError("Expected finite native complex IQ [40,128,16,16]")
    return iq.copy()


def validate_radar_arrays(arrays: dict, rf: RFMetadata, source_iq_sha256: str) -> dict:
    if set(arrays) != RADAR_FIELDS or _scalar_string(arrays["schema"], "radar schema") != RADAR_SCHEMA:
        raise ContractError("Unknown or incomplete prepared radar contract")
    if _sha_string(arrays["source_iq_sha256"], "source IQ SHA") != source_iq_sha256:
        raise ContractError("Prepared radar does not match current measured IQ")
    from . import _json_bytes
    cached_rf = RFMetadata.from_mapping(_json_bytes(_scalar_string(arrays["rf_json"], "RF JSON").encode()))
    if cached_rf.as_dict() != rf.as_dict():
        raise ContractError("Prepared radar RF metadata differs from public current RF metadata")
    result = dict(arrays)
    axes = {}
    for name in ("range_m", "azimuth_rad", "elevation_rad"):
        axis = np.asarray(arrays[name])
        if axis.ndim != 1 or len(axis) < 2:
            raise ContractError(f"Invalid physical axis: {name}")
        axis = finite_array(axis, axis.shape, name).astype(float)
        if np.any(np.diff(axis) <= 0):
            raise ContractError(f"Physical axis must be strictly increasing: {name}")
        axes[name] = axis
    expected_ranges = np.arange(rf.shape[1]) * 299792458. / (2 * rf.bandwidth_hz)
    if axes["range_m"].shape != (128,) or not np.allclose(axes["range_m"], expected_ranges, rtol=0, atol=1e-8):
        raise ContractError("Range grid must retain every native 32MHz range bin")
    if any(np.any(np.abs(axes[name]) >= np.pi / 2) for name in ("azimuth_rad", "elevation_rad")):
        raise ContractError("RA/RE angles must be physical radians strictly inside the front hemisphere")
    for prefix, axis in (("ra", "azimuth_rad"), ("re", "elevation_rad")):
        shape = (128, len(axes[axis]))
        power = finite_array(arrays[f"{prefix}_power"], shape, f"{prefix}_power").astype(np.float32)
        if np.any(power < 0):
            raise ContractError("RA/RE must contain nonnegative unnormalised linear power")
        result[f"{prefix}_power"] = power
        result[f"{prefix}_valid"] = finite_array(arrays[f"{prefix}_valid"], shape, f"{prefix}_valid", boolean=True)
    result.update(axes)
    return result


def spectra_from_iq(iq, rf: RFMetadata, *, source_iq_sha256: str | None = None,
                    angle_bins: int = 65, spatial_oversampling: int = 4) -> dict:
    """Reuse the native 32MHz rectangular-window RA/RE measurement kernel."""
    iq = validate_iq(iq, rf)
    if type(angle_bins) is not int or angle_bins < 3 or type(spatial_oversampling) is not int or spatial_oversampling < 1:
        raise ContractError("Positive native spatial FFT settings required")
    from rvloc.legacy import activate
    activate()
    from rvwm.group_observation import spectra_from_iq as native_spectra
    maps = native_spectra(iq, rf, dict(ra_re_angle_bins=angle_bins, ra_re_spatial_oversampling=spatial_oversampling))
    if source_iq_sha256 is None:
        source_iq_sha256 = hashlib.sha256(np.ascontiguousarray(iq).view(np.uint8)).hexdigest()
    arrays = {key: maps[key] for key in RADAR_ARRAY_FIELDS}
    arrays.update(schema=np.array(RADAR_SCHEMA), source_iq_sha256=np.array(source_iq_sha256),
                  rf_json=np.array(json.dumps(rf.as_dict(), sort_keys=True, allow_nan=False)))
    return validate_radar_arrays(arrays, rf, source_iq_sha256)


def prepare_radar(iq_path: str | Path, output_path: str | Path, *, rf: RFMetadata,
                  angle_bins: int = 65, spatial_oversampling: int = 4) -> dict:
    """Prepare RA/RE from every native current-frame IQ sample; no labels."""
    from io import BytesIO
    destination = _destination(output_path)
    raw = Path(iq_path).expanduser().resolve().read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    iq = np.load(BytesIO(raw), allow_pickle=False)
    arrays = spectra_from_iq(iq, rf, source_iq_sha256=source_sha,
                             angle_bins=angle_bins, spatial_oversampling=spatial_oversampling)
    np.savez_compressed(destination, **arrays)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "schema": RADAR_SCHEMA, "native_shape": list(rf.shape)}


__all__ = ["prepare_visual", "prepare_radar", "validate_visual_arrays", "validate_radar_arrays", "spectra_from_iq", "validate_iq"]
