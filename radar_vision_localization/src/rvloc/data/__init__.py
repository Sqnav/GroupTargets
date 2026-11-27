"""Strict, relocatable manifests and sensor-only observation loading.

``load_manifest`` and ``metadata_report`` read only the manifest. Iterating
observations verifies sensor receipts and centre evidence. Only the explicit
``load_labels`` function opens supervision. RGB mode never opens IQ or radar
cache files, including for range bounds or integrity checks. Explicit prepared
replay uses a manifest-bound full-IQ authentication receipt plus unchanged
source-file identity; it never substitutes an unverified IQ declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterator

import numpy as np

from rvloc.contracts import (Calibration, Camera, ContractError, Labels, Navigation,
                             Observation, PendingDataError, RFMetadata,
                             finite_array, finite_float, strict_fields)

SCHEMA = "rvloc.manifest.v1"
DIRECT_FUSION_SCHEMA = "public_visual_rays_ra_re_direct_v1"
DIRECT_DEPTH_SCHEMA = "public_visual_depth_direct_v1"
PREPARED_REPLAY_SCHEMA = "rvloc.prepared_replay.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes) -> dict:
    try:
        return json.loads(raw, object_pairs_hook=_no_duplicates,
                          parse_constant=lambda x: (_ for _ in ()).throw(ContractError(f"Nonfinite JSON: {x}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("Invalid UTF-8 JSON") from error


def _digest(value, name):
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lowercase SHA256")
    return value


def _relative(value, name):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ContractError(f"Invalid relative asset path: {name}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ContractError(f"Asset paths must stay inside data_root: {name}")
    return value


def _ref(value, name):
    strict_fields(value, {"path", "sha256"}, name=name)
    _relative(value["path"], name)
    _digest(value["sha256"], name)
    return value


def _identifier(value, name):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ContractError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class FrameRecord:
    episode_id: str
    frame_idx: int
    physical_time_s: float
    normal_path: str
    normal_sha256: str
    labels_path: str
    labels_sha256: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    data_root: Path
    code_root: Path
    status: str
    coordinate_frame: str
    rf: RFMetadata
    visual_geometry: dict
    center_contract: dict
    splits: dict[str, tuple[str, ...]]
    episodes: dict[str, tuple[FrameRecord, ...]]
    prepared_replay: dict | None = None

    def metadata_report(self) -> dict:
        """Report readiness without opening data, labels, evidence, or IQ."""
        pending = self.status != "ready" or self.center_contract["status"] != "verified" or not self.episodes
        return dict(schema=SCHEMA, status="pending_data_fix" if pending else "ready",
                    manifest_sha256=self.sha256, coordinate_frame=self.coordinate_frame,
                    center_contract_status=self.center_contract["status"],
                    episodes=len(self.episodes), frames=sum(map(len, self.episodes.values())),
                    splits={key: {"episodes": len(ids), "frames": sum(len(self.episodes[x]) for x in ids)}
                            for key, ids in self.splits.items()}, data_files_opened=False)

    def _resolve(self, relative: str) -> Path:
        _relative(relative, "asset")
        target = (self.data_root / relative).resolve()
        if not target.is_relative_to(self.data_root) or target.is_relative_to(self.code_root):
            raise ContractError("Asset resolves outside data_root or inside code_root")
        return target

    def _read(self, relative: str, expected: str) -> bytes:
        target = self._resolve(relative)
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ContractError(f"SHA256 mismatch: {relative}")
        return raw

    def frames(self, split: str) -> Iterator[FrameRecord]:
        if split not in self.splits:
            raise ContractError(f"Unknown split: {split}")
        for episode in self.splits[split]:
            yield from self.episodes[episode]

    def observations(self, split: str, *, modality: str = "rgb",
                     prepared_only: bool = False) -> Iterator[Observation]:
        """Load current sensors, optionally replaying authenticated RA/RE.

        The opt-in affects only ``rgb_radar``: RGB remains radar-free and FFT
        ``radar`` still reads measured IQ. Prepared replay checks the original
        source's size/mtime/ctime/device/inode on every frame, and fails closed
        after source changes or relocation until the exporter authenticates
        and registers the source again. No trainable CNN outputs are cached.
        """
        if modality not in ("rgb", "radar", "rgb_radar"):
            raise ContractError("modality must be rgb, radar, or rgb_radar")
        if type(prepared_only) is not bool:
            raise ContractError("prepared_only must be boolean")
        _verify_center(self)
        replay = _verify_prepared_replay(self) if prepared_only and modality == "rgb_radar" else None
        for frame in self.frames(split):
            yield load_observation(self, frame, modality=modality, _center_verified=True,
                                   prepared_only=prepared_only, _prepared_replay=replay)


def load_manifest(path: str | Path, *, data_root: str | Path | None = None,
                  code_root: str | Path | None = None, allow_pending: bool = False,
                  expected_sha256: str | None = None) -> Manifest:
    """Parse public metadata only; no dataset enumeration or asset reads.

    ``data_root`` may be supplied at deployment to relocate a manifest. Data,
    caches and output roots must be outside the installed source tree.
    Pending manifests remain inspectable with ``allow_pending=True`` but can
    never produce observations.
    """
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and _digest(expected_sha256, "manifest SHA") != digest:
        raise ContractError("Manifest SHA256 mismatch")
    top = _json_bytes(raw)
    strict_fields(top, {"schema", "status", "coordinate_frame", "rf", "visual_geometry", "center_contract", "splits", "episodes"}, {"data_root", "prepared_replay"}, name="manifest")
    if top["schema"] != SCHEMA or top["coordinate_frame"] != "base_local":
        raise ContractError("Expected rvloc.manifest.v1 in explicit base_local frame")
    if top["status"] not in ("ready", "pending_data_fix"):
        raise ContractError("Unknown data readiness status")
    code_root = Path(code_root).expanduser().resolve() if code_root is not None else Path(__file__).resolve().parents[3]
    selected_root = data_root if data_root is not None else top.get("data_root")
    if selected_root is None:
        raise ContractError("An explicit external data_root is required")
    selected_root = Path(selected_root).expanduser()
    if not selected_root.is_absolute():
        selected_root = path.parent / selected_root
    selected_root = selected_root.resolve()
    if selected_root.is_relative_to(code_root):
        raise ContractError("data_root must be outside code_root")
    rf = RFMetadata.from_mapping(top["rf"])
    geometry = strict_fields(top["visual_geometry"], {"depth_min_m", "depth_max_m", "range_scale_m"}, name="visual_geometry")
    geometry = {key: finite_float(value, key, positive=True) for key, value in geometry.items()}
    if geometry["depth_max_m"] <= geometry["depth_min_m"]:
        raise ContractError("Public visual depth support must be nonempty")
    center = strict_fields(top["center_contract"], {"reference", "co_location", "status", "evidence"}, name="center_contract")
    if center["reference"] != "observer_self_scatter_reference" or center["co_location"] != "camera_optical_center" or center["status"] not in ("verified", "pending"):
        raise ContractError("Explicit observer self-scatter reference/camera optical centre contract required")
    if center["evidence"] is not None:
        _ref(center["evidence"], "center evidence")
    if center["status"] == "verified" and center["evidence"] is None:
        raise ContractError("Verified centre contract requires explicit hashed evidence")
    strict_fields(top["splits"], {"train", "dev"}, name="splits")
    splits, assigned = {}, set()
    for split, ids in top["splits"].items():
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ContractError("Split membership must contain unique episode IDs")
        for item in ids:
            _identifier(item, "episode ID")
        if assigned.intersection(ids):
            raise ContractError("An episode occurs in multiple splits")
        assigned.update(ids)
        splits[split] = tuple(ids)
    if not isinstance(top["episodes"], list):
        raise ContractError("episodes must be a list")
    episodes = {}
    for episode in top["episodes"]:
        strict_fields(episode, {"episode_id", "frames"}, name="episode")
        episode_id = _identifier(episode["episode_id"], "episode ID")
        if episode_id in episodes or not isinstance(episode["frames"], list) or not episode["frames"]:
            raise ContractError("Ready episodes require unique IDs and nonempty chronological frames")
        frames = []
        for row in episode["frames"]:
            strict_fields(row, {"frame_idx", "physical_time_s", "normal_path", "normal_sha256", "labels_path", "labels_sha256"}, name="frame record")
            if type(row["frame_idx"]) is not int or row["frame_idx"] < 0:
                raise ContractError("frame_idx must be a nonnegative integer")
            timestamp = finite_float(row["physical_time_s"], "physical_time_s")
            for kind in ("normal", "labels"):
                _relative(row[f"{kind}_path"], kind)
                _digest(row[f"{kind}_sha256"], kind)
            if row["normal_path"] == row["labels_path"]:
                raise ContractError("Observation and supervision must be separate files")
            record = FrameRecord(episode_id=episode_id, **dict(row, physical_time_s=timestamp))
            if frames and (record.frame_idx != frames[-1].frame_idx + 1 or not np.isclose(record.physical_time_s - frames[-1].physical_time_s, rf.frame_interval_s, rtol=0, atol=1e-9)):
                raise ContractError("Episode frames must be chronological at the physical 1s interval")
            frames.append(record)
        episodes[episode_id] = tuple(frames)
    if set(episodes) != assigned:
        raise ContractError("Episode inventory differs from split membership")
    pending = top["status"] != "ready" or center["status"] != "verified" or not episodes
    if not episodes and top["status"] != "pending_data_fix":
        raise ContractError("Zero episodes must explicitly be pending_data_fix")
    if pending and not allow_pending:
        raise PendingDataError("Corrected data or camera/self-scatter coincidence evidence is pending")
    replay = top.get("prepared_replay")
    if replay is not None:
        _ref(replay, "prepared replay receipt")
    return Manifest(path, digest, selected_root, code_root, top["status"], top["coordinate_frame"], rf, geometry, center, splits, episodes, replay)


def _verify_center(manifest: Manifest) -> None:
    if manifest.metadata_report()["status"] != "ready":
        raise PendingDataError("Pending data cannot produce observations")
    ref = manifest.center_contract["evidence"]
    evidence = _json_bytes(manifest._read(ref["path"], ref["sha256"]))
    strict_fields(evidence, {"schema", "coordinate_frame", "reference", "co_location", "camera_to_scatter_reference_offset_m", "source", "capture_revision"}, name="center evidence")
    offset = finite_array(evidence["camera_to_scatter_reference_offset_m"], (3,), "camera/scatter reference offset")
    if (evidence["schema"] != "rvloc.center_evidence.v1" or evidence["coordinate_frame"] != "base_local"
            or evidence["reference"] != "observer_self_scatter_reference" or evidence["co_location"] != "camera_optical_center"
            or not np.array_equal(offset, np.zeros(3)) or not isinstance(evidence["source"], str) or not evidence["source"]
            or not isinstance(evidence["capture_revision"], str) or not evidence["capture_revision"]):
        raise ContractError("Evidence must explicitly prove zero camera/observer self-scatter offset")



def _iq_stat(path: Path) -> dict:
    """Source identity bound at full SHA verification; no IQ contents read."""
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino)


def _verify_prepared_replay(manifest: Manifest) -> dict:
    """Authenticate a complete, sensor-only export receipt, never its labels.

    The frozen manifest is the trust root, as with centre evidence. Its hashed
    receipt certifies that the exporter actually verified every registered raw
    IQ SHA before recording source stats. Bare normal-file IQ hashes are never
    sufficient. Current source stats and all prepared payload hashes are still
    checked while replaying, so changing an IQ requires full re-registration.
    """
    ref = manifest.prepared_replay
    if ref is None:
        raise ContractError("Prepared-only replay requires a registered full-IQ authentication receipt")
    _ref(ref, "prepared replay receipt")
    receipt = _json_bytes(manifest._read(ref["path"], ref["sha256"]))
    strict_fields(receipt, {"schema", "raw_iq_verified", "sensor_export_labels_read", "fft_source", "frames"},
                  name="prepared replay receipt")
    if (receipt["schema"] != PREPARED_REPLAY_SCHEMA or receipt["raw_iq_verified"] is not True
            or receipt["sensor_export_labels_read"] is not False):
        raise ContractError("Prepared replay requires full raw-IQ verification and label-free sensor export")
    fft = strict_fields(receipt["fft_source"], {"kind", "kernel", "sources"}, name="prepared replay FFT source")
    if (not isinstance(fft["kind"], str) or not fft["kind"]
            or fft["kernel"] != "rectangular_native_ra_re_v1"
            or not isinstance(fft["sources"], dict) or not fft["sources"]):
        raise ContractError("Prepared replay requires an authenticated native RA/RE kernel identity")
    for key, value in fft["sources"].items():
        if not isinstance(key, str) or not key:
            raise ContractError("Prepared replay source names must be nonempty strings")
        _digest(value, "prepared replay source SHA")
    if not isinstance(receipt["frames"], list):
        raise ContractError("Prepared replay frame inventory must be a list")
    expected = {(episode, frame.frame_idx): frame for episode, frames in manifest.episodes.items() for frame in frames}
    registered = {}
    for row in receipt["frames"]:
        strict_fields(row, {"episode_id", "frame_idx", "normal_sha256", "iq_path", "iq_sha256", "iq_stat",
                            "radar_sha256", "visual_sha256"}, name="prepared replay frame")
        _identifier(row["episode_id"], "prepared replay episode ID")
        if type(row["frame_idx"]) is not int or row["frame_idx"] < 0:
            raise ContractError("Prepared replay frame_idx must be a nonnegative integer")
        key = row["episode_id"], row["frame_idx"]
        if key not in expected or key in registered:
            raise ContractError("Prepared replay frames are duplicated or outside the manifest")
        for field in ("normal_sha256", "iq_sha256", "radar_sha256", "visual_sha256"):
            _digest(row[field], field)
        if row["normal_sha256"] != expected[key].normal_sha256:
            raise ContractError("Prepared replay normal identity differs from the current manifest")
        _relative(row["iq_path"], "prepared replay IQ path")
        stamp = strict_fields(row["iq_stat"], {"size", "mtime_ns", "ctime_ns", "device", "inode"}, name="prepared replay IQ stat")
        if any(type(value) is not int or value < 0 for value in stamp.values()):
            raise ContractError("Prepared replay IQ stat fields must be nonnegative integers")
        registered[key] = row
    if registered.keys() != expected.keys():
        raise ContractError("Prepared replay must certify every manifest frame, including all registered splits")
    return registered


def _verify_prepared_frame(manifest: Manifest, frame: FrameRecord, normal: dict, registered: dict) -> None:
    row = registered.get((frame.episode_id, frame.frame_idx))
    if row is None or row["normal_sha256"] != frame.normal_sha256:
        raise ContractError("Current frame lacks registered prepared replay authentication")
    if "iq" not in normal or "radar" not in normal:
        raise ContractError("Prepared-only replay requires explicit authenticated IQ and prepared radar assets")
    if (normal["iq"]["path"] != row["iq_path"] or normal["iq"]["sha256"] != row["iq_sha256"]
            or normal["radar"]["sha256"] != row["radar_sha256"]
            or normal["visual"]["sha256"] != row["visual_sha256"]):
        raise ContractError("Prepared replay sensor identities differ from the authenticated export")
    try:
        stamp = _iq_stat(manifest._resolve(normal["iq"]["path"]))
    except OSError as error:
        raise ContractError("Prepared replay IQ source is unavailable; authenticate and register it again") from error
    if stamp != row["iq_stat"]:
        raise ContractError("Prepared replay IQ source changed; authenticate and register it again")

def _navigation(value: dict, current: float) -> Navigation:
    strict_fields(value, {"physical_time_s", "position_m", "velocity_mps"}, name="navigation")
    times = np.asarray(value["physical_time_s"])
    if times.ndim != 1 or not len(times):
        raise ContractError("Navigation requires explicit current/past samples")
    times = finite_array(times, (len(times),), "navigation times").astype(float)
    if np.any(np.diff(times) <= 0) or np.any(times > current) or not np.isclose(times[-1], current, rtol=0, atol=1e-9):
        raise ContractError("Navigation may contain only chronological current/past physical times")
    return Navigation(times, finite_array(value["position_m"], (len(times), 3), "navigation positions").astype(float),
                      finite_array(value["velocity_mps"], (len(times), 3), "navigation velocities").astype(float))


def _npz(manifest: Manifest, ref: dict, required: set[str]) -> dict:
    _ref(ref, "prepared feature")
    # Read once and authenticate the exact bytes that are decoded.
    from io import BytesIO
    with np.load(BytesIO(manifest._read(ref["path"], ref["sha256"])), allow_pickle=False) as data:
        if len(data.files) != len(set(data.files)) or set(data.files) != required:
            raise ContractError(f"Unknown/missing prepared-feature fields: {sorted(set(data.files) ^ required)}")
        return {key: data[key].copy() for key in data.files}


def _visual(manifest: Manifest, ref: dict, camera: Camera) -> dict:
    from .prepare import VISUAL_FIELDS, validate_visual_arrays
    prepared = _npz(manifest, ref, VISUAL_FIELDS)
    arrays = validate_visual_arrays(prepared, (camera.width, camera.height))
    boxes, scores = arrays["bbox_xyxy"], arrays["detection_scores"]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    optical = np.column_stack((centers, np.ones(len(centers)))) @ np.linalg.inv(camera.K).T
    directions = optical[:, [2, 0, 1]] @ camera.rotation.T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    width, height = (boxes[:, 2:] - boxes[:, :2]).T
    nx, ny = width / camera.K[0, 0], height / camera.K[1, 1]
    bounds = np.broadcast_to([manifest.visual_geometry["depth_min_m"], manifest.visual_geometry["depth_max_m"]], (len(boxes), 2)).copy()
    return dict(schema="visual_bbox_base_rays_v1", depth_schema=DIRECT_DEPTH_SCHEMA,
                ray_origins=np.broadcast_to(camera.position, (len(boxes), 3)).copy(), ray_directions=directions,
                ray_valid=np.ones(len(boxes), bool), ray_depth_bounds=bounds, bbox_xyxy=boxes,
                detection_scores=scores, detector_roi_features=arrays["detector_roi_features"],
                visual_scale=np.column_stack((nx, ny, np.sqrt(nx * ny), np.log(nx), np.log(ny), np.log(width / height), scores)),
                camera_K=camera.K.copy(), camera_image_size_wh=np.array([camera.width, camera.height]),
                range_scale_m=manifest.visual_geometry["range_scale_m"], metadata={
                    "coordinate_frame": "base_local", "labels_read": False, "future_read": False,
                    "iq_read": False, "radar_features_read": False, "geometry_bounds_source": "public_visual_geometry",
                    "source_rgb_sha256": arrays["source_rgb_sha256"].item(),
                    "detector_sha256": arrays["detector_sha256"].item()})



def validate_direct_fusion_features(features: dict) -> dict:
    """Check global-fusion inputs without deriving optical bounds from IQ.

    ``range_scale_m`` is a common normalization scale, not a radar visibility
    sphere. The measured RA/RE axes and masks are validated independently of
    that scale. Validation detaches only temporary views of tensors, keeping
    the original values and their autograd graph unchanged.
    """
    from rvloc.legacy import activate
    activate()
    from precision_study.visual import ALLOWED_FIELDS, SCHEMA as VISUAL_SCHEMA, validate_visual_features
    from .prepare import RADAR_ARRAY_FIELDS
    required = set(ALLOWED_FIELDS) | {"detector_roi_features"} | set(RADAR_ARRAY_FIELDS)
    strict_fields(features, required, name="direct fusion features")
    if features.get("schema") != DIRECT_FUSION_SCHEMA or features.get("depth_schema") != DIRECT_DEPTH_SCHEMA:
        raise ContractError("Expected the shared public-visual direct-depth feature contract")

    def numeric(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    visual = {key: features[key] if key in ("schema", "depth_schema", "metadata", "range_scale_m")
              else numeric(features[key]) for key in ALLOWED_FIELDS}
    visual["schema"] = VISUAL_SCHEMA
    validate_visual_features(visual)
    n = len(visual["ray_directions"])
    roi = numeric(features["detector_roi_features"])
    if roi.shape != (n, 1024) or np.iscomplexobj(roi) or not np.isfinite(roi).all():
        raise ContractError("Direct fusion requires current detection-aligned finite ROI [N,1024]")
    if not isinstance(features["metadata"], dict):
        raise ContractError("Direct fusion requires public sensor provenance metadata")
    for key in ("labels_read", "future_read"):
        if features["metadata"].get(key, False) is not False:
            raise ContractError("Direct fusion cannot read labels or future frames")
    axes = {}
    for key in ("range_m", "azimuth_rad", "elevation_rad"):
        axis = numeric(features[key])
        if (axis.ndim != 1 or len(axis) < 2 or axis.dtype.kind not in "iuf"
                or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)):
            raise ContractError("Direct fusion needs finite increasing physical axis: " + key)
        axes[key] = axis
    if axes["range_m"][0] < 0:
        raise ContractError("Measured range samples must be nonnegative metres")
    for key in ("azimuth_rad", "elevation_rad"):
        if np.any(np.abs(axes[key]) >= np.pi/2):
            raise ContractError("Measured angular samples must be front-hemisphere radians")
    for power, angle, mask in (("ra_power", "azimuth_rad", "ra_valid"),
                               ("re_power", "elevation_rad", "re_valid")):
        values, valid = numeric(features[power]), numeric(features[mask])
        shape = (len(axes["range_m"]), len(axes[angle]))
        if (values.shape != shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ContractError("Direct fusion requires nonnegative measured linear power: " + power)
        if valid.shape != shape or valid.dtype != np.bool_:
            raise ContractError("Direct fusion requires independent measured boolean masks: " + mask)
    return features


def load_observation(manifest: Manifest, frame: FrameRecord, *, modality: str = "rgb",
                     prepared_only: bool = False, _center_verified: bool = False,
                     _prepared_replay: dict | None = None) -> Observation:
    if modality not in ("rgb", "radar", "rgb_radar"):
        raise ContractError("modality must be rgb, radar, or rgb_radar")
    if type(prepared_only) is not bool:
        raise ContractError("prepared_only must be boolean")
    if frame not in manifest.episodes.get(frame.episode_id, ()):
        raise ContractError("Frame is outside this manifest")
    if not _center_verified:
        _verify_center(manifest)
    normal = _json_bytes(manifest._read(frame.normal_path, frame.normal_sha256))
    strict_fields(normal, {"schema", "episode_id", "frame_idx", "physical_time_s", "camera", "calibration", "observer_position_m", "observer_scatter_reference_position_m", "navigation", "visual"}, {"iq", "radar"}, name="normal observation")
    normal_time = finite_float(normal["physical_time_s"], "physical_time_s")
    if normal["schema"] != "rvloc.observation.v1" or normal["episode_id"] != frame.episode_id or type(normal["frame_idx"]) is not int or normal["frame_idx"] != frame.frame_idx or normal_time != frame.physical_time_s:
        raise ContractError("Observation identity differs from manifest")
    # Validate declared refs lexically even in RGB mode; never resolve/open IQ.
    for name in ("visual", "iq", "radar"):
        if name in normal:
            _ref(normal[name], name)
    camera = Camera.from_mapping(normal["camera"])
    observer = finite_array(normal["observer_position_m"], (3,), "observer position").astype(float)
    scatter = finite_array(normal["observer_scatter_reference_position_m"], (3,), "observer self-scatter reference position").astype(float)
    if not np.array_equal(camera.position, scatter):
        raise ContractError("Current camera optical centre differs from observer self-scatter reference")
    nav = _navigation(normal["navigation"], frame.physical_time_s)
    if not np.allclose(nav.position_m[-1], observer, rtol=0, atol=1e-8):
        raise ContractError("Current observer position differs from current navigation")
    calibration = strict_fields(normal["calibration"], {"station_position_m", "array_basis_fru_to_frame"}, name="calibration")
    station = finite_array(calibration["station_position_m"], (3,), "station position").astype(float)
    basis = finite_array(calibration["array_basis_fru_to_frame"], (3, 3), "array basis").astype(float)
    if not np.array_equal(station, np.zeros(3)) or not np.array_equal(basis, np.eye(3)):
        raise ContractError("base_local inputs require station origin zero and array basis identity")
    calibration = Calibration(camera, manifest.rf, station, basis)
    features = {} if modality == "radar" else _visual(manifest, normal["visual"], camera)
    if modality != "radar":
        # In this optical-centre echo branch the known observer radar atom is
        # referenced to the camera, not the body navigation origin. Keep body
        # navigation in the auditable Observation, outside neural features.
        features["observer_position"] = scatter.copy()
        features["metadata"]["observer_position_reference"] = "camera_optical_center"
    source = {"manifest": manifest.sha256, "normal": frame.normal_sha256}
    if modality != "radar":
        source["visual"] = normal["visual"]["sha256"]
    iq = None
    if modality in ("radar", "rgb_radar"):
        if "iq" not in normal:
            raise ContractError("Radar mode requires an explicit authenticated current IQ asset")
        from io import BytesIO
        from .prepare import RADAR_FIELDS, spectra_from_iq, validate_iq, validate_radar_arrays
        if prepared_only and modality == "rgb_radar":
            registered = _prepared_replay if _prepared_replay is not None else _verify_prepared_replay(manifest)
            _verify_prepared_frame(manifest, frame, normal, registered)
            source["prepared_replay"] = manifest.prepared_replay["sha256"]
        else:
            iq = np.load(BytesIO(manifest._read(normal["iq"]["path"], normal["iq"]["sha256"])), allow_pickle=False)
            iq = validate_iq(iq, manifest.rf)
        source["iq"] = normal["iq"]["sha256"]
    if modality == "rgb_radar":
        if "radar" in normal:
            maps = validate_radar_arrays(_npz(manifest, normal["radar"], RADAR_FIELDS), manifest.rf, source["iq"])
            source["radar"] = normal["radar"]["sha256"]
        else:
            maps = spectra_from_iq(iq, manifest.rf, source_iq_sha256=source["iq"])
        # The direct-depth arms share the same public optical support and
        # normalization as RGB. RF range samples remain measured quantities;
        # they do not redefine the legal output interval or detection mask.
        features = dict(features)
        features.update({key: maps[key] for key in
            ("ra_power", "re_power", "range_m", "azimuth_rad", "elevation_rad", "ra_valid", "re_valid")})
        features["schema"] = DIRECT_FUSION_SCHEMA
        features["metadata"] = dict(features["metadata"], iq_read=not prepared_only, radar_features_read=True,
            geometry_bounds_source="public_visual_geometry", range_scale_source="public_visual_geometry",
            radar_range_limit_m=manifest.rf.range_limit, radar_support_defines_output_bounds=False)
        if prepared_only:
            features["metadata"]["prepared_replay_verified"] = True
        validate_direct_fusion_features(features)
    return Observation(frame.episode_id, frame.frame_idx, frame.physical_time_s, features, camera, manifest.rf,
                       calibration, observer, scatter, nav, iq, source)


def load_labels(manifest: Manifest, observation: Observation | FrameRecord) -> Labels:
    """Explicit offline-only supervision access; never called by observation loading."""
    key = observation.episode_id, observation.frame_idx, observation.physical_time_s
    candidates = [row for row in manifest.episodes.get(key[0], ()) if (row.episode_id, row.frame_idx, row.physical_time_s) == key]
    if len(candidates) != 1:
        raise ContractError("Labels requested for a frame outside the manifest")
    frame = candidates[0]
    data = _json_bytes(manifest._read(frame.labels_path, frame.labels_sha256))
    strict_fields(data, {"schema", "episode_id", "frame_idx", "physical_time_s", "positions_m", "member_ids"}, {"velocities_mps"}, name="offline labels")
    label_time = finite_float(data["physical_time_s"], "label physical_time_s")
    if data["schema"] != "rvloc.labels.v1" or type(data["frame_idx"]) is not int or (data["episode_id"], data["frame_idx"], label_time) != key:
        raise ContractError("Offline label identity differs from observation")
    ids = data["member_ids"]
    if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ContractError("Offline member IDs must be unique strings")
    raw_positions = np.asarray(data["positions_m"])
    if not ids and raw_positions.shape == (0,):
        raw_positions = raw_positions.reshape(0, 3)
    positions = finite_array(raw_positions, (len(ids), 3), "label positions").astype(float)
    raw_velocities = None if "velocities_mps" not in data else np.asarray(data["velocities_mps"])
    if raw_velocities is not None and not ids and raw_velocities.shape == (0,):
        raw_velocities = raw_velocities.reshape(0, 3)
    velocities = None if raw_velocities is None else finite_array(raw_velocities, positions.shape, "label velocities").astype(float)
    return Labels(*key, positions, tuple(ids), velocities)


from .prepare import prepare_radar, prepare_visual  # noqa: E402

__all__ = ["Manifest", "FrameRecord", "load_manifest", "load_observation", "load_labels", "prepare_visual", "prepare_radar", "sha256", "DIRECT_FUSION_SCHEMA", "DIRECT_DEPTH_SCHEMA", "validate_direct_fusion_features", "PREPARED_REPLAY_SCHEMA"]
