"""Public sensor contracts. Supervision is deliberately a separate type.

Positions use the fixed radar array frame ``base_local`` (forward/right/up).
Camera rotations map camera FRD vectors into that frame; their determinant
may therefore be -1. RF frequencies are explicit centred *baseband* samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ContractError(ValueError):
    """A manifest or sensor asset violates its declared contract."""


class PendingDataError(ContractError):
    """Corrected data and its centre-coincidence evidence are not available."""


def strict_fields(value: Any, required: set[str], optional: set[str] = frozenset(), *, name: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ContractError(f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def finite_array(value: Any, shape: tuple, name: str, *, boolean: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or np.iscomplexobj(array):
        raise ContractError(f"Invalid shape or complex values: {name}; expected {shape}")
    if boolean:
        if array.dtype != np.bool_:
            raise ContractError(f"{name} must be boolean")
    elif array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ContractError(f"{name} must contain finite real numbers")
    return array.copy()


def finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ContractError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ContractError(f"Invalid finite value: {name}")
    return result


@dataclass(frozen=True)
class RFMetadata:
    carrier_hz: float
    bandwidth_hz: float
    pulse_interval_s: float
    shape: tuple[int, int, int, int]
    spacing_y_m: float
    spacing_z_m: float
    frequencies_hz: tuple[float, ...]
    frequency_grid: str = "centered_baseband_uniform"
    frame_interval_s: float = 1.0

    def __post_init__(self):
        for key in ("carrier_hz", "bandwidth_hz", "pulse_interval_s", "spacing_y_m", "spacing_z_m", "frame_interval_s"):
            object.__setattr__(self, key, finite_float(getattr(self, key), key, positive=True))
        if tuple(self.shape) != (40, 128, 16, 16) or any(type(x) is not int for x in self.shape):
            raise ContractError("This study requires native IQ shape [40,128,16,16]")
        object.__setattr__(self, "shape", tuple(self.shape))
        for key, expected in (("carrier_hz", 4.9e9), ("bandwidth_hz", 32e6), ("pulse_interval_s", .0025), ("frame_interval_s", 1.)):
            if not np.isclose(getattr(self, key), expected, rtol=1e-12, atol=0):
                raise ContractError(f"This study requires {key}={expected}")
        frequencies = finite_array(self.frequencies_hz, (128,), "frequencies_hz").astype(float)
        expected = (np.arange(128) - 64) * self.bandwidth_hz / 128
        if self.frequency_grid != "centered_baseband_uniform" or not np.allclose(frequencies, expected, rtol=0, atol=1e-8):
            raise ContractError("Explicit centred uniform baseband frequency metadata required")
        object.__setattr__(self, "frequencies_hz", tuple(frequencies))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RFMetadata":
        fields = {"carrier_hz", "bandwidth_hz", "pulse_interval_s", "shape", "spacing_y_m", "spacing_z_m", "frequencies_hz", "frequency_grid", "frame_interval_s"}
        strict_fields(value, fields, name="rf")
        return cls(**value)

    @property
    def wavelength(self) -> float:
        return 299792458. / self.carrier_hz

    @property
    def range_limit(self) -> float:
        return 299792458. * self.shape[1] / (2 * self.bandwidth_hz)

    @property
    def pulse_times_s(self) -> np.ndarray:
        return np.arange(self.shape[0]) * self.pulse_interval_s

    def as_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in vars(self).items()}


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    width: int
    height: int

    @classmethod
    def from_mapping(cls, value: dict) -> "Camera":
        strict_fields(value, {"K", "position_m", "rotation_frd_to_frame", "image_size_wh"}, name="camera")
        k = finite_array(value["K"], (3, 3), "camera.K").astype(float)
        position = finite_array(value["position_m"], (3,), "camera.position_m").astype(float)
        rotation = finite_array(value["rotation_frd_to_frame"], (3, 3), "camera.rotation").astype(float)
        size = finite_array(value["image_size_wh"], (2,), "camera.image_size_wh")
        if np.any(size < 1) or np.any(size != np.floor(size)):
            raise ContractError("Explicit positive integer image dimensions required")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-10) or abs(np.linalg.det(k)) < 1e-12:
            raise ContractError("Invalid calibrated pinhole intrinsics")
        if not (0 <= k[0, 2] <= size[0] and 0 <= k[1, 2] <= size[1]):
            raise ContractError("Principal point lies outside original RGB image")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
            raise ContractError("Camera basis must be orthogonal")
        return cls(k, position, rotation, int(size[0]), int(size[1]))


@dataclass(frozen=True)
class Calibration:
    camera: Camera
    rf: RFMetadata
    station_position_m: np.ndarray
    array_basis_fru_to_frame: np.ndarray


@dataclass(frozen=True)
class Navigation:
    physical_time_s: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray


@dataclass(frozen=True)
class Observation:
    episode_id: str
    frame_idx: int
    physical_time_s: float
    features: dict[str, Any]
    camera: Camera
    rf: RFMetadata
    calibration: Calibration
    observer_position: np.ndarray
    observer_scatter_reference_position: np.ndarray
    navigation: Navigation
    iq: np.ndarray | None = None
    source_hashes: dict[str, str] | None = None

    @property
    def timestamp_s(self) -> float:
        return self.physical_time_s

    @property
    def frame_index(self) -> int:
        return self.frame_idx


@dataclass(frozen=True)
class Labels:
    """Offline supervision; never attached to an Observation."""
    episode_id: str
    frame_idx: int
    physical_time_s: float
    positions: np.ndarray
    member_ids: tuple[str, ...]
    velocities: np.ndarray | None = None


__all__ = ["ContractError", "PendingDataError", "RFMetadata", "Camera", "Calibration", "Navigation", "Observation", "Labels"]
