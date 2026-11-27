"""Observation/label separation and explicit sensor coordinate bases.

Dataset observations use NED as the reference frame. Coordinate adapters may
express these same observations in a fixed station frame. Camera/body rotation
fields always map their FRD axes into that observation reference frame; these
orthogonal bases can have determinant -1 after changing to array XYZ/up.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class RadarConfig:
    carrier_hz: float
    bandwidth_hz: float
    pulse_interval_s: float
    shape: tuple
    spacing_y_m: float
    spacing_z_m: float
    frame_interval_s: float = 1.
    @property
    def wavelength(self):return 299792458./self.carrier_hz
    @property
    def range_limit(self):return 299792458.*self.shape[1]/(2*self.bandwidth_hz)

@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    position: np.ndarray
    rotation: np.ndarray  # camera FRD -> observation reference frame
    width: int
    height: int

@dataclass(frozen=True)
class ObserverPose:
    position: np.ndarray  # observer body origin, not camera optical centre
    rotation: np.ndarray  # body FRD -> observation reference frame

@dataclass(frozen=True)
class Observations:
    episode_id: str
    frame_idx: int
    timestamp: float
    rgb: np.ndarray
    iq: np.ndarray
    camera: Camera
    radar: RadarConfig
    station_position: np.ndarray
    array_basis: np.ndarray  # array forward/right/up -> observation reference frame
    source_hashes: dict = field(default_factory=dict)
    observer: Optional[ObserverPose] = None
    calibration: Optional['Calibration'] = None
    observer_reference: Optional['ObserverReference'] = None
    timing: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Targets:
    positions: np.ndarray
    velocities: np.ndarray
    ids: tuple
    keypoints: np.ndarray
    boxes: np.ndarray
    visible: np.ndarray

@dataclass
class Detection:
    uv: np.ndarray
    covariance: np.ndarray
    bbox: np.ndarray
    score: float
    appearance: np.ndarray
    oracle_id: Optional[str] = None

@dataclass(frozen=True)
class Calibration:
    """Public sensor installation, independent of object supervision.

    T_A_from_B maps column vectors in B into A. The array FRU basis is
    orthogonal with determinant -1 relative to NED; it is not an SO(3) pose.
    Distortion is zero for the recorded simulator pinhole camera, not fitted.
    """
    camera_K: np.ndarray
    distortion: np.ndarray
    camera_offset_body_m: np.ndarray
    radar: RadarConfig
    observer_echo_model: str = 'exact_constant_velocity_cpi'
    observer_front_policy: str = 'ideal_two_sided'
    source: str = 'recorded simulator calibration'


@dataclass(frozen=True)
class ObserverReference:
    """Independent navigation reference at the radar scatterer, never IQ GT.

    None velocity denotes unknown velocity (a solver nuisance variable).
    None covariance denotes exact recorded simulator position, not exact
    complex scattering or known velocity. Non-null covariance is in [p,v].
    """
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    source: str = 'recorded simulator body pose; velocity unavailable'
    identity: str = 'observer'
    echo_offset_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    from_iq: bool = False
    position_covariance: Optional[np.ndarray] = None  # independent 3x3 position-only navigation factor


@dataclass(frozen=True)
class ObservationWindow:
    """At most five existing causal frames from one episode, current last."""
    frames: tuple
    valid_mask: Optional[tuple] = None

    def __post_init__(self):
        frames=tuple(self.frames)
        if not 1 <= len(frames) <= 5:raise ValueError('ObservationWindow needs 1..5 past/current frames')
        if any(not isinstance(f,Observations) for f in frames):raise TypeError('Only Observations are allowed')
        if len({f.episode_id for f in frames}) != 1:raise ValueError('Window crosses episode boundary')
        if any(not np.isfinite(f.timestamp) for f in frames):raise ValueError('Finite frame timestamps are required')
        if any(b.frame_idx<=a.frame_idx or b.timestamp<=a.timestamp for a,b in zip(frames,frames[1:])):
            raise ValueError('Window must be ordered past to current, without duplicates')
        if any(not np.isclose(b.timestamp-a.timestamp,b.frame_idx-a.frame_idx,atol=1e-8)
               for a,b in zip(frames,frames[1:])):
            raise ValueError('Physical frame interval must be 1 second; render timestamps are not elapsed physical time')
        mask=tuple(True for _ in frames) if self.valid_mask is None else tuple(self.valid_mask)
        if len(mask)!=len(frames) or any(not isinstance(m,(bool,np.bool_)) for m in mask) or not mask[-1]:
            raise ValueError('Boolean mask required and current frame must be valid')
        object.__setattr__(self,'frames',frames);object.__setattr__(self,'valid_mask',mask)

    @property
    def current(self):return self.frames[-1]


# Explicit supervision name; the legacy alias remains for existing training.
TrainingTargets = Targets
