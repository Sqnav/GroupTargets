"""Single-frame, sensor-only FFT detection in the fixed array base frame.

The power transform follows the audited ``precision_study.volume`` and
``rvwm.group_observation`` implementation: unitary centred-frequency IFFT,
spatial FFT, then mean native-pulse power divided by the native aperture size.
The cube is retained before the RA/RE max projections so azimuth and elevation
are paired by the measured joint response. No RGB, target count, source label,
motion model, IQ fitting, or risk calculation enters detection.

Peak thresholds below are configurable engineering defaults, not calibrated
performance claims. Freeze their TRAIN calibration before holdout evaluation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
from scipy import fft
from scipy.ndimage import maximum_filter

C = 299792458.0
# Code-only provenance of the trusted pre-projection transform adapted here.
# These are identifiers, never paths opened by the detector.
SIGNAL_CHAIN_LINEAGE = {
    "precision_study.volume.build_power_volume": "9611c1cd950a6724e07f268459f9b045623e05b95eea5a9d3397843c072cb421",
    "rvwm.group_observation.spectra_from_iq": "51e0638ea86e46364d649b8f72f05cce4a0d3ff0168d70aafa1d563996d6618d",
}


@dataclass(frozen=True)
class RadarSettings:
    spatial_oversampling: int = 4
    range_chunk: int = 4
    min_range_m: float = 0.0
    max_range_m: float | None = None
    min_snr_db: float = 12.0
    min_power: float = 0.0
    min_peak_fraction: float = 0.1
    noise_quantile: float = 0.5
    nms_range_bins: int = 1
    nms_spatial_bins: int | None = None
    observer_tag_radius_m: float = 5.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None):
        value = dict(value or {})
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown radar settings: {sorted(unknown)}")
        return cls(**value)

    def __post_init__(self):
        for name in ("spatial_oversampling", "range_chunk"):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("nms_range_bins", "nms_spatial_bins"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, minimum=0)
        for name in ("min_range_m", "min_power", "observer_tag_radius_m"):
            _number(getattr(self, name), name, minimum=0.0)
        _number(self.min_snr_db, "min_snr_db")
        if self.max_range_m is not None:
            _number(self.max_range_m, "max_range_m", minimum=self.min_range_m)
            if self.max_range_m <= self.min_range_m:
                raise ValueError("max_range_m must exceed min_range_m")
        for name in ("min_peak_fraction", "noise_quantile"):
            value = _number(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class RadarDetections:
    """One row per retained measured peak; IDs are deliberately absent.

    ``scores`` is peak power / (peak power + selection threshold), while
    ``quality`` is peak-to-estimated-noise power in dB. Neither is a calibrated
    target probability. ``observer_near`` only annotates spatial proximity to
    an independently provided scattering reference; it never removes a row.
    """
    points_base: np.ndarray
    scores: np.ndarray
    quality: np.ndarray
    valid: np.ndarray
    observer_near: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def positions(self):
        return self.points_base

    @property
    def points(self):
        return self.points_base

    def __len__(self):
        return len(self.points_base)


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def _get(value, key):
    if isinstance(value, Mapping):
        if key not in value:
            raise ValueError(f"Missing explicit RF metadata: {key}")
        return value[key]
    try:
        return getattr(value, key)
    except AttributeError as error:
        raise ValueError(f"Missing explicit RF metadata: {key}") from error


class FFTDetector:
    """Configured FFT transform and deterministic 3D local-peak selection.

    ``rf`` is RFMetadata or a mapping with the same public fields. Explicit
    frequencies are required; an inclusive linspace is never substituted.
    ``station``, when supplied, describes the already-local origin/basis and
    must be zero/identity. The output is always array-forward/right/up metres.
    """

    def __init__(self, rf, settings: RadarSettings | Mapping[str, Any] | None = None):
        self.settings = (settings if isinstance(settings, RadarSettings)
                         else RadarSettings.from_mapping(settings))
        self.rf = rf
        shape = tuple(_get(rf, "shape"))
        if len(shape) != 4:
            raise ValueError("RF shape must be [pulse, frequency, array_y, array_z]")
        self.shape = tuple(_integer(n, "RF dimension", minimum=1) for n in shape)
        if self.shape[1] < 2:
            raise ValueError("At least two frequency samples are required")
        self.carrier_hz = _number(_get(rf, "carrier_hz"), "carrier_hz", minimum=0.0)
        self.bandwidth_hz = _number(_get(rf, "bandwidth_hz"), "bandwidth_hz", minimum=0.0)
        self.pri_s = _number(_get(rf, "pulse_interval_s"), "pulse_interval_s", minimum=0.0)
        self.spacing_y_m = _number(_get(rf, "spacing_y_m"), "spacing_y_m", minimum=0.0)
        self.spacing_z_m = _number(_get(rf, "spacing_z_m"), "spacing_z_m", minimum=0.0)
        if min(self.carrier_hz, self.bandwidth_hz, self.pri_s,
               self.spacing_y_m, self.spacing_z_m) <= 0:
            raise ValueError("Physical RF parameters must be positive")
        if _get(rf, "frequency_grid") != "centered_baseband_uniform":
            raise ValueError("Expected explicit centered_baseband_uniform frequency_grid")
        frequency = np.asarray(_get(rf, "frequencies_hz"), dtype=float)
        k = self.shape[1]
        if frequency.shape != (k,) or not np.isfinite(frequency).all():
            raise ValueError("frequencies_hz must have one finite value per native frequency")
        expected = (np.arange(k, dtype=float) - k // 2) * self.bandwidth_hz / k
        if not np.allclose(frequency, expected, rtol=1e-10, atol=1e-7):
            raise ValueError("Explicit frequencies disagree with the centred uniform FFT grid")
        self.frequencies_hz = frequency.copy()
        self.frequency_step_hz = float(frequency[1] - frequency[0])
        self.wavelength_m = C / self.carrier_hz
        self.range_m = np.arange(k, dtype=float) * C / (2 * k * self.frequency_step_hz)
        over = self.settings.spatial_oversampling
        self.fft_shape = (self.shape[2] * over, self.shape[3] * over)
        self.u_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_shape[0])) * self.wavelength_m / self.spacing_y_m
        self.v_axis = np.fft.fftshift(np.fft.fftfreq(self.fft_shape[1])) * self.wavelength_m / self.spacing_z_m

    def power_volume(self, iq):
        """Native mean-pulse joint power [range,u,v], without peak selection.

        This is the audited transform, retaining rectangular windows and all
        pulses. Spatial zero padding only interpolates the finite aperture.
        """
        iq = np.asarray(iq)
        if (iq.shape != self.shape or not np.iscomplexobj(iq)
                or not np.isfinite(iq).all()):
            raise ValueError("Expected finite complex native IQ with the declared RF shape")
        _, k, ny, nz = iq.shape
        profiles = fft.ifft(fft.ifftshift(iq, axes=1), axis=1, norm="ortho", workers=1)
        volume = np.empty((k,) + self.fft_shape, np.float32)
        for start in range(0, k, self.settings.range_chunk):
            beam = fft.fft2(profiles[:, start:start + self.settings.range_chunk],
                            s=self.fft_shape, axes=(-2, -1), workers=1)
            power = np.mean(np.abs(beam) ** 2, axis=0, dtype=np.float64) / (ny * nz)
            if not np.isfinite(power).all() or np.any(power > np.finfo(np.float32).max):
                raise ValueError("Native FFT power overflow; IQ cannot be represented as finite float32 power")
            volume[start:start + len(power)] = fft.fftshift(power, axes=(-2, -1))
        return volume

    @staticmethod
    def _check_station(station):
        if station is None:
            return
        if isinstance(station, Mapping):
            origin = station.get("station_position_m", station.get("position", np.zeros(3)))
            basis = station.get("array_basis_fru_to_frame", station.get("basis", np.eye(3)))
        elif hasattr(station, "station_position_m"):
            origin = station.station_position_m
            basis = station.array_basis_fru_to_frame
        else:
            origin, basis = station, np.eye(3)
        origin, basis = np.asarray(origin, float), np.asarray(basis, float)
        if (origin.shape != (3,) or basis.shape != (3, 3)
                or not np.allclose(origin, 0., rtol=0., atol=1e-9)
                or not np.allclose(basis, np.eye(3), rtol=0., atol=1e-9)):
            raise ValueError("FFT output requires a zero-origin, identity-basis base_local station")

    @staticmethod
    def _observer_point(observer):
        if observer is None:
            return None
        if isinstance(observer, Mapping):
            # Only the independently supplied scattering point is accepted.
            forbidden = {"from_iq", "derived_from_iq"} & set(observer)
            if any(observer[name] for name in forbidden):
                raise ValueError("Observer reference must be independent of this IQ")
            point = observer.get("position", observer.get("position_base_m"))
        else:
            if bool(getattr(observer, "from_iq", False)):
                raise ValueError("Observer reference must be independent of this IQ")
            point = getattr(observer, "position", observer)
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Observer scattering reference must be a finite base-frame 3-vector")
        return point

    def detect(self, iq, station=None, observer=None):
        self._check_station(station)
        observer_point = self._observer_point(observer)
        volume = self.power_volume(iq)
        s = self.settings
        physical = self.u_axis[:, None] ** 2 + self.v_axis[None, :] ** 2 < 1.0 - 1e-12
        range_valid = (self.range_m > max(0.0, s.min_range_m))
        if s.max_range_m is not None:
            range_valid &= self.range_m <= s.max_range_m
        eligible = range_valid[:, None, None] & physical[None, :, :]
        values = volume[eligible]
        maximum = float(values.max()) if values.size else 0.0
        # The scale-relative floor avoids numerical FFT residue being called
        # noise-free infinite-SNR evidence. It does not normalize peak power.
        floor = max(np.finfo(np.float32).tiny, maximum * np.finfo(np.float32).eps)
        noise = max(float(np.quantile(values, s.noise_quantile)) if values.size else 0.0, floor)
        threshold = max(s.min_power, noise * 10.0 ** (s.min_snr_db / 10.0),
                        maximum * s.min_peak_fraction)
        angular_radius = (s.spatial_oversampling if s.nms_spatial_bins is None
                          else s.nms_spatial_bins)
        sizes = (2 * s.nms_range_bins + 1, 2 * angular_radius + 1, 2 * angular_radius + 1)
        masked_power = np.where(eligible, volume, -np.inf)
        local_max = maximum_filter(masked_power, size=sizes, mode=("constant", "wrap", "wrap"), cval=-np.inf)
        candidates = np.argwhere(eligible & (volume > 0.0) & (volume >= threshold) & (volume == local_max))
        # Stable descending power order makes ties reproducible. Resolve exact
        # plateaus with NMS, without inferring how many sources occupy a cell.
        if len(candidates):
            powers = volume[tuple(candidates.T)]
            order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], -powers))
            selected = []
            for index in order:
                cell = candidates[index]
                keep = True
                for previous in selected:
                    delta = np.abs(cell - previous)
                    delta[1:] = np.minimum(delta[1:], np.asarray(self.fft_shape) - delta[1:])
                    if np.all(delta <= [s.nms_range_bins, angular_radius, angular_radius]):
                        keep = False
                        break
                if keep:
                    selected.append(cell)
            cells = np.asarray(selected, dtype=int).reshape(-1, 3)
        else:
            cells = np.empty((0, 3), dtype=int)
        power = volume[tuple(cells.T)].astype(float) if len(cells) else np.empty(0)
        radius = self.range_m[cells[:, 0]]
        u, v = self.u_axis[cells[:, 1]], self.v_axis[cells[:, 2]]
        forward = np.sqrt(np.maximum(0.0, 1.0 - u * u - v * v))
        points = radius[:, None] * np.column_stack((forward, u, v))
        valid = np.isfinite(points).all(axis=1) & (points[:, 0] > 0.0)
        near = (np.linalg.norm(points - observer_point, axis=1) <= s.observer_tag_radius_m
                if observer_point is not None else np.zeros(len(points), dtype=bool))
        scores = power / (power + max(threshold, floor))
        quality = 10.0 * np.log10(np.maximum(power, floor) / noise)
        metadata = dict(
            version="pure_fft_ruv_peaks_v1", coordinate_frame="base_local",
            signal_chain_lineage=dict(SIGNAL_CHAIN_LINEAGE),
            shape=list(self.shape), angular_fft_shape=list(self.fft_shape),
            frequencies_hz=self.frequencies_hz.tolist(),
            pulse_times_s=(np.arange(self.shape[0]) * self.pri_s).tolist(),
            range_bin_m=float(self.range_m[1] - self.range_m[0]),
            range_phase="-4*pi*f*R/c", spatial_phase="+2*pi*(y*u+z*v)/wavelength",
            range_transform="ifft(ifftshift(centered_frequency), norm=ortho)",
            spatial_transform="fft2(y,z), fftshift(y,z); divide power by Ny*Nz",
            pulse_reduction="mean native-pulse beam power", window="rectangular",
            uses_every_native_sample=True, uses_history=False, uses_rgb=False,
            uses_target_count=False, uses_source_labels=False, uses_motion=False,
            iq_fit=False, risk_used=False, observer_subtracted=False,
            observer_cells_cleared=False, observer_policy="annotate_only",
            observer_reference_base_m=None if observer_point is None else observer_point.tolist(),
            selection_settings=asdict(s), calibration_status="configured; no calibration run by detector",
            noise_power=noise, selection_threshold_power=threshold,
            peak_power=power.tolist(), peak_cells=cells.tolist(),
            range_m=radius.tolist(), u=u.tolist(), v=v.tolist(),
            spatial_aliases_possible=bool(max(self.spacing_y_m, self.spacing_z_m) > self.wavelength_m / 2 * (1 + 1e-10)),
            resolution_policy="spatial zero padding interpolates; no resolution claim",
        )
        return RadarDetections(points, scores, quality, valid, near, metadata)
