"""Base-frame rays and numeric RA/RE spectra without a ray-depth grid.

Native IQ is consumed only by this deterministic adapter. RA/RE are correlated
power projections of the same mixture, never independent per-member ranges.
"""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from .coordinates import FixedStationFrame
from .geometry import rays
from .types import Observations

FEATURE_SCHEMA = 'base_ray_ra_re_v1'
SPECTRUM_VERSION = 'rectangular_native_ra_re_v1'
_ARRAY_FIELDS = ('ray_origins', 'ray_directions', 'ray_valid', 'observer_position',
                 'ra_power', 're_power', 'range_m', 'azimuth_rad', 'elevation_rad',
                 'ra_valid', 're_valid', 'ray_depth_bounds')


def _array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
    h.update(value.view(np.uint8))
    return h.hexdigest()


def _json_identity(value):
    """Canonical JSON values for observation provenance, without tensor data."""
    if is_dataclass(value):
        return {field.name: _json_identity(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return _json_identity(value.tolist())
    if isinstance(value, np.generic):
        return _json_identity(value.item())
    if isinstance(value, dict):
        return {str(key): _json_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_identity(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported observation provenance value: ' + type(value).__name__)


def validate_observation_inputs(observation):
    """Check the sensor contract before computation *or* any cache lookup.

    These checks do not read target labels or perform an FFT. A cache hit must
    never turn an IQ-derived reference, invalid pose, or unhandled distortion
    into an apparently valid observation.
    """
    if not isinstance(observation, Observations):
        raise TypeError('Only Observations are accepted')
    if (not isinstance(observation.episode_id, str) or not observation.episode_id
            or isinstance(observation.frame_idx, (bool, np.bool_))
            or int(observation.frame_idx) != observation.frame_idx or observation.frame_idx < 0
            or not np.isfinite(observation.timestamp)):
        raise ValueError('Finite timestamp, episode identity and nonnegative integer frame required')
    reference = observation.observer_reference
    if reference is None:
        raise ValueError('Independent observer reference required')
    if not isinstance(reference.from_iq, (bool, np.bool_)) or reference.from_iq:
        raise ValueError('Observer reference must be independent of the IQ being encoded')

    def vector(value, name):
        value = np.asarray(value)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Finite three-vector required: ' + name)

    def covariance(value, size, name):
        if value is None:
            return
        value = np.asarray(value, float)
        if (value.shape != (size, size) or not np.isfinite(value).all()
                or not np.allclose(value, value.T, rtol=1e-7, atol=1e-9)
                or np.linalg.eigvalsh(value).min() < -1e-10):
            raise ValueError('Finite symmetric positive-semidefinite covariance required: ' + name)

    vector(reference.position, 'observer position')
    vector(reference.echo_offset_body_m, 'observer echo offset')
    if reference.velocity is not None:
        vector(reference.velocity, 'observer velocity')
    covariance(reference.covariance, 6, 'observer state')
    covariance(reference.position_covariance, 3, 'observer position')
    radar = observation.radar
    iq = np.asarray(observation.iq)
    if (iq.ndim != 4 or iq.shape != tuple(radar.shape) or min(iq.shape) < 1
            or iq.shape[1] < 2 or not np.iscomplexobj(iq) or not np.isfinite(iq).all()):
        raise ValueError('Expected finite native complex IQ [pulse, frequency, y, z]')
    physical = (radar.carrier_hz, radar.bandwidth_hz, radar.pulse_interval_s,
                radar.spacing_y_m, radar.spacing_z_m, radar.frame_interval_s)
    if not np.isfinite(physical).all() or min(physical) <= 0:
        raise ValueError('Positive physical radar parameters required')
    if not np.isclose(radar.frame_interval_s, 1., rtol=0., atol=1e-9):
        raise ValueError('Physical frame interval must be 1 second; it is separate from pulse PRI')
    camera = observation.camera
    intrinsic = np.asarray(camera.K, float)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or np.linalg.matrix_rank(intrinsic) != 3):
        raise ValueError('Finite nonsingular camera intrinsics with positive focal lengths required')
    rgb = np.asarray(observation.rgb)
    if (camera.width < 1 or camera.height < 1
            or rgb.shape != (camera.height, camera.width, 3) or not np.isfinite(rgb).all()):
        raise ValueError('Finite RGB image must match calibrated camera dimensions')
    calibration = observation.calibration
    if calibration is not None:
        distortion = np.asarray(calibration.distortion, float)
        if not np.isfinite(distortion).all() or np.any(distortion != 0):
            raise ValueError('Nonzero distortion requires an explicit calibrated undistortion adapter')
        vector(calibration.camera_offset_body_m, 'camera installation offset')
        if (np.asarray(calibration.camera_K).shape != (3, 3)
                or not np.allclose(calibration.camera_K, intrinsic, rtol=1e-7, atol=1e-9)
                or _json_identity(calibration.radar) != _json_identity(radar)):
            raise ValueError('Calibration and observation sensor parameters disagree')
    # Reuse the coordinate adapter's orthogonality and finite-pose checks.
    FixedStationFrame.from_observation(observation).observations(observation)
    return observation


def observation_cache_identity(observation):
    """Bind the actual arrays and complete sensor/reference contract to a cache.

    Recorded file hashes remain useful provenance, but they do not replace
    content hashes: callers may perturb arrays or metadata after loading.
    """
    validate_observation_inputs(observation)
    identity = {field.name: _json_identity(getattr(observation, field.name))
                for field in fields(observation) if field.name not in ('rgb', 'iq')}
    for name in ('rgb', 'iq'):
        value = np.asarray(getattr(observation, name))
        identity[name] = dict(content_sha256=_array_hash(value), shape=list(value.shape), dtype=str(value.dtype))
    # Fail before writing a cache if provenance contains a nonfinite value.
    json.dumps(identity, sort_keys=True, allow_nan=False)
    return identity


def _positive_integer(config, key, default, minimum=1):
    value = config.get(key, default)
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError('Invalid positive integer: ' + key)
    return int(value)


def spectra_from_iq(iq, radar, config=None):
    """All-native rectangular-window range and true-angle Bartlett powers.

    Range = unitary IFFT(ifftshift(centred frequency)). Every pulse contributes
    to mean beam power: equivalently sum unitary Doppler powers / Q (Parseval).
    Spatial padding interpolates the finite-aperture response, not resolution.
    Actual element spacing maps physical direction cosines onto the FFT grid.
    """
    config = {} if config is None else config
    if config.get('ra_re_window', 'rectangular') != 'rectangular':
        raise ValueError('Only ra_re_window=rectangular is supported; refusing a silent window change')
    iq = np.asarray(iq)
    if (iq.ndim != 4 or iq.shape != tuple(radar.shape) or min(iq.shape) < 1
            or not np.iscomplexobj(iq) or not np.isfinite(iq).all()):
        raise ValueError('Expected finite native complex IQ [pulse, frequency, y, z]')
    physical = (radar.carrier_hz, radar.bandwidth_hz, radar.pulse_interval_s,
                radar.spacing_y_m, radar.spacing_z_m)
    if not np.isfinite(physical).all() or min(physical) <= 0:
        raise ValueError('Positive physical radar parameters required')
    q, k, ny, nz = iq.shape
    if k < 2:
        raise ValueError('At least two native frequency samples required')
    common_bins = _positive_integer(config, 'ra_re_angle_bins', 65, 3)
    az_bins = _positive_integer(config, 'ra_re_azimuth_bins', common_bins, 3)
    el_bins = _positive_integer(config, 'ra_re_elevation_bins', common_bins, 3)
    oversampling = _positive_integer(config, 'ra_re_spatial_oversampling', 4)
    chunk = _positive_integer(config, 'ra_re_range_chunk', 4)
    limit = float(config.get('ra_re_angle_limit_deg', 85.))
    if not np.isfinite(limit) or not 0 < limit < 90:
        raise ValueError('RA/RE angle limit must lie strictly between 0 and 90 degrees')
    azimuth = np.linspace(-np.deg2rad(limit), np.deg2rad(limit), az_bins)
    elevation = np.linspace(-np.deg2rad(limit), np.deg2rad(limit), el_bins)
    ranges = np.arange(k, dtype=float) * 299792458. / (2 * radar.bandwidth_hz)
    uy = np.cos(elevation)[None, :] * np.sin(azimuth)[:, None]
    uz = np.broadcast_to(np.sin(elevation)[None, :], uy.shape)
    my, mz = ny * oversampling, nz * oversampling
    fy0, fz0 = np.fft.fftshift(np.fft.fftfreq(my))[0], np.fft.fftshift(np.fft.fftfreq(mz))[0]
    coordinates = np.array([(uy * radar.spacing_y_m / radar.wavelength - fy0) * my,
                            (uz * radar.spacing_z_m / radar.wavelength - fz0) * mz])
    profiles = fft.ifft(fft.ifftshift(iq, axes=1), axis=1, norm='ortho', workers=1)
    ra = np.empty((k, az_bins), np.float32); re = np.empty((k, el_bins), np.float32)
    for start in range(0, k, chunk):
        beam = fft.fft2(profiles[:, start:start + chunk], s=(my, mz), axes=(-2, -1), workers=1)
        power = np.mean(np.abs(beam) ** 2, axis=0, dtype=np.float64) / (ny * nz)
        power = fft.fftshift(power, axes=(-2, -1))
        for offset, plane in enumerate(power):
            angular = map_coordinates(plane, coordinates, order=1, mode='grid-wrap', prefilter=False)
            ra[start + offset] = np.maximum(angular.max(axis=1), 0.)
            re[start + offset] = np.maximum(angular.max(axis=0), 0.)
    metadata = dict(
        version=SPECTRUM_VERSION, native_shape=list(iq.shape), iq_dtype=str(iq.dtype),
        iq_content_sha256=_array_hash(iq), native_sample_count=int(iq.size),
        uses_every_native_sample=True, source_mixture='group_plus_observer_unmodified',
        observer_subtracted=False, observer_cells_cleared=False,
        coordinate_frame='base_local', angle_unit='rad', range_unit='m',
        range_transform='ifft(ifftshift(frequency), norm=ortho); native length',
        spatial_transform='fft2(y,z) / sqrt(Ny*Nz); zero padding only',
        pulse_reduction='mean native-pulse beam power; sum unitary Doppler powers / Q by Parseval',
        projection='RA=max physical elevation; RE=max physical azimuth',
        angular_interpolation='linear periodic spatial FFT power at true-angle direction cosines',
        windows=dict(frequency='rectangular', pulse='rectangular', y='rectangular', z='rectangular'),
        normalization='linear power; no independent RA/RE peak normalization, no noise whitening',
        spectra_are_independent_observations=False, doppler_information_retained=False,
        complex_phase_retained=False, finite_aperture_resolution_unchanged=True,
        angular_fft_shape=[my, mz], angular_bins=dict(azimuth=az_bins, elevation=el_bins), angular_limit_deg=limit,
        carrier_hz=float(radar.carrier_hz), bandwidth_hz=float(radar.bandwidth_hz),
        pulse_interval_s=float(radar.pulse_interval_s),
        frequency_hz=((np.arange(k) - k // 2) * radar.bandwidth_hz / k).tolist(),
        pulse_time_s=(np.arange(q) * radar.pulse_interval_s).tolist(),
        antenna_y_m=(np.arange(ny) * radar.spacing_y_m).tolist(),
        antenna_z_m=(np.arange(nz) * radar.spacing_z_m).tolist(),
        spatial_aliases_possible=bool(radar.spacing_y_m > radar.wavelength / 2 * (1 + 1e-10)
                                      or radar.spacing_z_m > radar.wavelength / 2 * (1 + 1e-10)))
    return dict(ra_power=ra, re_power=re, range_m=ranges,
                azimuth_rad=azimuth, elevation_rad=elevation,
                ra_valid=np.ones(ra.shape, bool), re_valid=np.ones(re.shape, bool), metadata=metadata)


def _ray_intervals(origins, directions, requested, radius):
    """Analytic positive-depth range-sphere/front-halfspace intersection only."""
    n = len(directions); bounds = np.zeros((n, 2), float); valid = requested.copy()
    for i, (origin, direction) in enumerate(zip(origins, directions)):
        if not valid[i]:
            continue
        if not np.isclose(np.linalg.norm(direction), 1., atol=1e-6, rtol=1e-6):
            raise ValueError('Valid ray directions must be unit vectors')
        b = float(origin @ direction)
        discriminant = b * b - float(origin @ origin) + radius * radius
        if discriminant <= 0:
            valid[i] = False; continue
        root = np.sqrt(discriminant)
        lo, hi = max(0., -b - root), -b + root
        if direction[0] > 1e-12:
            lo = max(lo, -origin[0] / direction[0])
        elif direction[0] < -1e-12:
            hi = min(hi, -origin[0] / direction[0])
        elif origin[0] <= 0:
            valid[i] = False; continue
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            valid[i] = False; continue
        bounds[i] = (lo, hi)
    return bounds, valid


def build_spectral_features(*, ray_origins, ray_directions, observer_position,
                            ra_power, re_power, range_m, azimuth_rad, elevation_rad,
                            range_scale_m=None, ray_valid=None, ra_valid=None,
                            re_valid=None, metadata=None):
    """IQ-free adapter. All geometry is already base_local; spectra are numeric.

    Shared camera origin [3] is broadcast to [N,3]. Bounds are analytical, never
    sampled along the ray. An invalid detection remains present with a mask.
    """
    directions = np.asarray(ray_directions, dtype=float)
    if directions.size == 0:
        directions = directions.reshape(0, 3)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError('ray_directions must have shape [N,3]')
    n = len(directions); origins = np.asarray(ray_origins, dtype=float)
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, (n, 3)).copy()
    if origins.shape != (n, 3) or not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError('Finite base-frame ray origins and directions required')
    ranges = np.asarray(range_m, dtype=float)
    if range_scale_m is None:
        if ranges.ndim != 1 or len(ranges) < 2:
            raise ValueError('At least two physical range samples required')
        range_scale_m = float(ranges[-1] + .5 * (ranges[-1] - ranges[-2]))
    scale = float(range_scale_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive public range_scale_m required')
    requested = np.ones(n, bool) if ray_valid is None else np.asarray(ray_valid)
    if requested.shape != (n,) or requested.dtype != np.bool_:
        raise ValueError('ray_valid must be a boolean [N] mask')
    bounds, valid = _ray_intervals(origins, directions, requested, scale)
    if np.iscomplexobj(ra_power) or np.iscomplexobj(re_power):
        raise ValueError('RA/RE inputs must be real nonnegative linear power, not complex amplitudes')
    ra = np.asarray(ra_power, dtype=np.float32); re = np.asarray(re_power, dtype=np.float32)
    provenance = dict(metadata or {})
    if provenance.get('coordinate_frame', 'base_local') != 'base_local':
        raise ValueError('Spectral features require base_local geometry')
    if provenance.get('angle_unit', 'rad') != 'rad':
        raise ValueError('Spectral angle_unit must be rad; convert the axes explicitly')
    if provenance.get('range_unit', 'm') != 'm':
        raise ValueError('Spectral range_unit must be m; convert the axes explicitly')
    provenance.update(coordinate_frame='base_local', angle_unit='rad', range_unit='m',
                      ray_depth_discretized=False, ray_interval='analytic public range sphere intersect array front',
                      model_inputs='base ray origins/directions, independent observer position, numeric RA/RE and physical axes/masks')
    features = dict(schema=FEATURE_SCHEMA, ray_origins=origins.copy(), ray_directions=directions.copy(),
                    ray_valid=valid, observer_position=np.asarray(observer_position, dtype=float).copy(),
                    ra_power=ra.copy(), re_power=re.copy(), range_m=ranges.copy(),
                    azimuth_rad=np.asarray(azimuth_rad, dtype=float).copy(),
                    elevation_rad=np.asarray(elevation_rad, dtype=float).copy(),
                    ra_valid=np.ones(ra.shape, bool) if ra_valid is None else np.asarray(ra_valid).copy(),
                    re_valid=np.ones(re.shape, bool) if re_valid is None else np.asarray(re_valid).copy(),
                    ray_depth_bounds=bounds, range_scale_m=scale, metadata=provenance)
    return validate_features(features)


def validate_features(features):
    """Validate the numeric sensor-only contract without changing the input."""
    required = set(_ARRAY_FIELDS) | {'schema', 'range_scale_m', 'metadata'}
    if not isinstance(features, dict) or not required <= features.keys():
        raise ValueError('Incomplete RA/RE ray feature contract')
    if features['schema'] != FEATURE_SCHEMA:
        raise ValueError('Unsupported RA/RE feature schema')
    scale = float(features['range_scale_m'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite range scale required')
    for key in _ARRAY_FIELDS:
        if not np.isfinite(np.asarray(features[key])).all():
            raise ValueError('Nonfinite spectral feature: ' + key)
    directions = np.asarray(features['ray_directions']); n = len(directions)
    shapes = dict(ray_origins=(n, 3), ray_directions=(n, 3), ray_valid=(n,),
                  observer_position=(3,), ray_depth_bounds=(n, 2))
    for key, shape in shapes.items():
        if np.asarray(features[key]).shape != shape:
            raise ValueError('Feature shape mismatch: ' + key)
    for key in ('range_m', 'azimuth_rad', 'elevation_rad'):
        axis = np.asarray(features[key])
        if axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0):
            raise ValueError('Physical axes must be strictly increasing: ' + key)
    if np.asarray(features['range_m'])[0] < 0:
        raise ValueError('Range axis must be nonnegative metres')
    for key in ('azimuth_rad', 'elevation_rad'):
        if np.any(np.abs(features[key]) >= np.pi / 2):
            raise ValueError('Front-hemisphere angular axes must be radians in (-pi/2,pi/2)')
    shapes = dict(ra_power=(len(features['range_m']), len(features['azimuth_rad'])),
                  re_power=(len(features['range_m']), len(features['elevation_rad'])))
    for key, shape in shapes.items():
        if np.asarray(features[key]).shape != shape or np.any(np.asarray(features[key]) < 0):
            raise ValueError('Expected nonnegative linear power with physical axes: ' + key)
        mask = np.asarray(features[key.replace('power', 'valid')])
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError('Spectral validity masks must be boolean with matching shape')
    valid = np.asarray(features['ray_valid']); bounds = np.asarray(features['ray_depth_bounds'])
    if valid.dtype != np.bool_:
        raise ValueError('ray_valid must be boolean')
    if not np.allclose(np.linalg.norm(directions[valid], axis=-1), 1., atol=1e-6, rtol=1e-6):
        raise ValueError('Valid ray directions must be unit vectors')
    if np.any(bounds[valid, 0] < 0) or np.any(bounds[valid, 1] <= bounds[valid, 0]):
        raise ValueError('Valid rays require a positive analytic depth interval')
    if np.any(bounds[~valid] != 0):
        raise ValueError('Invalid ray intervals must be explicitly zero')
    expected_bounds, expected_valid = _ray_intervals(
        np.asarray(features['ray_origins']), directions, valid, scale)
    if (not np.array_equal(expected_valid, valid)
            or not np.allclose(bounds, expected_bounds, rtol=1e-6, atol=1e-6)):
        raise ValueError('Ray depth bounds must equal the analytic public geometry intersection')
    if not isinstance(features['metadata'], dict) or features['metadata'].get('coordinate_frame') != 'base_local':
        raise ValueError('Explicit base_local feature metadata required')
    if features['metadata'].get('angle_unit', 'rad') != 'rad':
        raise ValueError('RA/RE angular coordinates must use radians')
    if features['metadata'].get('range_unit', 'm') != 'm':
        raise ValueError('RA/RE ranges must use metres')
    from .single_frame import validate_single_frame_features
    validate_single_frame_features(features)
    return features


def describe_observation(observation, detections, config):
    """Adapt RGB-derived rays and the original mixture to the model boundary."""
    validate_observation_inputs(observation)
    base = FixedStationFrame.from_observation(observation).observations(observation)
    spectra = spectra_from_iq(base.iq, base.radar, config)
    n = len(detections)
    uv = np.asarray([d.uv for d in detections], dtype=float).reshape(n, 2)
    valid = np.isfinite(uv).all(axis=1)
    directions = np.zeros((n, 3), float)
    if valid.any():
        directions[valid] = rays(uv[valid], base.camera)
    metadata = dict(schema=FEATURE_SCHEMA, preprocessing=spectra.pop('metadata'),
                    raw_hashes=dict(base.source_hashes), episode_id=base.episode_id,
                    frame_idx=int(base.frame_idx), timestamp_s=float(base.timestamp),
                    frame_interval_s=float(base.radar.frame_interval_s),
                    observer_reference_source=base.observer_reference.source,
                    observer_from_iq=False, observer_excluded_from_detection_count=True,
                    raw_iq_is_model_input=False, appearance_is_model_input=False,
                    uses_target_labels=False, uses_future=False, uncertainty_calibrated=False,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return build_spectral_features(ray_origins=base.camera.position, ray_directions=directions,
                                   observer_position=base.observer_reference.position,
                                   ray_valid=valid, range_scale_m=base.radar.range_limit,
                                   metadata=metadata, **spectra)


def feature_digest(features):
    """Bind geometry, physical axes, masks, raw data and preprocessing version."""
    validate_features(features)
    h = hashlib.sha256(); h.update(features['schema'].encode())
    for key in _ARRAY_FIELDS:
        h.update(key.encode()); h.update(_array_hash(features[key]).encode())
    h.update(repr(float(features['range_scale_m'])).encode())
    h.update(json.dumps(features['metadata'], sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()
