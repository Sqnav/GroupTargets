"""Calibrated joint range / array-direction power, before RA/RE projections.

This module neither loads labels nor changes IQ. Spatial FFT bins are direction
cosines (u=y/R, v=z/R), not azimuth/elevation angles. All windows, pulse reduction
and linear-power normalization match ``rvwm.group_observation.spectra_from_iq``.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
import torch
from torch.nn import functional as F

C = 299792458.
VERSION = 'joint_ruv_native_power_v1'


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def build_power_volume(iq, radar, config=None):
    """Return native linear ``volume_power[R,U,V]`` and calibrated axes.

    The complete periodic FFT rectangle is retained, including beam-forming
    samples outside the physical unit disk. Those numerical samples are needed
    to interpolate near its boundary; physical validity is checked at each query.
    ``physical_direction_mask`` is descriptive, never a confidence estimate.
    """
    config = {} if config is None else config
    if config.get('ra_re_window', 'rectangular') != 'rectangular':
        raise ValueError('Joint comparison requires unchanged rectangular windows')
    iq = np.asarray(iq)
    if (iq.ndim != 4 or iq.shape != tuple(radar.shape) or min(iq.shape) < 1
            or not np.iscomplexobj(iq) or not np.isfinite(iq).all()):
        raise ValueError('Expected finite complex IQ [pulse, frequency, y, z]')
    physical = (radar.carrier_hz, radar.bandwidth_hz, radar.pulse_interval_s,
                radar.spacing_y_m, radar.spacing_z_m)
    if not np.isfinite(physical).all() or min(physical) <= 0 or iq.shape[1] < 2:
        raise ValueError('Positive radar calibration and at least two frequencies required')
    over = _positive_integer(config.get('ra_re_spatial_oversampling', 4), 'spatial oversampling')
    chunk = _positive_integer(config.get('ra_re_range_chunk', 4), 'range chunk')
    q, k, ny, nz = iq.shape
    my, mz = ny * over, nz * over
    wavelength = C / radar.carrier_hz
    ranges = np.arange(k, dtype=np.float64) * C / (2 * radar.bandwidth_hz)
    u = np.fft.fftshift(np.fft.fftfreq(my)) * wavelength / radar.spacing_y_m
    v = np.fft.fftshift(np.fft.fftfreq(mz)) * wavelength / radar.spacing_z_m
    profiles = fft.ifft(fft.ifftshift(iq, axes=1), axis=1, norm='ortho', workers=1)
    volume = np.empty((k, my, mz), np.float32)
    for start in range(0, k, chunk):
        beam = fft.fft2(profiles[:, start:start + chunk], s=(my, mz),
                        axes=(-2, -1), workers=1)
        power = np.mean(np.abs(beam) ** 2, axis=0, dtype=np.float64) / (ny * nz)
        volume[start:start + len(power)] = fft.fftshift(power, axes=(-2, -1))
    digest = hashlib.sha256()
    digest.update(str(iq.dtype).encode()); digest.update(str(iq.shape).encode())
    digest.update(np.ascontiguousarray(iq).view(np.uint8))
    return dict(volume_power=volume, range_m=ranges, u_axis=u, v_axis=v,
        angular_valid=np.ones((my, mz), bool),
        physical_direction_mask=u[:, None] ** 2 + v[None, :] ** 2 <= 1. + 1e-12,
        metadata=dict(version=VERSION, native_shape=list(iq.shape), fft_shape=[my, mz],
            iq_sha256=digest.hexdigest(), dtype=str(iq.dtype), coordinate_frame='base_local',
            axes=['range_m', 'u=array_y/range', 'v=array_z/range'],
            direction_from_angles='u=cos(el)*sin(az); v=sin(el)',
            range_transform='ifft(ifftshift(centered frequency), norm=ortho); native length',
            spatial_transform='fft2(y,z), fftshift(y,z); power divided by Ny*Nz',
            pulse_reduction='mean native-pulse beam power',
            frequency_hz=((np.arange(k) - k // 2) * radar.bandwidth_hz / k).tolist(),
            windows=dict(frequency='rectangular', pulse='rectangular', y='rectangular', z='rectangular'),
            power_unit='same linear power as existing RA/RE before max projection',
            power_scale_policy='caller reuses RA/RE mean power; no independent peak normalization',
            noise_whitening=False, noise_rms_read=False, complex_phase_retained=False,
            doppler_information_retained=False, uses_every_native_sample=True,
            uses_labels=False, uses_history=False, uses_future=False,
            observer_subtracted=False, source_mixture='group_plus_observer_unmodified',
            angular_boundary='periodic FFT seam; physical disk checked on original query',
            spatial_aliases_possible=bool(radar.spacing_y_m > wavelength/2*(1+1e-10)
                                         or radar.spacing_z_m > wavelength/2*(1+1e-10)),
            carrier_hz=float(radar.carrier_hz), bandwidth_hz=float(radar.bandwidth_hz),
            spacing_y_m=float(radar.spacing_y_m), spacing_z_m=float(radar.spacing_z_m)))


power_volume_from_iq = build_power_volume


def positions_to_ruv(positions):
    """Array-local XYZ -> range/u/v and explicit finite/front-halfspace mask."""
    if isinstance(positions, torch.Tensor):
        if positions.shape[-1:] != (3,):
            raise ValueError('Expected positions [...,3]')
        radius = positions.square().sum(-1).clamp_min(1e-24).sqrt()
        valid = torch.isfinite(positions).all(-1) & (positions[..., 0] > 0.) & (radius > 1e-6)
        return dict(ranges=radius, u=positions[..., 1]/radius,
                    v=positions[..., 2]/radius, valid=valid)
    positions = np.asarray(positions)
    if positions.shape[-1:] != (3,) or np.iscomplexobj(positions):
        raise ValueError('Expected real positions [...,3]')
    radius = np.maximum(np.sum(positions ** 2, -1), 1e-24) ** .5
    valid = np.isfinite(positions).all(-1) & (positions[..., 0] > 0.) & (radius > 1e-6)
    return dict(ranges=radius, u=positions[..., 1]/radius,
                v=positions[..., 2]/radius, valid=valid)


def _numpy_axis(axis, uniform=False):
    axis = np.asarray(axis, float)
    if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0.)):
        raise ValueError('Axes must be finite, strictly increasing vectors')
    if uniform and not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-6, atol=1e-10):
        raise ValueError('Periodic direction axes must be uniform native FFT axes')
    return axis


def _numpy_index(values, axis):
    lower = np.clip(np.searchsorted(axis, values, side='right')-1, 0, len(axis)-2)
    return lower + (values-axis[lower])/(axis[lower+1]-axis[lower])


def sample_power_volume_numpy(volume, *, ranges, u, v, range_axis, u_axis, v_axis,
                              angular_valid=None, candidate_valid=None):
    """Joint trilinear sample with periodic angular seams and zero invalid output."""
    volume = np.asarray(volume)
    if volume.ndim == 3:
        volume = volume[None]
    if volume.ndim != 4 or np.iscomplexobj(volume):
        raise ValueError('Volume must be real [R,U,V] or [C,R,U,V]')
    axes = [_numpy_axis(range_axis), _numpy_axis(u_axis, True), _numpy_axis(v_axis, True)]
    if volume.shape[1:] != tuple(len(a) for a in axes):
        raise ValueError('Volume shape and physical axes differ')
    ranges, u, v = np.broadcast_arrays(np.asarray(ranges, float), np.asarray(u, float), np.asarray(v, float))
    valid = (np.isfinite(ranges) & np.isfinite(u) & np.isfinite(v)
             & (ranges >= axes[0][0]) & (ranges <= axes[0][-1])
             & (u*u + v*v <= 1. + 1e-10))
    if candidate_valid is not None:
        candidate_valid = np.asarray(candidate_valid)
        if candidate_valid.shape != valid.shape or candidate_valid.dtype != np.bool_:
            raise ValueError('Candidate mask must be a shape-aligned boolean array')
        valid &= candidate_valid
    safe = [np.where(np.isfinite(a), a, 0.) for a in (ranges, u, v)]
    coords = [_numpy_index(safe[0], axes[0])]
    for query, axis in zip(safe[1:], axes[1:]):
        period = (axis[1]-axis[0])*len(axis)
        coords.append(np.mod(query-axis[0], period)/(axis[1]-axis[0]))
    shape = valid.shape
    coordinates = np.array([a.reshape(-1) for a in coords])
    padded = np.concatenate((volume, volume[..., :1]), -1)
    padded = np.concatenate((padded, padded[..., :1, :]), -2)
    sampled = np.stack([map_coordinates(c, coordinates, order=1, mode='constant',
                        cval=0., prefilter=False) for c in padded], -1).reshape(shape+(len(volume),))
    if angular_valid is not None:
        mask = np.asarray(angular_valid)
        if mask.dtype != np.bool_ or mask.shape not in (volume.shape[-2:], volume.shape[-3:]):
            raise ValueError('Volume mask must be boolean [U,V] or [R,U,V]')
        mask = np.broadcast_to(mask, volume.shape[-3:]).astype(float)
        mask = np.concatenate((mask, mask[..., :1]), -1)
        mask = np.concatenate((mask, mask[..., :1, :]), -2)
        weight = map_coordinates(mask, coordinates, order=1, mode='constant', cval=0., prefilter=False).reshape(shape)
        valid &= weight >= 1.-1e-6
    return dict(values=np.where(valid[..., None], sampled, 0.), valid=valid,
                coordinates=np.stack(coords, -1))


def _torch_axis(axis, reference, uniform=False):
    axis = torch.as_tensor(axis, dtype=reference.dtype, device=reference.device)
    if (axis.ndim != 1 or len(axis) < 2 or not bool(torch.isfinite(axis).all())
            or not bool((axis[1:] > axis[:-1]).all())):
        raise ValueError('Axes must be finite, strictly increasing vectors')
    if uniform and not torch.allclose(axis[1:]-axis[:-1], (axis[1]-axis[0]).expand(len(axis)-1), rtol=1e-5, atol=1e-7):
        raise ValueError('Periodic direction axes must be uniform native FFT axes')
    return axis


def sample_power_volume(volume, *, ranges, u, v, range_axis, u_axis, v_axis,
                        angular_valid=None, candidate_valid=None):
    """Differentiable joint sampling; output ``values[query_shape..., channels]``.

    grid_sample input is [N,C,R,U,V], so its grid order is (v,u,R). ``u``/``v``
    wrap only the FFT seam; physical disk/front masks are evaluated separately.
    Pass ``positions_to_ruv(...)[valid]`` as candidate_valid to reject back rays.
    """
    if not isinstance(volume, torch.Tensor) or not volume.is_floating_point():
        raise ValueError('Volume must be a real floating tensor')
    if volume.ndim == 3:
        volume = volume[None]
    if volume.ndim != 4:
        raise ValueError('Volume must have shape [R,U,V] or [C,R,U,V]')
    queries = [torch.as_tensor(a, dtype=volume.dtype, device=volume.device) for a in (ranges, u, v)]
    ranges, u, v = torch.broadcast_tensors(*queries)
    axes = [_torch_axis(range_axis, volume), _torch_axis(u_axis, volume, True),
            _torch_axis(v_axis, volume, True)]
    if volume.shape[1:] != tuple(len(a) for a in axes):
        raise ValueError('Volume shape and physical axes differ')
    valid = (torch.isfinite(ranges) & torch.isfinite(u) & torch.isfinite(v)
             & (ranges >= axes[0][0]) & (ranges <= axes[0][-1])
             & (u*u+v*v <= 1.+1e-6))
    if candidate_valid is not None:
        mask = torch.as_tensor(candidate_valid, device=volume.device)
        if mask.dtype != torch.bool or mask.shape != valid.shape:
            raise ValueError('Candidate mask must be a shape-aligned boolean tensor')
        valid = valid & mask
    safe = [torch.where(torch.isfinite(a), a, torch.zeros_like(a)) for a in (ranges, u, v)]
    axis = axes[0]
    lower = torch.searchsorted(axis, safe[0].contiguous(), right=True).sub(1).clamp(0, len(axis)-2)
    ri = lower.to(volume.dtype)+(safe[0]-axis[lower])/(axis[lower+1]-axis[lower])
    grid_r = 2.*ri/(len(axis)-1)-1.
    grids = []
    for query, axis in zip(safe[1:], axes[1:]):
        period = (axis[1]-axis[0])*len(axis)
        index = torch.remainder(query-axis[0], period)/(axis[1]-axis[0])
        grids.append(2.*index/len(axis)-1.)
    grid = torch.stack((grids[1], grids[0], grid_r), -1)
    if ranges.numel() == 0:
        return dict(values=volume.new_empty(tuple(ranges.shape)+(len(volume),)), valid=valid, grid=grid)
    # Add one periodic endpoint, not a zero-padding angular boundary.
    padded = torch.cat((volume, volume[..., :1]), -1)
    padded = torch.cat((padded, padded[..., :1, :]), -2)
    flat = grid.reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(padded[None], flat, mode='bilinear', padding_mode='zeros',
                            align_corners=True)[0, :, 0, 0].T.reshape(tuple(ranges.shape)+(len(volume),))
    if angular_valid is not None:
        mask = torch.as_tensor(angular_valid, device=volume.device)
        if mask.dtype != torch.bool or tuple(mask.shape) not in (tuple(volume.shape[-2:]), tuple(volume.shape[-3:])):
            raise ValueError('Volume mask must be boolean [U,V] or [R,U,V]')
        mask = torch.broadcast_to(mask, volume.shape[-3:]).to(volume.dtype)
        mask = torch.cat((mask, mask[..., :1]), -1)
        mask = torch.cat((mask, mask[..., :1, :]), -2)
        weight = F.grid_sample(mask[None, None], flat, mode='bilinear', padding_mode='zeros',
                               align_corners=True)[0, 0, 0, 0].reshape(ranges.shape)
        valid = valid & (weight >= 1.-1e-6)
    return dict(values=torch.where(valid[..., None], sampled, torch.zeros_like(sampled)),
                valid=valid, grid=grid)
