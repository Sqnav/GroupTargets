"""Differentiable sampling on calibrated physical RA/RE axes.

These are sensor-coordinate operations, not assignments between members and
radar peaks. Both projections sample the same candidate slant range.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _axis_index(values, axis):
    axis = torch.as_tensor(axis, dtype=values.dtype, device=values.device)
    if axis.ndim != 1 or len(axis) < 2 or not bool(torch.isfinite(axis).all()) or not bool((axis[1:] > axis[:-1]).all()):
        raise ValueError('Physical axes must be finite, strictly increasing vectors')
    lower = torch.searchsorted(axis, values.contiguous(), right=True).sub(1).clamp(0, len(axis)-2)
    index = lower.to(values.dtype) + (values-axis[lower])/(axis[lower+1]-axis[lower])
    valid = torch.isfinite(values) & (values >= axis[0]) & (values <= axis[-1])
    return 2.*index/(len(axis)-1)-1., valid


def physical_grid(ranges, angles, range_axis, angle_axis):
    """Return grid_sample's (angle-x, range-y) grid and in-axis validity.

    Works for nonuniform physical axes too; align_corners=True is required.
    Values are not clipped into the image: outside coordinates stay invalid.
    """
    if ranges.shape != angles.shape:
        raise ValueError('Range and angle query shapes must match')
    y, ry = _axis_index(ranges, range_axis)
    x, ax = _axis_index(angles, angle_axis)
    grid = torch.stack((x, y), -1)
    return grid, ry & ax


def candidate_geometry(origins, directions, depths, depth_bounds, ray_valid):
    """Camera slant depths [N,S] -> fixed-station positions and angles."""
    if depths.ndim != 2 or origins.shape != (len(depths), 3) or directions.shape != origins.shape:
        raise ValueError('Expected origins/directions [N,3] and depths [N,S]')
    if depth_bounds.shape != (len(depths), 2) or ray_valid.shape != (len(depths),):
        raise ValueError('Ray support/validity shapes differ')
    positions = origins[:, None] + depths[..., None]*directions[:, None]
    radius = positions.square().sum(-1).clamp_min(1e-12).sqrt()
    horizontal = positions[..., :2].square().sum(-1).clamp_min(1e-12).sqrt()
    azimuth = torch.atan2(positions[..., 1], positions[..., 0])
    elevation = torch.atan2(positions[..., 2], horizontal)
    valid = (ray_valid[:, None] & torch.isfinite(depths) &
             (depths >= depth_bounds[:, 0, None]) & (depths <= depth_bounds[:, 1, None]) &
             (depths > 0.) & (positions[..., 0] > 0.) & (radius > 1e-6))
    return dict(positions=positions, ranges=radius, azimuth=azimuth, elevation=elevation, valid=valid)


def _sample(feature_map, mask, grid, in_axis, candidate_valid):
    if feature_map.ndim != 3 or mask.shape != feature_map.shape[-2:]:
        raise ValueError('Expected feature map [C,range,angle] and matching mask')
    n, count = grid.shape[:2]
    if n == 0:
        return feature_map.new_empty((0, count, feature_map.shape[0])), in_axis
    # Sanitize before grid_sample; an invalid NaN must not contaminate an
    # otherwise masked result. Physical validity is retained separately.
    finite = torch.isfinite(grid).all(-1)
    safe = torch.where(finite[..., None], grid, torch.full_like(grid, 2.))
    sampled = F.grid_sample(feature_map[None], safe[None], mode='bilinear',
                            padding_mode='zeros', align_corners=True)[0].permute(1, 2, 0)
    weight = F.grid_sample(mask[None, None].to(feature_map.dtype), safe[None],
                           mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
    valid = in_axis & finite & candidate_valid & (weight >= 1.-1e-6)
    return torch.where(valid[..., None], sampled, torch.zeros_like(sampled)), valid


def sample_paired_spectra(ra_features, re_features, *, ranges, azimuth, elevation,
                          range_axis, azimuth_axis, elevation_axis, ra_valid, re_valid,
                          candidate_valid=None):
    """Sample RA/RE features jointly without treating them as likelihoods."""
    if candidate_valid is None:
        candidate_valid = torch.ones_like(ranges, dtype=torch.bool)
    ra_grid, ra_axis_valid = physical_grid(ranges, azimuth, range_axis, azimuth_axis)
    re_grid, re_axis_valid = physical_grid(ranges, elevation, range_axis, elevation_axis)
    ra, av = _sample(ra_features, ra_valid, ra_grid, ra_axis_valid, candidate_valid)
    re, ev = _sample(re_features, re_valid, re_grid, re_axis_valid, candidate_valid)
    return dict(ra=ra, re=re, ra_valid=av, re_valid=ev, valid=av | ev,
                joint_valid=av & ev, ra_grid=ra_grid, re_grid=re_grid)
