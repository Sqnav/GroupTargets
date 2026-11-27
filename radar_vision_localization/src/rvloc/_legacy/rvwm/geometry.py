"""Calibrated pinhole geometry; camera FRD and world NED are explicit.

K uses the standard optical vector [right, down, forward]. Camera.rotation
maps [forward, right, down] to world NED, matching AirSim camera quaternions.
"""
import numpy as np
import torch


def quaternion_rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Camera quaternion must be a finite nonzero wxyz vector')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project(positions, camera):
    p = np.asarray(positions, dtype=float)
    local = (p - np.asarray(camera.position)) @ np.asarray(camera.rotation)
    optical = local[..., [1, 2, 0]]
    pixels = optical @ np.asarray(camera.K).T
    with np.errstate(invalid='ignore', divide='ignore'):
        result = pixels[..., :2] / pixels[..., 2:3]
    return np.where(local[..., :1] > 0, result, np.nan)


def rays(uv, camera):
    uv = np.asarray(uv, dtype=float)
    optical = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1) @ np.linalg.inv(camera.K).T
    frd = optical[..., [2, 0, 1]]
    world = frd @ np.asarray(camera.rotation).T
    return world / np.linalg.norm(world, axis=-1, keepdims=True)


def torch_project(positions, camera):
    """Differentiable projection. Solver must enforce positive camera depth."""
    args = dict(dtype=positions.dtype, device=positions.device)
    local = (positions - torch.as_tensor(camera.position, **args)) @ torch.as_tensor(camera.rotation, **args)
    optical = local[..., [1, 2, 0]]
    pixels = optical @ torch.as_tensor(camera.K, **args).T
    denominator = pixels[..., 2:3]
    denominator = torch.where(denominator.abs() < 1e-8,
                              torch.full_like(denominator, 1e-8), denominator)
    return pixels[..., :2] / denominator


def sphere_ray_interval(origin, direction, center, radius):
    """Positive slant-depth interval inside a public sensor range sphere."""
    offset = np.asarray(origin) - np.asarray(center)
    b = float(np.dot(offset, direction))
    disc = b*b - float(offset @ offset) + float(radius)**2
    if disc <= 0:return None
    lo, hi = -b-np.sqrt(disc), -b+np.sqrt(disc)
    lo = max(lo, .1)
    return (float(lo), float(hi)) if hi > lo else None
