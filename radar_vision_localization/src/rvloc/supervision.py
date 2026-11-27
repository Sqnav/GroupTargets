"""Explicit training-only ray supervision for the direct-depth models.

The public optical depth interval defines supervision support. A feature
normalization scale is never treated as a station visibility/range bound.
Targets are supplied separately after forward; target identities and motion
are not arguments and never become normal model inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _array(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError("Supervision geometry must be real: " + name)
    return value


def match_direct_ray_labels(features, positions, config=None):
    """Partial angular assignment inside the same public camera-ray support.

    Returns the legacy rows/cols/depths_m structure for the training adapter.
    No base-frame x sign, station-distance radius or radar detection evidence
    restricts a visually supported target. The angle gate is configured before
    evaluation, not a trained confidence estimate.
    """
    config = {} if config is None else config
    raw_gate = config.get("group_supervision_max_angle_deg", 3.)
    if isinstance(raw_gate, (bool, str)):
        raise ValueError("Supervision angle gate must be a finite number")
    gate = float(raw_gate)
    if not np.isfinite(gate) or not 0. < gate < 90.:
        raise ValueError("Supervision angle gate must lie in (0,90) degrees")
    origins = _array(features["ray_origins"], "ray_origins").astype(float)
    directions = _array(features["ray_directions"], "ray_directions").astype(float)
    valid = _array(features["ray_valid"], "ray_valid")
    bounds = _array(features["ray_depth_bounds"], "ray_depth_bounds").astype(float)
    scale = float(features["range_scale_m"])
    true = _array(positions, "positions").astype(float)
    if true.size == 0:
        true = true.reshape(0, 3)
    n = len(origins)
    if (origins.shape != (n, 3) or directions.shape != (n, 3) or valid.shape != (n,)
            or valid.dtype != np.bool_ or bounds.shape != (n, 2)
            or not np.isfinite(scale) or scale <= 0
            or not np.isfinite(origins).all() or not np.isfinite(directions).all()
            or not np.isfinite(bounds).all()):
        raise ValueError("Finite calibrated public ray geometry is required for direct-depth supervision")
    if true.ndim != 2 or true.shape[1:] != (3,) or not np.isfinite(true).all():
        raise ValueError("Separate finite 3D supervision [M,3] required")
    if (not np.allclose(np.linalg.norm(directions[valid], axis=-1), 1., atol=1e-6, rtol=1e-6)
            or np.any(bounds[valid, 0] < 0) or np.any(bounds[valid, 1] <= bounds[valid, 0])):
        raise ValueError("Unit valid rays with positive nonempty optical support required")
    active = np.flatnonzero(valid)
    tolerance = max(1e-6, scale*1e-7)
    delta = true[None] - origins[active, None]
    distance = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(distance[..., None], 1e-12)
    cosine = np.clip(np.sum(unit*directions[active, None], axis=-1), -1., 1.)
    angular = np.degrees(np.arccos(cosine))
    depth = np.sum(delta*directions[active, None], axis=-1)
    angle_ok = (distance > 1e-12) & (angular <= gate+1e-10)
    depth_ok = ((depth >= bounds[active, 0, None]-tolerance)
                & (depth <= bounds[active, 1, None]+tolerance))
    feasible = angle_ok & depth_ok
    row, col = np.empty(0, np.int64), np.empty(0, np.int64)
    if len(active) and len(true):
        unmatched_cost = float(len(active)+1)
        cost = np.full((len(active), len(true)+len(active)), unmatched_cost)
        cost[:, :len(true)] = np.where(feasible, angular/gate, unmatched_cost*(len(active)+2))
        assigned_rows, assigned_cols = linear_sum_assignment(cost)
        selected = assigned_cols < len(true)
        row, col = assigned_rows[selected], assigned_cols[selected]
        if len(row):
            selected = feasible[row, col]
            row, col = row[selected], col[selected]
    rows = active[row]
    diagnostics = dict(supervision_usable=bool(len(rows)), matched_training_count=len(rows),
        training_detected_count=n, valid_training_rays=len(active), label_member_count=len(true),
        unmatched_detection_count=len(active)-len(rows), invalid_detection_count=n-len(active),
        unmatched_label_count=len(true)-len(rows), angle_gate_deg=gate,
        angle_gate_status="configured engineering rejection threshold; not calibrated",
        supervision_domain="public_camera_ray_support", station_radius_filter_applied=False,
        station_front_halfspace_filter_applied=False, rejected_angle_pairs=int((~angle_ok).sum()),
        rejected_support_pairs=int((~depth_ok).sum()), depth_outside_support_pair_count=int((~depth_ok).sum()),
        outside_support_label_count=int((~depth_ok.any(axis=0)).sum()),
        matched_angle_errors_deg=angular[row, col].tolist(),
        rejection_reason=None if len(rows) else "no_matches_within_angle_and_public_optical_support")
    return dict(rows=rows.astype(np.int64), cols=col.astype(np.int64),
                depths_m=depth[row, col], diagnostics=diagnostics)
