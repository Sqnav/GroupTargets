"""Pure radar position/velocity association; no image inputs or target roster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import _scalar


@dataclass
class _RadarTrack:
    position: np.ndarray
    velocity: np.ndarray
    last_time: float
    hits: int = 1
    missed: int = 0


class RadarAssociator:
    """One-to-one Hungarian matching with explicit unmatched alternatives.

    Predicted-position and measured-velocity residuals use only current FFT
    positions and earlier radar observations. Invalid input rows receive None;
    unmatched finite observations receive fresh episode-local IDs. No source
    count is supplied or inferred to limit the number of observations.
    """
    def __init__(self, position_scale_m=10.0, velocity_scale_m_s=10.0,
                 position_gate_m=30.0, velocity_gate_m_s=30.0,
                 velocity_weight=0.25, cost_gate=3.0,
                 ema_velocity=0.5, max_missed=5):
        self.position_scale_m = _scalar(position_scale_m, "position_scale_m")
        self.velocity_scale_m_s = _scalar(velocity_scale_m_s, "velocity_scale_m_s")
        self.position_gate_m = _scalar(position_gate_m, "position_gate_m")
        self.velocity_gate_m_s = _scalar(velocity_gate_m_s, "velocity_gate_m_s")
        self.cost_gate = _scalar(cost_gate, "cost_gate")
        if min(self.position_scale_m, self.velocity_scale_m_s, self.cost_gate) <= 0:
            raise ValueError("Cost scales and cost_gate must be positive")
        self.velocity_weight = _scalar(velocity_weight, "velocity_weight")
        self.ema_velocity = _scalar(ema_velocity, "ema_velocity")
        if max(self.velocity_weight, self.ema_velocity) > 1.0:
            raise ValueError("velocity_weight and ema_velocity must lie in [0,1]")
        if isinstance(max_missed, bool) or not isinstance(max_missed, (int, np.integer)) or max_missed < 0:
            raise ValueError("max_missed must be a nonnegative integer")
        self.max_missed = int(max_missed)
        self.reset()

    def reset(self):
        self.tracks = {}
        self.time = None
        self.next_id = 1
        self.last_result = None

    def step(self, time, points):
        now = _scalar(time, "time", minimum=-np.inf)
        if self.time is not None and now <= self.time:
            raise ValueError("Radar association time must strictly increase")
        points = np.asarray(points, float)
        if points.ndim == 1 and points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Radar association expects positions [N,3]")
        active = np.flatnonzero(np.isfinite(points).all(axis=1))
        identities = np.full(len(points), None, dtype=object)
        tracks = sorted(self.tracks)
        costs = np.full((len(tracks), len(active)), np.inf)
        components = {}
        for row, identity in enumerate(tracks):
            track = self.tracks[identity]
            elapsed = now - track.last_time
            predicted = track.position + track.velocity * elapsed
            for col, index in enumerate(active):
                position_error = float(np.linalg.norm(points[index] - predicted))
                velocity_error = float(np.linalg.norm((points[index] - track.position) / elapsed - track.velocity))
                cost = ((1.0 - self.velocity_weight) * position_error / self.position_scale_m
                        + self.velocity_weight * velocity_error / self.velocity_scale_m_s)
                components[row, col] = dict(position_residual_m=position_error,
                                            velocity_residual_m_s=velocity_error)
                if (position_error <= self.position_gate_m and velocity_error <= self.velocity_gate_m_s
                        and cost <= self.cost_gate):
                    costs[row, col] = cost
        matches = []
        if tracks and len(active):
            forbidden = (self.cost_gate + 1.0) * (len(tracks) + len(active) + 1)
            augmented = np.full((len(tracks), len(active) + len(tracks)), forbidden)
            augmented[:, :len(active)] = np.where(np.isfinite(costs), costs, forbidden)
            augmented[np.arange(len(tracks)), len(active) + np.arange(len(tracks))] = self.cost_gate + 1e-9
            rows, columns = linear_sum_assignment(augmented)
            for row, col in zip(rows, columns):
                if col >= len(active) or not np.isfinite(costs[row, col]):
                    continue
                identity, index = tracks[row], int(active[col])
                track = self.tracks[identity]
                measured_velocity = (points[index] - track.position) / (now - track.last_time)
                track.velocity = (measured_velocity if track.hits == 1 else
                                  (1 - self.ema_velocity) * track.velocity + self.ema_velocity * measured_velocity)
                track.position = points[index].copy()
                track.last_time = now
                track.hits += 1
                track.missed = 0
                identities[index] = identity
                matches.append(dict(track_id=identity, detection_index=index, cost=float(costs[row, col]),
                                    **components[row, col]))
        used_ids = set(identities[identities != None])  # noqa: E711
        for identity in tracks:
            if identity not in used_ids:
                self.tracks[identity].missed += 1
                if self.tracks[identity].missed > self.max_missed:
                    del self.tracks[identity]
        for index in active:
            if identities[index] is not None:
                continue
            identity = self.next_id
            self.next_id += 1
            self.tracks[identity] = _RadarTrack(points[index].copy(), np.zeros(3), now)
            identities[index] = identity
        self.time = now
        self.last_result = dict(time=now, ids=identities.tolist(), associations=matches,
                                current_observations=len(points), uses_rgb=False,
                                uses_target_ids=False, cost_inputs=["radar_position", "radar_velocity"])
        return identities
