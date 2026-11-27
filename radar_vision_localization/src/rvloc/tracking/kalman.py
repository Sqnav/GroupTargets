"""Six-state Cartesian constant-velocity Kalman bank.

State order is [x,y,z,vx,vy,vz]. Association IDs are supplied by the relevant
sensor-only associator; positions never feed back into visual association.
Every returned row corresponds to an input observation, including invalid or
unassociated rows. The bank does not emit prediction-only detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _scalar(value, name, minimum=0.0):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def _covariance(value, size, name):
    value = np.asarray(value, float)
    if (value.shape != (size, size) or not np.isfinite(value).all()
            or not np.allclose(value, value.T, rtol=1e-8, atol=1e-10)
            or np.linalg.eigvalsh(value).min() < -1e-10):
        raise ValueError(f"{name} must be a finite symmetric positive-semidefinite covariance")
    return value


def cv_predict(state, covariance, dt, acceleration_variance=1.0):
    """Predict using piecewise-constant, independent Cartesian accelerations."""
    dt = _scalar(dt, "dt")
    variance = _scalar(acceleration_variance, "acceleration_variance")
    state = np.asarray(state, float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite six-vector")
    covariance = _covariance(covariance, 6, "state covariance")
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    process = variance * np.block([
        [np.eye(3) * dt ** 4 / 4.0, np.eye(3) * dt ** 3 / 2.0],
        [np.eye(3) * dt ** 3 / 2.0, np.eye(3) * dt ** 2],
    ])
    predicted = transition @ state
    propagated = transition @ covariance @ transition.T + process
    return predicted, (propagated + propagated.T) / 2.0


def position_update(state, covariance, position, measurement_covariance):
    """Position-only measurement update, using Joseph covariance form."""
    state = np.asarray(state, float)
    position = np.asarray(position, float)
    if state.shape != (6,) or position.shape != (3,) or not np.isfinite(state).all() or not np.isfinite(position).all():
        raise ValueError("Finite state[6] and position[3] are required")
    covariance = _covariance(covariance, 6, "state covariance")
    noise = _covariance(measurement_covariance, 3, "measurement covariance")
    innovation = position - state[:3]
    total = covariance[:3, :3] + noise
    if np.linalg.eigvalsh(total).min() <= 0.0:
        raise ValueError("Innovation covariance must be positive definite")
    gain = np.linalg.solve(total, covariance[:3, :]).T
    updated = state + gain @ innovation
    residual = np.eye(6)
    residual[:, :3] -= gain
    posterior = residual @ covariance @ residual.T + gain @ noise @ gain.T
    return updated, (posterior + posterior.T) / 2.0


@dataclass
class _State:
    x: np.ndarray
    covariance: np.ndarray
    last_observation_time: float


class KalmanBank:
    def __init__(self, acceleration_variance=1.0, measurement_variance=4.0,
                 initial_velocity_variance=100.0, max_gap_s=None):
        self.acceleration_variance = _scalar(acceleration_variance, "acceleration_variance")
        self.measurement_variance = _scalar(measurement_variance, "measurement_variance")
        self.initial_velocity_variance = _scalar(initial_velocity_variance, "initial_velocity_variance")
        if self.measurement_variance <= 0:
            raise ValueError("measurement_variance must be positive")
        self.max_gap_s = None if max_gap_s is None else _scalar(max_gap_s, "max_gap_s")
        self.reset()

    def reset(self):
        self.states = {}
        self.time = None
        self.last_valid = np.empty(0, dtype=bool)
        self.last_velocities = np.empty((0, 3), dtype=float)

    def update(self, time, ids, positions, covariance=None, valid=None):
        now = _scalar(time, "time", minimum=-np.inf)
        if self.time is not None and now <= self.time:
            raise ValueError("Kalman time must strictly increase within an episode")
        points = np.asarray(positions, float)
        if points.ndim == 1 and points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("positions must have shape [N,3]")
        identities = list(ids)
        if len(identities) != len(points):
            raise ValueError("There must be one estimated ID per observation")
        assigned = [identity for identity in identities if identity is not None]
        try:
            if len(set(assigned)) != len(assigned):
                raise ValueError("Each assigned member ID may occur at most once per frame")
        except TypeError as error:
            raise ValueError("Estimated IDs must be hashable") from error
        if valid is None:
            active = np.isfinite(points).all(axis=1)
        else:
            active = np.asarray(valid)
            if active.dtype != np.bool_ or active.shape != (len(points),):
                raise ValueError("valid must be a boolean [N] mask")
            active = active & np.isfinite(points).all(axis=1)
        if covariance is None:
            noise = np.broadcast_to(np.eye(3) * self.measurement_variance, (len(points), 3, 3))
        else:
            noise = np.asarray(covariance, float)
            if noise.shape == (3, 3):
                noise = np.broadcast_to(noise, (len(points), 3, 3))
            if noise.shape != (len(points), 3, 3):
                raise ValueError("covariance must have shape [3,3] or [N,3,3]")
            for index in np.flatnonzero(active):
                _covariance(noise[index], 3, "measurement covariance")
        # Validate before mutating state so rejected calls remain retryable.
        for identity in assigned:
            if isinstance(identity, (bool, np.bool_)):
                raise ValueError("Boolean IDs are not member identities")
        if self.max_gap_s is not None:
            for identity, state in list(self.states.items()):
                if now - state.last_observation_time > self.max_gap_s:
                    del self.states[identity]
        if self.time is not None:
            for state in self.states.values():
                state.x, state.covariance = cv_predict(state.x, state.covariance, now - self.time,
                                                       self.acceleration_variance)
        filtered = points.copy()
        velocities = np.full((len(points), 3), np.nan)
        for index, identity in enumerate(identities):
            if not active[index] or identity is None:
                continue
            if identity not in self.states:
                initial = np.zeros((6, 6))
                initial[:3, :3] = noise[index]
                initial[3:, 3:] = np.eye(3) * self.initial_velocity_variance
                state = _State(np.r_[points[index], np.zeros(3)], initial, now)
                self.states[identity] = state
            else:
                state = self.states[identity]
                state.x, state.covariance = position_update(state.x, state.covariance, points[index], noise[index])
                state.last_observation_time = now
            filtered[index] = state.x[:3]
            velocities[index] = state.x[3:]
        self.time = now
        self.last_valid = active.copy()
        self.last_velocities = velocities
        return filtered
