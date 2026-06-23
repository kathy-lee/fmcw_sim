"""Cartesian world-frame tracker (M4): a CTRV-EKF with Doppler.

State  x = [x, y, v, ψ, ω]   (m, m, m/s, rad, rad/frame)
Meas   z = [r, θ, ṙ]         (m, rad, m/s)  -- Doppler always included.

Δt is folded as 1 frame (one frame = one time step), matching the radial
processing.tracking.Tracker. Lifecycle/association mirror that module; no
shared base is extracted (umbrella §3.2 -- only one Cartesian tracker exists).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from processing.clustering import ClusterObject
from processing.config import CartesianTrackerConfig

_EPS_OMEGA = 1e-4    # |ω| below this uses the straight-line limit


def wrap_to_pi(a: float) -> float:
    """Wrap an angle (or array) to (-π, π]."""
    return np.pi - (np.pi - a) % (2.0 * np.pi)


def ctrv_f(state: np.ndarray) -> np.ndarray:
    """CTRV one-frame state transition (Δt = 1)."""
    x, y, v, psi, w = state
    if abs(w) < _EPS_OMEGA:
        # First-order-in-ω straight-line limit (keeps f consistent with ctrv_F's
        # analytic ω-column and avoids the v/ω singularity).
        s, c = np.sin(psi), np.cos(psi)
        return np.array([x + v * c - 0.5 * v * w * s,
                         y + v * s + 0.5 * v * w * c,
                         v, wrap_to_pi(psi + w), w])
    s, c = np.sin(psi), np.cos(psi)
    sw, cw = np.sin(psi + w), np.cos(psi + w)
    return np.array([x + (v / w) * (sw - s),
                     y + (v / w) * (c - cw),
                     v, wrap_to_pi(psi + w), w])


def ctrv_F(state: np.ndarray) -> np.ndarray:
    """Jacobian ∂f/∂x (5×5)."""
    x, y, v, psi, w = state
    s, c = np.sin(psi), np.cos(psi)
    F = np.eye(5)
    if abs(w) < _EPS_OMEGA:
        F[0, 2], F[0, 3], F[0, 4] = c, -v * s, -0.5 * v * s
        F[1, 2], F[1, 3], F[1, 4] = s, v * c, 0.5 * v * c
        F[3, 4] = 1.0
        return F
    sw, cw = np.sin(psi + w), np.cos(psi + w)
    F[0, 2] = (sw - s) / w
    F[0, 3] = (v / w) * (cw - c)
    F[0, 4] = v * (cw * w - (sw - s)) / (w * w)
    F[1, 2] = (c - cw) / w
    F[1, 3] = (v / w) * (sw - s)
    F[1, 4] = v * (sw * w - (c - cw)) / (w * w)
    F[3, 4] = 1.0
    return F


def cartesian_h(state: np.ndarray) -> np.ndarray:
    """Measurement h(x) = [r, θ, ṙ]; ṙ = v·cos(ψ−θ)."""
    x, y, v, psi, _w = state
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return np.array([r, theta, v * np.cos(psi - theta)])


def cartesian_H(state: np.ndarray) -> np.ndarray:
    """Jacobian ∂h/∂x (3×5)."""
    x, y, v, psi, _w = state
    r2 = x * x + y * y
    r = np.sqrt(r2)
    theta = np.arctan2(y, x)
    sd = np.sin(psi - theta)
    cd = np.cos(psi - theta)
    H = np.zeros((3, 5))
    H[0, 0], H[0, 1] = x / r, y / r
    H[1, 0], H[1, 1] = -y / r2, x / r2
    H[2, 0] = -v * y * sd / r2
    H[2, 1] = v * x * sd / r2
    H[2, 2] = cd
    H[2, 3] = -v * sd
    return H


def ctrv_Q(state: np.ndarray, sigma_a: float, sigma_yaw: float) -> np.ndarray:
    """CTRV process noise from longitudinal-accel std σ_a and yaw-accel std
    σ_yaw, mapped through the standard noise-input matrix G (Δt = 1)."""
    psi = state[3]
    G = np.array([[0.5 * np.cos(psi), 0.0],
                  [0.5 * np.sin(psi), 0.0],
                  [1.0,               0.0],
                  [0.0,               0.5],
                  [0.0,               1.0]])
    D = np.diag([sigma_a * sigma_a, sigma_yaw * sigma_yaw])
    return G @ D @ G.T


def ekf_predict(state, cov, Q):
    F = ctrv_F(state)
    return ctrv_f(state), F @ cov @ F.T + Q


def _innovation(state, z):
    """z − h(x) with the azimuth component wrapped to (-π, π]."""
    y = z - cartesian_h(state)
    y[1] = wrap_to_pi(y[1])
    return y


def ekf_update(state, cov, z, R):
    H = cartesian_H(state)
    y = _innovation(state, z)
    S = H @ cov @ H.T + R
    K = cov @ H.T @ np.linalg.inv(S)
    new_state = state + K @ y
    new_state[3] = wrap_to_pi(new_state[3])
    new_cov = (np.eye(5) - K @ H) @ cov
    new_cov = 0.5 * (new_cov + new_cov.T)         # numerical symmetry
    assert np.all(np.isfinite(new_cov)), f"non-finite cov: {new_cov}"
    return new_state, new_cov


def mahalanobis(state, cov, z, R):
    H = cartesian_H(state)
    y = _innovation(state, z)
    S = H @ cov @ H.T + R
    return float(y @ np.linalg.solve(S, y))


def object_to_measurement(obj: ClusterObject) -> np.ndarray:
    return np.array([np.hypot(obj.x, obj.y),
                     np.arctan2(obj.y, obj.x), obj.v_radial])


@dataclass
class CartesianTrack:
    id: int
    state: np.ndarray                 # (5,) [x, y, v, ψ, ω]
    cov: np.ndarray                   # (5, 5)
    hits: int
    misses: int
    age: int
    status: str                       # "tentative" | "confirmed" | "coasting"
    hit_history: Optional[deque] = None
    history: Optional[deque] = None   # last N (x, y) estimates
    assoc_det_idx: Optional[int] = None


def _associate_nn(cost, feasible):
    out, used_t, used_d = [], set(), set()
    flat = [(cost[ti, di], ti, di)
            for ti in range(cost.shape[0]) for di in range(cost.shape[1])
            if feasible[ti, di]]
    flat.sort()
    for _, ti, di in flat:
        if ti in used_t or di in used_d:
            continue
        out.append((ti, di)); used_t.add(ti); used_d.add(di)
    return out


def _associate_gnn(cost, feasible, gate):
    big = float(gate) * 1e6 + 1.0
    c = np.where(feasible, cost, big)
    ti_idx, di_idx = linear_sum_assignment(c)
    return [(int(ti), int(di)) for ti, di in zip(ti_idx, di_idx)
            if feasible[ti, di]]


class CartesianTracker:
    """Cartesian CTRV-EKF multi-target tracker. step(objects) -> tracks.
    Lifecycle/association mirror processing.tracking.Tracker."""

    def __init__(self, config: CartesianTrackerConfig):
        self.cfg = config
        self.R = np.diag([config.sigma_r ** 2, config.sigma_az ** 2,
                          config.sigma_rdot ** 2])
        self._next_id = 0
        self.tracks: list = []

    def step(self, objects: list) -> list:
        for t in self.tracks:
            Q = ctrv_Q(t.state, self.cfg.sigma_a, self.cfg.sigma_yaw_accel)
            t.state, t.cov = ekf_predict(t.state, t.cov, Q)
            t.age += 1
            t.assoc_det_idx = None

        assignments = self._associate(objects)

        assigned_t, assigned_d = set(), set()
        for ti, di in assignments:
            self._update_track(self.tracks[ti], objects[di])
            self.tracks[ti].assoc_det_idx = di
            assigned_t.add(ti); assigned_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti in assigned_t:
                continue
            t.misses += 1
            if t.hit_history is not None:
                t.hit_history.append(False)

        for di, obj in enumerate(objects):
            if di in assigned_d:
                continue
            self._spawn(obj, di)

        self.tracks = [t for t in self.tracks if self._lifecycle_keep(t)]
        return list(self.tracks)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    def _associate(self, objects):
        if not self.tracks or not objects:
            return []
        gate = self.cfg.gate_threshold
        nT, nD = len(self.tracks), len(objects)
        cost = np.full((nT, nD), np.inf)
        feasible = np.zeros((nT, nD), dtype=bool)
        zs = [object_to_measurement(o) for o in objects]
        for ti, t in enumerate(self.tracks):
            for di, z in enumerate(zs):
                d2 = mahalanobis(t.state, t.cov, z, self.R)
                if d2 <= gate:
                    cost[ti, di] = d2
                    feasible[ti, di] = True
        if self.cfg.method == "nn":
            return _associate_nn(cost, feasible)
        return _associate_gnn(cost, feasible, gate)

    def _update_track(self, t, obj):
        z = object_to_measurement(obj)
        t.state, t.cov = ekf_update(t.state, t.cov, z, self.R)
        t.hits += 1
        t.misses = 0
        if t.hit_history is not None:
            t.hit_history.append(True)
        if t.history is not None:
            t.history.append((float(t.state[0]), float(t.state[1])))

    def _spawn(self, obj, det_idx):
        cfg = self.cfg
        theta = np.arctan2(obj.y, obj.x)
        state = np.array([obj.x, obj.y, obj.v_radial, theta, 0.0])
        cov = np.diag([cfg.init_pos_var, cfg.init_pos_var, cfg.init_v_var,
                       cfg.init_psi_var, cfg.init_omega_var])
        t = CartesianTrack(
            id=self._next_id, state=state, cov=cov,
            hits=1, misses=0, age=1, status="tentative",
            hit_history=deque([True], maxlen=cfg.confirm_N),
            history=deque([(float(obj.x), float(obj.y))], maxlen=cfg.history_len),
            assoc_det_idx=det_idx)
        self._next_id += 1
        self.tracks.append(t)

    def _lifecycle_keep(self, t) -> bool:
        cfg = self.cfg
        if t.status == "tentative":
            hits_in_window = sum(t.hit_history) if t.hit_history else 0
            if hits_in_window >= cfg.confirm_M:
                t.status = "confirmed"
                return True
            return t.misses <= cfg.max_misses_tentative
        if t.status == "confirmed":
            if t.misses == 0:
                return True
            t.status = "coasting"
            return True
        if t.status == "coasting":
            if t.misses == 0:
                t.status = "confirmed"
                return True
            return t.misses < cfg.max_coast
        return True
