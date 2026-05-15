"""Frame-to-frame tracking with constant-velocity Kalman filter.

Public surface: `Track` (dataclass) and `Tracker` (stateful, owns track list).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from processing.clustering import Detection
from processing.config import TrackerConfig

_F = np.array([[1.0, 1.0], [0.0, 1.0]])
_Q_BASE = np.array([[1.0 / 4.0, 1.0 / 2.0], [1.0 / 2.0, 1.0]])
_H = np.array([[1.0, 0.0]])
_I = np.eye(2)


@dataclass
class Track:
    id: int
    state: np.ndarray                # (2,) [range_bin, range_rate]
    cov: np.ndarray                  # (2, 2)
    last_doppler_bin: float
    hits: int
    misses: int
    age: int                         # frames since spawn
    status: str                      # "tentative" | "confirmed" | "coasting"
    hit_history: Optional[deque] = None
    history: Optional[deque] = None  # last N (range_bin, doppler_bin) pairs


def _kf_predict(state: np.ndarray, cov: np.ndarray, q: float):
    new_state = _F @ state
    new_cov = _F @ cov @ _F.T + q * _Q_BASE
    return new_state, new_cov


def _kf_update(state: np.ndarray, cov: np.ndarray, z_range: float, sigma_r2: float):
    R = np.array([[sigma_r2]])
    y = np.array([z_range]) - _H @ state
    S = _H @ cov @ _H.T + R
    K = cov @ _H.T @ np.linalg.inv(S)
    new_state = state + (K @ y).flatten()
    new_cov = (_I - K @ _H) @ cov
    assert np.all(np.isfinite(new_cov)), f"non-finite covariance: {new_cov}"
    return new_state, new_cov


class Tracker:
    """Frame-to-frame multi-target tracker.

    Per-frame: predict each track, gate, associate detections
    (greedy nearest-neighbor when method='nn', GNN when method='gnn'),
    update assigned, age unassigned, spawn from leftover detections.
    """

    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self._next_id = 0
        self.tracks: list = []

    # ---- Public API ----

    def step(self, detections: list) -> list:
        # 1) Predict.
        for t in self.tracks:
            t.state, t.cov = _kf_predict(t.state, t.cov, self.cfg.process_noise_q)
            t.age += 1

        # 2) Build cost matrix and associate.
        assignments = self._associate(detections)

        # 3) Update assigned tracks.
        assigned_tracks = set()
        assigned_dets = set()
        for ti, di in assignments:
            self._update_track(self.tracks[ti], detections[di])
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # 4) Age unassigned tracks.
        for ti, t in enumerate(self.tracks):
            if ti in assigned_tracks:
                continue
            t.misses += 1
            if t.hit_history is not None:
                t.hit_history.append(False)

        # 5) Spawn from unassigned detections.
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            self._spawn(det)

        # 6) Lifecycle: confirm / mark coasting / delete.
        self.tracks = [t for t in self.tracks if self._lifecycle_keep(t)]

        return list(self.tracks)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    # ---- Internal ----

    def _associate(self, detections: list) -> list:
        if not self.tracks or not detections:
            return []
        cost, feasible = self._build_cost_matrix(detections)
        if self.cfg.method == "nn":
            return self._associate_nn(cost, feasible)
        if self.cfg.method == "gnn":
            return self._associate_gnn(cost, feasible)
        raise ValueError(f"unknown association method: {self.cfg.method!r}")

    def _build_cost_matrix(self, detections: list):
        nT = len(self.tracks)
        nD = len(detections)
        cost = np.full((nT, nD), np.inf)
        feasible = np.zeros((nT, nD), dtype=bool)
        gate = self.cfg.gate_threshold
        sigma_d2 = self.cfg.doppler_meas_var
        sigma_r2 = self.cfg.range_meas_var
        for ti, t in enumerate(self.tracks):
            for di, det in enumerate(detections):
                dr = det.range_bin - t.state[0]
                dd = det.doppler_bin - t.last_doppler_bin
                d2 = dr * dr / (sigma_r2 + t.cov[0, 0]) + dd * dd / sigma_d2
                if d2 <= gate:
                    cost[ti, di] = d2
                    feasible[ti, di] = True
        return cost, feasible

    def _associate_nn(self, cost: np.ndarray, feasible: np.ndarray) -> list:
        out = []
        used_t = set()
        used_d = set()
        # Greedy: repeatedly pick the smallest feasible cost.
        flat = []
        nT, nD = cost.shape
        for ti in range(nT):
            for di in range(nD):
                if feasible[ti, di]:
                    flat.append((cost[ti, di], ti, di))
        flat.sort()
        for _, ti, di in flat:
            if ti in used_t or di in used_d:
                continue
            out.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
        return out

    def _associate_gnn(self, cost: np.ndarray, feasible: np.ndarray) -> list:
        # Replace inf with a large finite cost so linear_sum_assignment can solve;
        # we drop infeasible assignments after solving.
        big = float(self.cfg.gate_threshold) * 1e6 + 1.0
        c = np.where(feasible, cost, big)
        ti_idx, di_idx = linear_sum_assignment(c)
        out = []
        for ti, di in zip(ti_idx, di_idx):
            if feasible[ti, di]:
                out.append((int(ti), int(di)))
        return out

    def _update_track(self, t: Track, det: Detection) -> None:
        t.state, t.cov = _kf_update(t.state, t.cov, det.range_bin,
                                    self.cfg.range_meas_var)
        beta = self.cfg.doppler_smoothing_beta
        t.last_doppler_bin = (1.0 - beta) * t.last_doppler_bin + beta * det.doppler_bin
        t.hits += 1
        t.misses = 0
        if t.hit_history is not None:
            t.hit_history.append(True)
        if t.history is not None:
            t.history.append((det.range_bin, det.doppler_bin))

    def _spawn(self, det: Detection) -> None:
        cfg = self.cfg
        t = Track(
            id=self._next_id,
            state=np.array([det.range_bin, 0.0]),
            cov=np.diag([cfg.range_meas_var, cfg.init_velocity_var]),
            last_doppler_bin=det.doppler_bin,
            hits=1,
            misses=0,
            age=1,
            status="tentative",
            hit_history=deque([True], maxlen=cfg.confirm_N),
            history=deque([(det.range_bin, det.doppler_bin)], maxlen=cfg.history_len),
        )
        self._next_id += 1
        self.tracks.append(t)

    def _lifecycle_keep(self, t: Track) -> bool:
        cfg = self.cfg
        # Confirm tentative tracks via M-of-N rule.
        if t.status == "tentative":
            hits_in_window = sum(t.hit_history) if t.hit_history else 0
            if hits_in_window >= cfg.confirm_M:
                t.status = "confirmed"
                return True
            if t.misses > cfg.max_misses_tentative:
                return False
            return True
        # Confirmed → coasting → delete.
        if t.status == "confirmed":
            if t.misses == 0:
                return True
            t.status = "coasting"
            return True
        if t.status == "coasting":
            if t.misses == 0:
                t.status = "confirmed"
                return True
            if t.misses >= cfg.max_coast:
                return False
            return True
        return True
