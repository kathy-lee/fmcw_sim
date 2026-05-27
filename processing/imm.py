"""IMM (Interacting Multiple Model) meta-tracker.

Holds N constituent Kalman filters (linear CV or nonlinear drag-EKF, both
with 2D state). Each per-frame cycle:

1. Mix         — compute mode-conditional initial states from prior mode
                 estimates using the Markov transition matrix Pi.
2. Predict     — each constituent predicts its mixed initial state forward
                 using its own motion model (CV uses _kf_predict; drag uses
                 _ekf_predict).
3. Associate   — gate + Hungarian assignment of detections to tracks, using
                 the combined (probability-weighted) predicted state.
4. Update      — each constituent independently updates given the
                 associated detection and produces an innovation likelihood
                 used to refresh the mode probabilities.
5. Combine     — produce the track's exposed state as the probability-
                 weighted mean of constituent states.

Exposes the same step() interface as processing.tracking.Tracker so the
C stress-test harness can evaluate IMMTracker without modification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.optimize

from processing.config import IMMConfig
from processing.tracking import (
    _H, _Q_BASE,
    _ekf_predict, _kf_predict, _kf_update, _make_drag_filter,
)


@dataclass
class IMMTrack:
    """Track held by IMMTracker. Stores one (state, cov) per constituent
    mode plus the mode-probability vector. The `state` and `cov` properties
    expose the IMM-combined estimate (probability-weighted mean and spread)
    so this object satisfies the same field interface a normal Track does:
    id, state[0], last_doppler_bin, status, cov[0,0]."""
    id: int
    mode_states: list                     # list[np.ndarray] per mode
    mode_covs: list                       # list[np.ndarray] per mode
    mode_probs: np.ndarray                # length-N probability vector
    last_doppler_bin: float
    age: int = 0
    misses: int = 0
    hits: int = 0
    status: str = "tentative"
    hit_history: Optional[deque] = None
    assoc_det_idx: Optional[int] = None

    @property
    def state(self) -> np.ndarray:
        """Combined (probability-weighted) state over modes."""
        return sum(p * s for p, s in zip(self.mode_probs, self.mode_states))

    @property
    def cov(self) -> np.ndarray:
        """Combined covariance, including spread-of-means term."""
        x = self.state
        return sum(
            p * (c + np.outer(s - x, s - x))
            for p, s, c in zip(self.mode_probs, self.mode_states, self.mode_covs)
        )


def _mix(states, covs, mu, Pi):
    """IMM mix step.

    Returns (mixed_states, mixed_covs, mu_ij) where mu_ij[i][j] is the
    conditional probability of having been in mode i given now being in
    mode j. The mixed state/cov for mode j is the moment-matched mixture
    of all modes weighted by mu_ij[:,j].
    """
    n = len(states)
    c = [sum(Pi[i][j] * mu[i] for i in range(n)) for j in range(n)]
    mu_ij = [
        [Pi[i][j] * mu[i] / c[j] if c[j] > 1e-12 else 0.0 for j in range(n)]
        for i in range(n)
    ]

    mixed_states, mixed_covs = [], []
    for j in range(n):
        x_mix = sum(mu_ij[i][j] * states[i] for i in range(n))
        p_mix = sum(
            mu_ij[i][j] * (covs[i] + np.outer(states[i] - x_mix,
                                                states[i] - x_mix))
            for i in range(n)
        )
        mixed_states.append(x_mix)
        mixed_covs.append(p_mix)
    return mixed_states, mixed_covs, mu_ij


def _combine(states, covs, mu):
    """IMM combine step: probability-weighted mean and spread-aware
    covariance over modes. Returns (x, P)."""
    x = sum(m * s for m, s in zip(mu, states))
    p = sum(
        m * (c + np.outer(s - x, s - x))
        for m, s, c in zip(mu, states, covs)
    )
    return x, p


def _innovation_likelihood(z, x_pred, P_pred, sigma_r2):
    """Gaussian likelihood Λ = N(z; H·x_pred, S) of a scalar range measurement
    under the predicted state. Used to update mode probabilities."""
    z_pred = _H @ x_pred
    S = _H @ P_pred @ _H.T + np.array([[sigma_r2]])
    innov = float(z - z_pred[0])
    det_S = float(S[0, 0])
    return (1.0 / np.sqrt(2 * np.pi * det_S)) * np.exp(
        -0.5 * innov * innov / det_S
    )


class IMMTracker:
    """Multi-model tracker with IMM blending. See module docstring."""

    def __init__(self, config: IMMConfig):
        self.cfg = config
        self.Pi = np.array(config.transition_matrix)
        self.mu0 = np.array(config.initial_mode_probs)

        # Build per-mode predict callables. None means "use linear KF".
        # For nonlinear motion (drag), store the (f, F_jac) pair.
        self._mode_predictors = []
        for spec in config.models:
            if spec.motion_model == "cv":
                self._mode_predictors.append(None)
            elif spec.motion_model == "drag":
                k = spec.motion_params.get("drag_coefficient_k", 0.5)
                self._mode_predictors.append(_make_drag_filter(k))
            else:
                raise ValueError(
                    f"unknown motion_model: {spec.motion_model!r}"
                )

        self._next_id = 0
        self.tracks: list[IMMTrack] = []

    # ---- Public API ----

    def step(self, detections: list) -> list:
        # 1) Predict (with mix step) for each existing track
        for t in self.tracks:
            self._predict_track(t)
            t.age += 1
            t.assoc_det_idx = None

        # 2) Gate + Hungarian assignment using combined predicted state
        assignments = self._associate(detections)

        # 3) Update assigned tracks
        assigned_tracks = set()
        assigned_dets = set()
        for ti, di in assignments:
            self._update_track(self.tracks[ti], detections[di])
            self.tracks[ti].assoc_det_idx = di
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # 4) Age unassigned tracks
        for ti, t in enumerate(self.tracks):
            if ti in assigned_tracks:
                continue
            t.misses += 1
            if t.hit_history is not None:
                t.hit_history.append(False)

        # 5) Spawn new tracks from leftover detections
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            self._spawn(det)

        # 6) Lifecycle: confirm / coast / drop
        self.tracks = [t for t in self.tracks if self._lifecycle_keep(t)]

        return list(self.tracks)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    # ---- Internals ----

    def _predict_track(self, t: IMMTrack) -> None:
        # Mix step
        mixed_states, mixed_covs, _ = _mix(
            t.mode_states, t.mode_covs, t.mode_probs, self.Pi
        )
        # Per-mode predict
        new_states, new_covs = [], []
        for i, predictor in enumerate(self._mode_predictors):
            if predictor is None:
                ns, nc = _kf_predict(
                    mixed_states[i], mixed_covs[i], self.cfg.process_noise_q
                )
            else:
                f, F_jac = predictor
                ns, nc = _ekf_predict(
                    mixed_states[i], mixed_covs[i], f, F_jac,
                    self.cfg.process_noise_q,
                )
            new_states.append(ns)
            new_covs.append(nc)
        t.mode_states = new_states
        t.mode_covs = new_covs

    def _associate(self, detections):
        if not self.tracks or not detections:
            return []
        gate = self.cfg.gate_threshold
        sigma_d2 = self.cfg.doppler_meas_var
        sigma_r2 = self.cfg.range_meas_var
        big = float(gate) * 1e6 + 1.0
        cost = np.full((len(self.tracks), len(detections)), big)
        for ti, t in enumerate(self.tracks):
            x_combined = t.state
            cov_pos = float(t.cov[0, 0])
            for di, det in enumerate(detections):
                dr = det.range_bin - x_combined[0]
                dd = det.doppler_bin - t.last_doppler_bin
                d2 = dr * dr / (sigma_r2 + cov_pos) + dd * dd / sigma_d2
                if d2 <= gate:
                    cost[ti][di] = d2
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        out = []
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] <= gate:
                out.append((int(r), int(c)))
        return out

    def _update_track(self, t: IMMTrack, det) -> None:
        z_range = det.range_bin
        sigma_r2 = self.cfg.range_meas_var

        new_states, new_covs, likelihoods = [], [], []
        for i in range(len(t.mode_states)):
            x_pred = t.mode_states[i]
            P_pred = t.mode_covs[i]
            lh = _innovation_likelihood(z_range, x_pred, P_pred, sigma_r2)
            ns, nc = _kf_update(x_pred, P_pred, z_range, sigma_r2)
            new_states.append(ns)
            new_covs.append(nc)
            likelihoods.append(lh)

        # Mode probability update: mu_new_j ∝ Λ_j · Σ_i Pi[i,j] · mu_i
        n = len(t.mode_probs)
        c_pred = np.array([
            sum(self.Pi[i][j] * t.mode_probs[i] for i in range(n))
            for j in range(n)
        ])
        mu_new = np.array([likelihoods[j] * c_pred[j] for j in range(n)])
        total = mu_new.sum()
        if total > 1e-12:
            mu_new = mu_new / total
        else:
            # Numerical degeneracy: revert to initial mode probabilities.
            mu_new = np.array(self.cfg.initial_mode_probs, dtype=float)

        t.mode_states = new_states
        t.mode_covs = new_covs
        t.mode_probs = mu_new
        t.misses = 0
        t.hits += 1
        if t.hit_history is not None:
            t.hit_history.append(True)
        beta = self.cfg.doppler_smoothing_beta
        t.last_doppler_bin = (
            beta * det.doppler_bin + (1.0 - beta) * t.last_doppler_bin
        )

    def _spawn(self, det) -> None:
        n = len(self.cfg.models)
        x0 = np.array([float(det.range_bin), 0.0])
        cov0 = np.diag([self.cfg.range_meas_var, self.cfg.init_velocity_var])
        t = IMMTrack(
            id=self._next_id,
            mode_states=[x0.copy() for _ in range(n)],
            mode_covs=[cov0.copy() for _ in range(n)],
            mode_probs=self.mu0.copy(),
            last_doppler_bin=float(det.doppler_bin),
            hits=1,
            hit_history=deque([True], maxlen=self.cfg.history_len),
        )
        self._next_id += 1
        self.tracks.append(t)

    def _lifecycle_keep(self, t: IMMTrack) -> bool:
        cfg = self.cfg
        if t.status == "tentative":
            hits_in_window = (
                sum(t.hit_history) if t.hit_history is not None else t.hits
            )
            if hits_in_window >= cfg.confirm_M:
                t.status = "confirmed"
                return True
            if t.misses > cfg.max_misses_tentative:
                return False
            return True
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
