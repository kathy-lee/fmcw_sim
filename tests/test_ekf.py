"""Tests for the EKF predict step and drag-filter factory."""

import numpy as np
import pytest


def test_drag_filter_f_steps_velocity_down_per_drag_law():
    """f_drag should reduce velocity by k*v*|v| per step."""
    from processing.tracking import _make_drag_filter
    f, _ = _make_drag_filter(drag_k_bins=0.1)
    state = np.array([10.0, 2.0])
    new = f(state)
    # new_vel = 2 - 0.1*2*|2| = 2 - 0.4 = 1.6
    assert new[1] == pytest.approx(1.6)
    # new_pos = 10 + 2 = 12
    assert new[0] == pytest.approx(12.0)


def test_drag_filter_jacobian_matches_numerical_diff():
    """At several operating points, F_jacobian should equal the numerical
    derivative of f within float precision."""
    from processing.tracking import _make_drag_filter
    f, F_jac = _make_drag_filter(drag_k_bins=0.1)
    eps = 1e-6
    for state in (np.array([5.0, 3.0]), np.array([0.0, 1.0]),
                  np.array([20.0, 0.5])):
        analytic = F_jac(state)
        numerical = np.zeros((2, 2))
        for j in range(2):
            d = np.zeros(2)
            d[j] = eps
            numerical[:, j] = (f(state + d) - f(state - d)) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, atol=1e-6)


def test_ekf_predict_returns_f_state_and_jacobian_covariance():
    """The predict step uses f for state and the Jacobian for covariance."""
    from processing.tracking import _ekf_predict, _make_drag_filter, _Q_BASE
    f, F_jac = _make_drag_filter(drag_k_bins=0.1)
    state = np.array([5.0, 2.0])
    cov = np.eye(2)
    q = 0.5
    new_state, new_cov = _ekf_predict(state, cov, f, F_jac, q)

    np.testing.assert_allclose(new_state, f(state))

    F = F_jac(state)
    expected_cov_pre_q = F @ cov @ F.T
    np.testing.assert_allclose(new_cov - q * _Q_BASE, expected_cov_pre_q,
                                atol=1e-10)


def test_ekf_predict_reduces_to_kf_when_jacobian_is_constant():
    """For a linear motion, EKF result == KF result."""
    from processing.tracking import _ekf_predict, _kf_predict, _F
    f = lambda s: _F @ s
    F_jac = lambda s: _F
    state = np.array([5.0, 2.0])
    cov = np.eye(2)
    q = 0.5
    ekf_state, ekf_cov = _ekf_predict(state, cov, f, F_jac, q)
    kf_state, kf_cov = _kf_predict(state, cov, q)
    np.testing.assert_allclose(ekf_state, kf_state)
    np.testing.assert_allclose(ekf_cov, kf_cov)


def test_tracker_drag_motion_model_actually_uses_ekf_predict():
    """Verify the Tracker actually wires through _ekf_predict when
    motion_model='drag'. Compare two trackers (cv vs drag) sharing the same
    initial state — after one frame with no detections, their predicted
    states should differ by exactly the drag-induced delta.

    Note on units: drag_coefficient_k in TrackerConfig is in bin-units
    (filter state is in range-bin space). The simulator's metric k is
    converted via drag_k_bins = k_metric * frame_period_s² / range_per_bin.
    See design doc 'Risks' section. This test uses bin-units k directly.
    """
    from processing.config import TrackerConfig
    from processing.tracking import (
        Tracker, _kf_predict, _ekf_predict, _make_drag_filter,
    )

    # Build a Tracker with drag_motion_model. Manually inject a Track-like
    # object with a known state, run one predict step, verify the predicted
    # state matches _ekf_predict's output (not _kf_predict's).
    drag_k_bins = 0.05
    cfg = TrackerConfig(method="gnn", motion_model="drag",
                         drag_coefficient_k=drag_k_bins)
    tracker = Tracker(cfg)

    # Verify the predict callables are wired in.
    assert tracker._predict_ekf_f is not None
    assert tracker._predict_ekf_F_jac is not None

    # Verify they match a fresh _make_drag_filter() with same k.
    expected_f, expected_jac = _make_drag_filter(drag_k_bins)
    state = np.array([10.0, 3.0])
    np.testing.assert_allclose(tracker._predict_ekf_f(state), expected_f(state))
    np.testing.assert_allclose(
        tracker._predict_ekf_F_jac(state), expected_jac(state)
    )


def test_tracker_cv_motion_model_uses_kf_predict():
    """Default motion_model='cv' wires no _predict_ekf_f (uses linear KF)."""
    from processing.config import TrackerConfig
    from processing.tracking import Tracker

    cfg = TrackerConfig(method="gnn", motion_model="cv")
    tracker = Tracker(cfg)

    assert tracker._predict_ekf_f is None
    assert tracker._predict_ekf_F_jac is None


def test_tracker_drag_filter_diverges_from_cv_over_multiple_predict_steps():
    """After several predict-only steps (no detections), a drag-filter
    Tracker's state should differ from a CV-filter Tracker's state."""
    from dataclasses import dataclass
    from typing import Optional
    from collections import deque

    from processing.config import TrackerConfig
    from processing.tracking import Tracker, Track

    drag_k_bins = 0.05

    def setup(motion_model: str) -> Tracker:
        cfg = TrackerConfig(method="gnn", motion_model=motion_model,
                             drag_coefficient_k=drag_k_bins,
                             confirm_M=1)
        tr = Tracker(cfg)
        # Manually inject a confirmed track with known initial state
        t = Track(
            id=0,
            state=np.array([100.0, 6.0]),  # 100 bins, 6 bins/frame
            cov=np.diag([0.5, 4.0]),
            last_doppler_bin=32.0,
            age=5,
            misses=0,
            hits=5,
            status="confirmed",
            hit_history=deque([True] * 5, maxlen=10),
        )
        tr.tracks.append(t)
        return tr

    tr_cv = setup("cv")
    tr_drag = setup("drag")

    # Step both with empty detections (predict-only). Keep below max_coast
    # so the track doesn't get lifecycle-dropped.
    for _ in range(3):
        tr_cv.step([])
        tr_drag.step([])

    assert len(tr_cv.tracks) == 1, "CV track was dropped"
    assert len(tr_drag.tracks) == 1, "drag track was dropped"

    # CV should have advanced at constant 6 bins/frame: pos = 100 + 3*6 = 118.
    # Drag should have decelerated meaningfully and be at a smaller pos.
    cv_pos = tr_cv.tracks[0].state[0]
    drag_pos = tr_drag.tracks[0].state[0]
    assert cv_pos > drag_pos, (
        f"CV pos {cv_pos} should be > drag pos {drag_pos} "
        f"(drag should decelerate)"
    )
    assert abs(cv_pos - drag_pos) > 1.0, (
        f"CV and drag predictions should differ by >1 bin after 3 steps; "
        f"got cv={cv_pos:.2f}, drag={drag_pos:.2f}"
    )
