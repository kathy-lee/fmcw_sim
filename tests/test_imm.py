"""Tests for processing.imm.IMMTracker."""

import numpy as np
import pytest


def _make_basic_imm_config():
    from processing.config import MotionModelSpec, IMMConfig
    return IMMConfig(
        models=[
            MotionModelSpec(motion_model="cv"),
            MotionModelSpec(motion_model="drag",
                            motion_params={"drag_coefficient_k": 0.05}),
        ],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        initial_mode_probs=[0.5, 0.5],
    )


class _Det:
    """Minimal Detection stand-in for tests."""
    def __init__(self, range_bin, doppler_bin, peak_db=30.0):
        self.range_bin = range_bin
        self.doppler_bin = doppler_bin
        self.peak_db = peak_db


def test_imm_tracker_step_returns_empty_when_no_tracks_and_no_detections():
    from processing.imm import IMMTracker
    tracker = IMMTracker(_make_basic_imm_config())
    out = tracker.step(detections=[])
    assert out == []


def test_imm_tracker_spawns_track_from_detection():
    from processing.imm import IMMTracker
    tracker = IMMTracker(_make_basic_imm_config())
    tracker.step([_Det(20.0, 32.0)])
    assert len(tracker.tracks) == 1


def test_imm_mode_probabilities_sum_to_one_each_frame():
    """After each step, every track's mode probabilities should sum to 1."""
    from processing.imm import IMMTracker
    tracker = IMMTracker(_make_basic_imm_config())
    for k in range(6):
        tracker.step([_Det(20.0 + k * 0.2, 32.0)])
    for t in tracker.tracks:
        assert abs(sum(t.mode_probs) - 1.0) < 1e-6


def test_imm_combine_step_outputs_weighted_mean():
    """The IMM-combined state is the probability-weighted mean of mode states."""
    from processing.imm import _combine
    states = [np.array([10.0, 1.0]), np.array([20.0, 2.0])]
    covs = [np.eye(2), np.eye(2)]
    mu = np.array([0.7, 0.3])
    combined_state, _ = _combine(states, covs, mu)
    expected_state = 0.7 * states[0] + 0.3 * states[1]
    np.testing.assert_allclose(combined_state, expected_state)


def test_imm_mix_step_conditional_probs_sum_to_one_per_column():
    """In the mix step, the conditional probability mu_ij of having been in
    mode i given now in mode j should sum to 1 down each column j."""
    from processing.imm import _mix
    states = [np.array([10.0, 1.0]), np.array([20.0, 2.0])]
    covs = [np.eye(2), np.eye(2)]
    mu = np.array([0.5, 0.5])
    Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
    _, _, mu_ij = _mix(states, covs, mu, Pi)
    for j in range(2):
        col_sum = sum(mu_ij[i][j] for i in range(2))
        assert abs(col_sum - 1.0) < 1e-6
