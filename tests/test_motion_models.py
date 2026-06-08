"""Tests for simulator.motion_models — analytic ground-truth trajectories."""

from __future__ import annotations

import math
import pytest


class _FakeTarget:
    """Light stand-in for the Target dataclass so motion_models tests don't
    depend on importing the simulator (which fans out a lot of code)."""

    def __init__(self, range_m, velocity_mps, motion_model, motion_params=None):
        self.range_m = range_m
        self.velocity_mps = velocity_mps
        self.motion_model = motion_model
        self.motion_params = motion_params or {}


def test_constant_velocity_range_at_zero_frame_matches_initial():
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=4.0, velocity_mps=1.0,
                    motion_model="constant_velocity")
    r, v = range_at(t, frame_idx=0, frame_period_s=0.05)
    assert r == pytest.approx(4.0)
    assert v == pytest.approx(1.0)


def test_constant_velocity_range_at_later_frame_matches_linear_formula():
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=4.0, velocity_mps=1.0,
                    motion_model="constant_velocity")
    # 20 frames at 0.05s each = 1s elapsed; r = 4 + 1*1 = 5
    r, v = range_at(t, frame_idx=20, frame_period_s=0.05)
    assert r == pytest.approx(5.0)
    assert v == pytest.approx(1.0)


def test_drag_range_at_zero_frame_matches_initial():
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=1.0, velocity_mps=5.0,
                    motion_model="drag",
                    motion_params={"drag_coefficient_k": 0.5})
    r, v = range_at(t, frame_idx=0, frame_period_s=0.05)
    assert r == pytest.approx(1.0)
    assert v == pytest.approx(5.0)


def test_drag_range_at_later_frame_matches_analytic_solution():
    """Quadratic drag: v(t) = v0/(1+k*v0*t), r(t) = r0 + log(1+k*v0*t)/k."""
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=1.0, velocity_mps=5.0,
                    motion_model="drag",
                    motion_params={"drag_coefficient_k": 0.5})
    # frame_idx=80 at 0.05s = 4s elapsed
    elapsed = 80 * 0.05
    denom = 1.0 + 0.5 * 5.0 * elapsed
    expected_v = 5.0 / denom
    expected_r = 1.0 + math.log(denom) / 0.5
    r, v = range_at(t, frame_idx=80, frame_period_s=0.05)
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)


def test_drag_with_zero_velocity_stays_at_initial_range():
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=3.0, velocity_mps=0.0,
                    motion_model="drag",
                    motion_params={"drag_coefficient_k": 0.5})
    r, v = range_at(t, frame_idx=50, frame_period_s=0.05)
    assert r == pytest.approx(3.0)
    assert v == pytest.approx(0.0)


def test_range_at_rejects_unknown_motion_model():
    from simulator.motion_models import range_at
    t = _FakeTarget(range_m=1.0, velocity_mps=1.0,
                    motion_model="nope")
    with pytest.raises(ValueError, match="motion_model"):
        range_at(t, frame_idx=0, frame_period_s=0.05)


def test_piecewise_motion_uses_first_phase_for_early_frames():
    from simulator.motion_models import range_at
    t = _FakeTarget(
        range_m=1.0, velocity_mps=2.0,
        motion_model="piecewise",
        motion_params={
            "phases": [
                {"motion_model": "constant_velocity",
                 "start_frame": 0, "end_frame": 40},
                {"motion_model": "drag",
                 "start_frame": 40, "end_frame": 80,
                 "motion_params": {"drag_coefficient_k": 0.5}},
            ]
        },
    )
    # Frame 10 is in phase 1 (CV): r = 1.0 + 2.0 * 10*0.05 = 2.0
    r, v = range_at(t, frame_idx=10, frame_period_s=0.05)
    assert r == pytest.approx(2.0)
    assert v == pytest.approx(2.0)


def test_piecewise_motion_uses_second_phase_for_later_frames():
    from simulator.motion_models import range_at
    t = _FakeTarget(
        range_m=1.0, velocity_mps=2.0,
        motion_model="piecewise",
        motion_params={
            "phases": [
                {"motion_model": "constant_velocity",
                 "start_frame": 0, "end_frame": 40},
                {"motion_model": "drag",
                 "start_frame": 40, "end_frame": 80,
                 "motion_params": {"drag_coefficient_k": 0.5}},
            ]
        },
    )
    # Frame 60 is in phase 2 (drag). Phase boundary at frame 40 (t=2s),
    # where position is 1.0 + 2.0*2.0 = 5.0 and velocity is 2.0. Drag then
    # decelerates from there for 20 more frames (1s elapsed).
    import math
    r0_drag, v0_drag = 5.0, 2.0
    elapsed = 1.0  # 20 frames * 0.05
    k = 0.5
    denom = 1.0 + k * v0_drag * elapsed
    expected_v = v0_drag / denom
    expected_r = r0_drag + math.log(denom) / k
    r, v = range_at(t, frame_idx=60, frame_period_s=0.05)
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)
