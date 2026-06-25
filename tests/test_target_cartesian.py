"""Cartesian-primary Target with derived radial properties."""
import math

from simulator.beat_signal_generator import Target


def test_on_axis_reduces_to_radial_exactly():
    t = Target(x=4.1, y=0.0, vx=0.5, vy=0.0)
    assert t.range_m == 4.1            # exact: no sqrt round-trip
    assert t.velocity_mps == 0.5
    assert t.azimuth_rad == 0.0


def test_on_axis_negative_velocity_keeps_sign():
    # Inbound target (vx < 0) must report negative radial velocity, not +|vx|.
    t = Target(x=6.0, y=0.0, vx=-1.0, vy=0.0)
    assert t.velocity_mps == -1.0


def test_off_axis_polar_derivation():
    t = Target(x=3.0, y=4.0, vx=0.0, vy=0.0)
    assert t.range_m == 5.0
    assert math.isclose(t.azimuth_rad, math.atan2(4.0, 3.0))


def test_off_axis_radial_velocity():
    t = Target(x=3.0, y=4.0, vx=1.0, vy=1.0)
    assert math.isclose(t.velocity_mps, (3 * 1.0 + 4 * 1.0) / 5.0)


def test_defaults_are_stationary_on_axis():
    t = Target(x=7.0)
    assert (t.y, t.vx, t.vy) == (0.0, 0.0, 0.0)
    assert t.range_m == 7.0 and t.velocity_mps == 0.0


import json
import os

from simulator.beat_signal_generator import load_scenario


def _write(tmp, obj):
    p = os.path.join(tmp, "s.json")
    with open(p, "w") as f:
        f.write(json.dumps(obj))
    return p


def test_legacy_radial_scenario_loads_on_axis(tmp_path):
    p = _write(str(tmp_path), {"radar": {}, "targets": [
        {"range_m": 4.0, "velocity_mps": 0.5, "amplitude": 0.9}]})
    _, targets = load_scenario(p)
    t = targets[0]
    assert (t.x, t.y, t.vx, t.vy) == (4.0, 0.0, 0.5, 0.0)
    assert t.amplitude == 0.9


def test_native_cartesian_scenario_loads(tmp_path):
    p = _write(str(tmp_path), {"radar": {}, "targets": [
        {"x": 3.0, "y": 4.0, "vx": 1.0, "vy": 0.0}]})
    _, targets = load_scenario(p)
    assert (targets[0].x, targets[0].y) == (3.0, 4.0)


def test_legacy_scenario_with_motion_model_preserved(tmp_path):
    p = _write(str(tmp_path), {"radar": {}, "targets": [
        {"range_m": 10.0, "velocity_mps": 5.0, "motion_model": "drag",
         "motion_params": {"drag_coefficient_k": 0.3}}]})
    _, targets = load_scenario(p)
    t = targets[0]
    assert t.x == 10.0 and t.vx == 5.0
    assert t.motion_model == "drag"
    assert t.motion_params == {"drag_coefficient_k": 0.3}
