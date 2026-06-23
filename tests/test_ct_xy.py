import math
import numpy as np

from simulator.beat_signal_generator import Target
from simulator.motion_models import (
    ct_xy_range, ct_xy_azimuth, cv_xy_range, range_at, azimuth_at)


def _t(turn_rate, vx=0.0, vy=2.0, x=10.0, y=0.0):
    return Target(x=x, y=y, vx=vx, vy=vy, amplitude=1.0,
                  motion_model="ct_xy", motion_params={"turn_rate": turn_rate})


def _to_cv(tgt):
    return Target(x=tgt.x, y=tgt.y, vx=tgt.vx, vy=tgt.vy, amplitude=1.0,
                  motion_model="cv_xy")


def test_ct_xy_zero_turn_matches_cv_xy():
    tgt = _t(0.0)
    cv = Target(x=10.0, y=0.0, vx=0.0, vy=2.0, amplitude=1.0, motion_model="cv_xy")
    for fi in range(5):
        r_ct, v_ct = ct_xy_range(tgt, fi, 0.05)
        r_cv, v_cv = cv_xy_range(cv, fi, 0.05)
        assert r_ct == r_cv and v_ct == v_cv          # byte-identical to cv_xy


def test_ct_xy_preserves_speed():
    # speed = |(vx,vy)| must be conserved under a coordinated turn
    tgt = _t(0.5, vx=2.0, vy=0.0)
    r0, _ = ct_xy_range(tgt, 0, 0.05)
    assert r0 == math.hypot(10.0, 0.0)                # starts at (10,0)


def test_ct_xy_curves_off_the_straight_line():
    straight = cv_xy_range(_to_cv(_t(0.0, vx=0.0, vy=2.0)), 10, 0.05)[0]
    turned = ct_xy_range(_t(0.6, vx=0.0, vy=2.0), 10, 0.05)[0]
    assert abs(turned - straight) > 1e-3              # turn changes the range


def test_ct_xy_registered_in_dispatch():
    tgt = _t(0.3)
    assert range_at(tgt, 3, 0.05) == ct_xy_range(tgt, 3, 0.05)
    assert azimuth_at(tgt, 3, 0.05) == ct_xy_azimuth(tgt, 3, 0.05)
