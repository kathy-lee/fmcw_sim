"""Per-target motion models (ground-truth trajectory functions).

The simulator dispatches through `range_at()` based on each Target's
motion_model attribute. Adding a new motion model means writing a new
function with the same signature and registering it in _MOTION_DISPATCH.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.beat_signal_generator import Target


def constant_velocity_range(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Position and velocity at frame_idx for a constant-velocity target."""
    t = frame_idx * frame_period_s
    return target.range_m + target.velocity_mps * t, target.velocity_mps


def drag_range(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Position and velocity under quadratic drag dv/dt = -k*v*|v|.

    Closed-form solution for v0 > 0:
        v(t) = v0 / (1 + k*v0*t)
        r(t) = r0 + log(1 + k*v0*t) / k

    For v0 == 0 the target is stationary (no drag force without motion).
    Negative initial velocities are not supported in this sub-project; the
    test scenarios use only nonnegative velocities.
    """
    t = frame_idx * frame_period_s
    v0 = target.velocity_mps
    r0 = target.range_m
    k = target.motion_params.get("drag_coefficient_k", 0.5)
    if v0 == 0.0:
        return r0, 0.0
    denom = 1.0 + k * v0 * t
    v_t = v0 / denom
    r_t = r0 + math.log(denom) / k
    return r_t, v_t


def piecewise_range(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Position and velocity for a target whose motion model changes by phase.

    motion_params['phases'] is a list of phase dicts:
        [{"motion_model": <kind>, "start_frame": <int>, "end_frame": <int>,
          "motion_params": {<kind-specific>}}]

    Within each phase the target evolves under that phase's motion model
    starting from the position and velocity at the phase boundary (which is
    the prior phase's terminal state). Phases are assumed consecutive (no
    gaps; phase n's end_frame equals phase n+1's start_frame).
    """
    phases = target.motion_params["phases"]
    active = None
    for phase in phases:
        if phase["start_frame"] <= frame_idx < phase["end_frame"]:
            active = phase
            break
    if active is None:
        # frame_idx is past the last phase's end — use the last phase
        # extrapolated, which we treat as "stay at the last phase's terminal
        # state". This is a defensive default; scenarios should not query
        # past their declared end_frame.
        active = phases[-1]

    class _PhaseTarget:
        pass

    # Walk completed phases, propagating (r, v) through each one's full span.
    phase_r, phase_v = target.range_m, target.velocity_mps
    for phase in phases:
        if phase is active:
            break
        local_frames = phase["end_frame"] - phase["start_frame"]
        pt = _PhaseTarget()
        pt.range_m = phase_r
        pt.velocity_mps = phase_v
        pt.motion_model = phase["motion_model"]
        pt.motion_params = phase.get("motion_params", {})
        phase_r, phase_v = range_at(pt, local_frames, frame_period_s)

    # Within the active phase, advance by frame_idx - active.start_frame.
    frame_offset = frame_idx - active["start_frame"]
    pt = _PhaseTarget()
    pt.range_m = phase_r
    pt.velocity_mps = phase_v
    pt.motion_model = active["motion_model"]
    pt.motion_params = active.get("motion_params", {})
    return range_at(pt, frame_offset, frame_period_s)


def cv_xy_range(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Range and radial velocity for a target moving at constant velocity in the
    (x, y) plane. range = |position(t)|; radial velocity is the projection of
    (vx, vy) onto the line of sight at time t. Native Cartesian (2-D) motion —
    use this for off-boresight / cross-range targets (MIMO)."""
    t = frame_idx * frame_period_s
    x_t = target.x + target.vx * t
    y_t = target.y + target.vy * t
    r = math.hypot(x_t, y_t)
    if r == 0.0:
        return 0.0, 0.0
    v_rad = (x_t * target.vx + y_t * target.vy) / r
    return r, v_rad


def cv_xy_azimuth(
    target: "Target", frame_idx: int, frame_period_s: float
) -> float:
    """Azimuth (rad) of a cv_xy target at frame_idx."""
    t = frame_idx * frame_period_s
    return math.atan2(target.y + target.vy * t, target.x + target.vx * t)


def _ct_xy_state(target: "Target", t: float):
    """(x, y, vx, vy) of a coordinated-turn target at time t.
    turn_rate ω (rad/s) rotates the velocity vector; ω→0 reduces to cv_xy."""
    w = target.motion_params.get("turn_rate", 0.0)
    if abs(w) < 1e-9:
        return (target.x + target.vx * t, target.y + target.vy * t,
                target.vx, target.vy)
    s = math.hypot(target.vx, target.vy)
    phi0 = math.atan2(target.vy, target.vx)
    x_t = target.x + (s / w) * (math.sin(phi0 + w * t) - math.sin(phi0))
    y_t = target.y + (s / w) * (math.cos(phi0) - math.cos(phi0 + w * t))
    return x_t, y_t, s * math.cos(phi0 + w * t), s * math.sin(phi0 + w * t)


def ct_xy_range(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Range and radial velocity for a coordinated-turn (constant turn rate +
    speed) planar target. range = |position(t)|; radial velocity is (vx,vy)
    projected onto the line of sight at time t."""
    t = frame_idx * frame_period_s
    x_t, y_t, vx_t, vy_t = _ct_xy_state(target, t)
    r = math.hypot(x_t, y_t)
    if r == 0.0:
        return 0.0, 0.0
    return r, (x_t * vx_t + y_t * vy_t) / r


def ct_xy_azimuth(
    target: "Target", frame_idx: int, frame_period_s: float
) -> float:
    """Azimuth (rad) of a ct_xy target at frame_idx."""
    x_t, y_t, _, _ = _ct_xy_state(target, frame_idx * frame_period_s)
    return math.atan2(y_t, x_t)


_MOTION_DISPATCH = {
    "constant_velocity": constant_velocity_range,
    "drag": drag_range,
    "piecewise": piecewise_range,
    "cv_xy": cv_xy_range,
    "ct_xy": ct_xy_range,
}


def azimuth_at(
    target: "Target", frame_idx: int, frame_period_s: float
) -> float:
    """Frame-start azimuth (rad). cv_xy/ct_xy evolve with the planar motion; the
    radial models keep the target's (static) initial azimuth — 0 for on-axis
    legacy targets, so single-channel generation is unaffected."""
    if target.motion_model == "cv_xy":
        return cv_xy_azimuth(target, frame_idx, frame_period_s)
    if target.motion_model == "ct_xy":
        return ct_xy_azimuth(target, frame_idx, frame_period_s)
    return target.azimuth_rad


def range_at(
    target: "Target", frame_idx: int, frame_period_s: float
) -> tuple[float, float]:
    """Dispatch to the registered motion function for target.motion_model."""
    fn = _MOTION_DISPATCH.get(target.motion_model)
    if fn is None:
        raise ValueError(
            f"unknown motion_model: {target.motion_model!r} "
            f"(known: {sorted(_MOTION_DISPATCH)})"
        )
    return fn(target, frame_idx, frame_period_s)
