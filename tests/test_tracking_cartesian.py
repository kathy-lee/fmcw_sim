import numpy as np
import pytest

from processing.config import CartesianTrackerConfig


def test_config_defaults_valid():
    c = CartesianTrackerConfig()
    assert c.method == "gnn"
    assert c.gate_threshold == pytest.approx(11.34)
    assert c.sigma_a > 0 and c.sigma_r > 0
    # ω-frozen baseline must be expressible (these may be 0):
    CartesianTrackerConfig(sigma_yaw_accel=0.0, init_omega_var=0.0)


def test_config_rejects_bad_method():
    with pytest.raises(ValueError):
        CartesianTrackerConfig(method="banana")


def test_config_rejects_nonpositive_meas_noise():
    with pytest.raises(ValueError):
        CartesianTrackerConfig(sigma_r=0.0)


def test_config_rejects_confirm_M_gt_N():
    with pytest.raises(ValueError):
        CartesianTrackerConfig(confirm_M=6, confirm_N=5)


from processing.tracking_cartesian import ctrv_f, ctrv_F, wrap_to_pi


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(3 * np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(-3 * np.pi) == pytest.approx(np.pi)  # -3π wraps to +π


def test_ctrv_straight_line_when_omega_zero():
    # [x,y,v,ψ,ω] heading +x, speed 2, no turn -> advance 2 m in x
    out = ctrv_f(np.array([0.0, 0.0, 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [2.0, 0.0, 2.0, 0.0, 0.0], atol=1e-9)


def test_ctrv_turn_preserves_speed_and_advances_heading():
    out = ctrv_f(np.array([0.0, 0.0, 2.0, 0.0, 0.1]))
    assert out[2] == pytest.approx(2.0)          # speed unchanged
    assert out[3] == pytest.approx(0.1)          # heading += ω
    assert out[4] == pytest.approx(0.1)          # ω unchanged
    assert out[1] > 0.0                          # left turn curves +y


def _numeric_jac(fn, x, eps=1e-6):
    n = len(x)
    J = np.zeros((len(fn(x)), n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        J[:, i] = (fn(x + dx) - fn(x - dx)) / (2 * eps)
    return J


@pytest.mark.parametrize("state", [
    np.array([3.0, 1.0, 2.5, 0.4, 0.2]),     # turning
    np.array([3.0, 1.0, 2.5, 0.4, 1e-7]),    # near-straight (ω→0 branch)
])
def test_ctrv_F_matches_finite_difference(state):
    # wrap_to_pi inside f makes ψ' jump at ±π; this state is far from the seam.
    np.testing.assert_allclose(ctrv_F(state), _numeric_jac(ctrv_f, state),
                               atol=1e-5)


from processing.tracking_cartesian import cartesian_h, cartesian_H


def test_h_radial_motion_full_doppler():
    # at (10,0) heading +x (toward +range) at 3 m/s -> r=10, θ=0, ṙ=+3
    z = cartesian_h(np.array([10.0, 0.0, 3.0, 0.0, 0.0]))
    np.testing.assert_allclose(z, [10.0, 0.0, 3.0], atol=1e-9)


def test_h_tangential_motion_zero_doppler():
    # at (10,0) heading +y (ψ=π/2) -> motion is cross-range -> ṙ = 0
    z = cartesian_h(np.array([10.0, 0.0, 3.0, np.pi / 2, 0.0]))
    assert z[2] == pytest.approx(0.0, abs=1e-9)


def test_H_matches_finite_difference():
    state = np.array([8.0, 2.0, 2.5, 0.4, 0.2])
    np.testing.assert_allclose(cartesian_H(state),
                               _numeric_jac(cartesian_h, state), atol=1e-5)


from processing.clustering import ClusterObject
from processing.tracking_cartesian import (
    ctrv_Q, ekf_predict, ekf_update, mahalanobis, object_to_measurement)


def test_ekf_update_reduces_error_and_covariance():
    state = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.diag([1.0, 1.0, 25.0, 2.5, 1.0])
    R = np.diag([0.04, 0.0025, 0.04])
    z = np.array([10.0, 0.0, 3.0])          # says ṙ = 3 (state has v=0)
    new_state, new_cov = ekf_update(state, cov, z, R)
    assert abs(new_state[2]) > 0.0          # speed pulled toward the measurement
    assert np.trace(new_cov) < np.trace(cov)
    np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-9)  # symmetric
    assert np.all(np.isfinite(new_cov))


def test_ctrv_Q_is_psd():
    Q = ctrv_Q(np.array([1.0, 2.0, 3.0, 0.5, 0.1]), sigma_a=1.0, sigma_yaw=0.3)
    assert Q.shape == (5, 5)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)


def test_object_to_measurement():
    z = object_to_measurement(ClusterObject(3.0, 4.0, -1.5, 1, 0.0))
    np.testing.assert_allclose(z, [5.0, np.arctan2(4.0, 3.0), -1.5], atol=1e-9)


def test_mahalanobis_zero_for_exact_match():
    state = np.array([8.0, 2.0, 1.0, 0.3, 0.0])
    cov = np.eye(5)
    R = np.diag([0.04, 0.0025, 0.04])
    z = cartesian_h(state)
    assert mahalanobis(state, cov, z, R) == pytest.approx(0.0, abs=1e-9)


from processing.tracking_cartesian import CartesianTrack, CartesianTracker


def _obj(x, y, vr):
    return ClusterObject(float(x), float(y), float(vr), 1, 0.0)


def _cv_objects(x0, y0, vx, vy, n):
    """A constant-velocity ground-truth track as a ClusterObject sequence,
    each with the exact radial velocity ṙ = (x·vx+y·vy)/r."""
    objs = []
    for k in range(n):
        x, y = x0 + vx * k, y0 + vy * k
        r = np.hypot(x, y)
        objs.append(_obj(x, y, (x * vx + y * vy) / r))
    return objs


def test_tracks_straight_line_one_stable_track():
    tr = CartesianTracker(CartesianTrackerConfig())
    seq = _cv_objects(10.0, 0.0, 0.0, 0.0, 8)          # static at (10,0)
    last = []
    for o in seq:
        last = tr.step([o])
    confirmed = [t for t in last if t.status == "confirmed"]
    assert len(confirmed) == 1
    t = confirmed[0]
    assert np.hypot(t.state[0] - 10.0, t.state[1] - 0.0) < 0.5   # position RMSE-ish
    assert abs(t.state[4]) < 0.1                                  # ω ≈ 0
    # ID stable across the run:
    ids = set()
    tr2 = CartesianTracker(CartesianTrackerConfig())
    for o in seq:
        for t in tr2.step([o]):
            ids.add(t.id)
    assert len(ids) == 1


def test_spawn_observability_psi_omega_var_shrinks():
    tr = CartesianTracker(CartesianTrackerConfig())
    seq = _cv_objects(10.0, 0.0, 0.5, 0.3, 6)
    tr.step([seq[0]])
    t0 = tr.tracks[0]
    psi_var0, w_var0 = t0.cov[3, 3], t0.cov[4, 4]
    assert psi_var0 > 1.0                       # large at spawn
    for o in seq[1:]:
        tr.step([o])
    t = tr.tracks[0]
    assert t.cov[3, 3] < psi_var0               # heading var shrinks (observable)
    assert np.isfinite(t.cov[4, 4])             # ω stays finite (unobservable on a
    #                                             straight track; convergence is
    #                                             tested on a turn in Task 7)


def test_two_separated_targets_two_tracks():
    tr = CartesianTracker(CartesianTrackerConfig())
    a = _cv_objects(10.0, 3.0, 0.0, 0.0, 8)
    b = _cv_objects(6.0, -4.0, 0.0, 0.0, 8)
    last = []
    for oa, ob in zip(a, b):
        last = tr.step([oa, ob])
    assert len([t for t in last if t.status == "confirmed"]) == 2


def test_azimuth_innovation_wraps_across_pi():
    # predicted heading/bearing just below +π, measurement just above -π:
    # raw difference ≈ -2π, wrapped must be small.
    from processing.tracking_cartesian import _innovation
    state = np.array([-10.0, 0.001, 1.0, 0.0, 0.0])     # θ_pred ≈ +π
    z = np.array([10.0001, -np.pi + 0.01, 1.0])         # θ_meas ≈ -π + 0.01
    y = _innovation(state, z)
    assert abs(y[1]) < 0.1                               # wrapped, not ~2π


def test_tracker_handles_target_crossing_behind():
    # target moving along -x through x≈0 region stays one track (bearing swings
    # through ±π); we sample y slightly off-axis so θ is well-defined.
    tr = CartesianTracker(CartesianTrackerConfig())
    seq = _cv_objects(-6.0, 0.5, 0.0, 0.0, 6)   # static behind, near -x axis
    last = []
    for o in seq:
        last = tr.step([o])
    assert len([t for t in last if t.status == "confirmed"]) == 1


# ---- Phase M4b: turn validation ----

def _ctrv_truth_objects(x0, y0, v0, psi0, omega, n):
    """Ground-truth turning trajectory rolled out with the CTRV model itself,
    emitted as ClusterObjects with the exact radial velocity."""
    state = np.array([x0, y0, v0, psi0, omega])
    objs = []
    for _ in range(n):
        z = cartesian_h(state)
        objs.append(_obj(state[0], state[1], z[2]))
        state = ctrv_f(state)
    return objs


def _position_rmse(tracker, objs, truth_xy):
    errs = []
    for o, (gx, gy) in zip(objs, truth_xy):
        tracks = tracker.step([o])
        conf = [t for t in tracks if t.status == "confirmed"]
        if conf:
            t = conf[0]
            errs.append(np.hypot(t.state[0] - gx, t.state[1] - gy))
    return float(np.sqrt(np.mean(np.square(errs[-5:]))))  # steady-state tail


def test_turn_omega_converges_and_beats_frozen():
    true_omega = 0.06
    objs = _ctrv_truth_objects(12.0, 0.0, 2.0, 0.3, true_omega, 30)
    truth_xy = [(o.x, o.y) for o in objs]

    # Lenient + identical measurement/gate for both runs, so the ω-frozen (CV)
    # baseline keeps lock and the only difference is the motion model.
    base = dict(sigma_r=0.4, sigma_az=0.08, sigma_rdot=0.4, gate_threshold=30.0)

    full = CartesianTracker(CartesianTrackerConfig(**base))
    rmse_full = _position_rmse(full, objs, truth_xy)
    t = [tr for tr in full.tracks if tr.status == "confirmed"][0]
    assert abs(t.state[4] - true_omega) < 0.025         # ω converged

    # ω-frozen baseline: no yaw process noise + zero init ω variance -> CV.
    frozen = CartesianTracker(
        CartesianTrackerConfig(sigma_yaw_accel=0.0, init_omega_var=0.0, **base))
    rmse_frozen = _position_rmse(frozen, objs, truth_xy)

    assert rmse_full < rmse_frozen                      # CTRV beats CV on a turn
