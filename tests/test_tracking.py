import math

import numpy as np
import pytest

from processing.tracking import Track, _kf_predict, _kf_update


def _make_track():
    return Track(
        id=0,
        state=np.array([100.0, 1.5]),
        cov=np.diag([0.5, 4.0]),
        last_doppler_bin=4.0,
        hits=1,
        misses=0,
        age=1,
        status="tentative",
        hit_history=None,    # filled by Tracker; not exercised here
        history=None,
    )


def test_kf_predict_advances_range_by_velocity_one_step():
    state = np.array([100.0, 1.5])
    cov = np.diag([0.5, 4.0])
    q = 0.5
    new_state, new_cov = _kf_predict(state, cov, q)
    assert new_state[0] == pytest.approx(101.5, abs=1e-9)
    assert new_state[1] == pytest.approx(1.5, abs=1e-9)
    # Position covariance should grow.
    assert new_cov[0, 0] > cov[0, 0]


def test_kf_update_pulls_state_toward_measurement():
    state = np.array([100.0, 0.0])
    cov = np.diag([1.0, 4.0])
    z_range = 102.0
    sigma_r2 = 0.5
    new_state, new_cov = _kf_update(state, cov, z_range, sigma_r2)
    # Estimate moves toward the measurement, but not all the way (Kalman gain < 1).
    assert 100.0 < new_state[0] < 102.0
    # Covariance shrinks after update.
    assert new_cov[0, 0] < cov[0, 0]


def test_kf_converges_with_repeated_measurements():
    state = np.array([0.0, 0.0])
    cov = np.diag([100.0, 100.0])
    truth_pos = 50.0
    truth_vel = 1.0
    sigma_r2 = 0.5
    q = 0.1
    rng = np.random.default_rng(0)
    for k in range(40):
        state, cov = _kf_predict(state, cov, q)
        z = (truth_pos + (k + 1) * truth_vel) + rng.normal(0, math.sqrt(sigma_r2))
        state, cov = _kf_update(state, cov, z, sigma_r2)
    assert state[0] == pytest.approx(truth_pos + 40 * truth_vel, abs=2.0)
    assert state[1] == pytest.approx(truth_vel, abs=0.5)


from processing.tracking import Tracker
from processing.config import TrackerConfig
from processing.clustering import Detection


def _det(range_bin: float, doppler_bin: float, peak_db: float = 5.0):
    return Detection(range_bin=range_bin, doppler_bin=doppler_bin,
                     peak_db=peak_db, cell_count=1)


def _make_tracker_nn(**overrides):
    cfg = TrackerConfig(method="nn", **overrides)
    return Tracker(cfg)


def test_tracker_spawn_then_confirm_after_M_hits():
    tr = _make_tracker_nn(confirm_M=3, confirm_N=5, max_misses_tentative=2)
    # 3 consecutive hits at the same location (with small motion).
    for k in range(3):
        out = tr.step([_det(range_bin=100.0 + k, doppler_bin=4.0)])
    assert len(out) == 1
    assert out[0].status == "confirmed"


def test_tracker_tentative_dies_after_too_many_misses():
    tr = _make_tracker_nn(confirm_M=3, confirm_N=5, max_misses_tentative=2)
    out = tr.step([_det(range_bin=100.0, doppler_bin=4.0)])
    assert len(out) == 1 and out[0].status == "tentative"
    # 2 empty frames -> tentative deleted.
    out = tr.step([])
    out = tr.step([])
    out = tr.step([])
    assert all(t.status == "confirmed" for t in out) or len(out) == 0
    # Should be empty: tentative track dropped.
    assert len(out) == 0


def test_tracker_confirmed_track_coasts_then_deletes():
    tr = _make_tracker_nn(confirm_M=2, confirm_N=3, max_coast=3)
    tr.step([_det(100.0, 4.0)])
    tr.step([_det(101.0, 4.0)])
    out = tr.step([_det(102.0, 4.0)])
    assert len(out) == 1 and out[0].status == "confirmed"
    track_id = out[0].id

    # 3 misses -> coasting then deleted.
    for k in range(3):
        out = tr.step([])
    assert all(t.status == "coasting" for t in out) or len(out) == 0
    out = tr.step([])
    assert track_id not in {t.id for t in out}, "track should be deleted after max_coast"


def test_tracker_single_target_linear_motion_id_stable():
    tr = _make_tracker_nn(confirm_M=3, confirm_N=5)
    track_ids_seen = set()
    for k in range(20):
        z_range = 100.0 + 1.0 * k     # 1 bin/frame
        out = tr.step([_det(z_range, doppler_bin=4.0)])
        for t in out:
            track_ids_seen.add(t.id)
    confirmed = [t for t in out if t.status == "confirmed"]
    assert len(confirmed) == 1
    assert len(track_ids_seen) == 1, f"unexpected ID churn: {track_ids_seen}"
    # State at end should be near (100 + 19*1, 1.0) = (119, 1.0).
    assert confirmed[0].state[0] == pytest.approx(119.0, abs=2.0)
    assert confirmed[0].state[1] == pytest.approx(1.0, abs=0.3)


def test_tracker_two_well_separated_targets():
    tr = _make_tracker_nn(confirm_M=3, confirm_N=5)
    for k in range(10):
        out = tr.step([
            _det(100.0 + k, doppler_bin=4.0),
            _det(400.0 + 0.5 * k, doppler_bin=-3.0),
        ])
    confirmed = [t for t in out if t.status == "confirmed"]
    assert len(confirmed) == 2
    ids = {t.id for t in confirmed}
    assert len(ids) == 2


def test_tracker_reset_clears_state():
    tr = _make_tracker_nn(confirm_M=2, confirm_N=3)
    tr.step([_det(100.0, 4.0)])
    tr.step([_det(101.0, 4.0)])
    out = tr.step([_det(102.0, 4.0)])
    assert len(out) == 1
    tr.reset()
    out = tr.step([])
    assert out == []


def _make_tracker(method: str, **overrides):
    cfg = TrackerConfig(method=method, **overrides)
    return Tracker(cfg)


def test_gnn_preserves_ids_through_crossing_targets():
    """Two targets crossing in (range, doppler) space.

    Truth: track A goes from (range=100, dop=+3) to (range=120, dop=+3) over 20 frames
           moving +1 bin/frame in range.
           Track B goes from (range=120, dop=-3) to (range=100, dop=-3) moving -1 bin/frame.
           They cross around frame 10 in the (range, doppler) plane.

    Greedy NN may swap IDs at the crossing; GNN should preserve IDs.
    """
    def make_dets(k: int):
        a = _det(range_bin=100.0 + k, doppler_bin=+3.0)
        b = _det(range_bin=120.0 - k, doppler_bin=-3.0)
        return [a, b]

    # GNN run.
    tr_gnn = _make_tracker("gnn", confirm_M=3, confirm_N=5)
    id_history_gnn = []
    for k in range(20):
        out = tr_gnn.step(make_dets(k))
        confirmed = [(t.id, t.last_doppler_bin) for t in out if t.status == "confirmed"]
        confirmed.sort(key=lambda x: x[1])  # sort by doppler so order is stable
        id_history_gnn.append(tuple(t[0] for t in confirmed))

    # After confirmation, the per-doppler ID order should be stable across all frames.
    confirmed_frames = [ids for ids in id_history_gnn if len(ids) == 2]
    assert len(confirmed_frames) >= 5, "expected several frames with both confirmed"
    first = confirmed_frames[0]
    for later in confirmed_frames[1:]:
        assert later == first, (
            f"GNN swapped IDs across crossing: {first} -> {later}"
        )
