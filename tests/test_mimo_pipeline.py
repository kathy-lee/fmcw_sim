import numpy as np

from simulator.beat_signal_generator import load_scenario
from evaluation.mimo_pipeline import run_pipeline_mimo


def test_run_pipeline_mimo_tracks_single_target():
    cfg, targets = load_scenario("scenarios/scenario_mimo_2d.json")
    history = run_pipeline_mimo(cfg, targets, n_frames=10)
    assert len(history) == 10
    last = history[-1][1]                                  # snapshots, final frame
    confirmed = [s for s in last if s.status == "confirmed"]
    assert len(confirmed) == 1                             # one object tracked
    # GT is near (8, 2); the confirmed track should be close.
    s = confirmed[0]
    assert np.hypot(s.state[0] - 8.0, s.state[1] - 2.0) < 1.5


from evaluation.mimo_pipeline import (
    build_gt_xy, associate_xy, track_rmse_xy, CartesianTrackSnapshot)
from simulator.beat_signal_generator import FMCWBeatSignalGenerator


def test_build_gt_xy_converts_polar_to_cartesian():
    cfg, targets = load_scenario("scenarios/scenario_mimo_2d.json")
    gen = FMCWBeatSignalGenerator(cfg)
    gt = build_gt_xy(targets, gen, 3)
    assert len(gt) == 3
    _, rows = gt[0]
    gid, x, y, vx, vy, az = rows[0]
    assert np.hypot(x, y) > 0                       # placed off-origin
    assert abs(np.degrees(np.arctan2(y, x)) - az) < 1e-6


def test_associate_xy_matches_nearest():
    gt_rows = [(0, 8.0, 2.0, 0.0, 0.0, 14.0)]
    snaps = [CartesianTrackSnapshot(3, np.array([8.1, 2.0, 0, 0, 0]), "confirmed")]
    assert associate_xy(gt_rows, snaps, max_dist_m=1.0) == {0: 3}
    # too far -> unmatched
    far = [CartesianTrackSnapshot(3, np.array([20.0, 2.0, 0, 0, 0]), "confirmed")]
    assert associate_xy(gt_rows, far, max_dist_m=1.0) == {0: None}


def test_track_rmse_xy_zero_for_perfect_track():
    gt = [(0, [(0, 8.0, 2.0, 0.0, 0.0, 14.0)])]
    hist = [(0, [CartesianTrackSnapshot(3, np.array([8.0, 2.0, 0, 0, 0]), "confirmed")])]
    assoc = [(0, {0: 3})]
    assert track_rmse_xy(gt, hist, assoc) == 0.0
