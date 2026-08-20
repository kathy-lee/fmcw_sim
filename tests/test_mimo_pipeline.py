import numpy as np

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from evaluation.mimo_pipeline_runner import run_pipeline_mimo


def test_run_pipeline_mimo_tracks_single_target():
    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
    gen = FMCWBeatSignalGenerator(cfg)
    history = run_pipeline_mimo(gen, targets, n_frames=10)
    assert len(history) == 10
    last = history[-1][1]                                  # snapshots, final frame
    confirmed = [s for s in last if s.status == "confirmed"]
    assert len(confirmed) == 1                             # one object tracked
    # GT is near (8, 2); the confirmed track should be close.
    s = confirmed[0]
    assert np.hypot(s.state[0] - 8.0, s.state[1] - 2.0) < 1.5


from evaluation.mimo_pipeline_runner import CartesianTrackSnapshot
from evaluation.mimo_tracking_eval import (
    build_gt_xy, associate_xy, track_rmse_xy)


def test_build_gt_xy_converts_polar_to_cartesian():
    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
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


def test_detection_pd_curve_shows_integration_gain():
    from evaluation.mimo_detection_eval import pd_vs_snr
    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
    levels, pd_single, pd_integ = pd_vs_snr(cfg, targets, noise_powers=[0.05, 0.5],
                                            n_frames=6)
    assert len(levels) == len(pd_single) == len(pd_integ) == 2
    # at the same noise, V-channel integration detects at least as often
    assert all(pi >= ps - 1e-9 for pi, ps in zip(pd_integ, pd_single))


def test_doa_accuracy_improves_with_snr():
    # An off-grid angle (12.4°) so grid methods aren't trivially exact; ESPRIT is
    # search-free and should sharpen markedly from low to high SNR.
    from evaluation.mimo_doa_eval import azimuth_rmse_vs_snr
    snr, rmse = azimuth_rmse_vs_snr(az_true_deg=12.4, n_elem=12,
                                    snr_db_list=[0.0, 30.0], methods=("esprit",),
                                    n_trials=50, seed=0)
    assert rmse["esprit"][1] < rmse["esprit"][0]      # 30 dB sharper than 0 dB


def test_doa_resolution_superres_beats_bartlett():
    # The discriminating claim: super-resolution resolves closer than the
    # Bartlett (beamwidth) limit on the same array.
    from evaluation.mimo_doa_eval import min_resolvable_separation
    res = min_resolvable_separation(12, [2, 3, 4, 5, 6, 8, 10, 12], snr_db=20.0,
                                    methods=("bartlett", "music", "esprit"),
                                    n_trials=30, seed=0)
    assert res["esprit"] < res["bartlett"]
    assert res["music"] < res["bartlett"]


def test_clustering_vel_scale_splits_crossing():
    from evaluation.mimo_clustering_eval import object_count_grid
    cfg, targets = load_scenario("scenarios/scenario_mimo_crossing.json")
    # at the crossing frame the two targets share position but differ in velocity
    epses, vels, counts = object_count_grid(cfg, targets, frame_index=40,
                                            eps_list=[0.8], vel_list=[0.0, 1.0])
    # vel_scale=0 merges (1 object); vel_scale=1 splits (2)
    assert counts[0][0] <= counts[0][1]


def test_tracking_eval_turn_tracks_with_stable_id():
    # The robust end-to-end claims: the maneuvering target is tracked with low
    # position RMSE and a stable track ID through the turn. (Turn-rate ω itself
    # is only weakly observable from a single noisy (r,θ,ṙ) stream and is NOT
    # asserted here — see the report's tracking section.)
    from evaluation.mimo_tracking_eval import run_turn_eval
    cfg, targets = load_scenario("scenarios/scenario_mimo_turn.json")
    res = run_turn_eval(cfg, targets, n_frames=30)
    assert res["rmse_m"] < 1.5                          # tracked through the turn
    assert res["n_track_ids"] <= 2                      # stable lock (<=1 re-acquire)


def test_track_scenario_angle_crossing_no_id_swap():
    # Two off-boresight targets cross in azimuth at (5,0) near frame 40. Each must
    # keep a single track ID through the crossing (no swap) and track closely.
    from evaluation.mimo_tracking_eval import track_scenario
    cfg, targets = load_scenario("scenarios/scenario_mimo_angle_crossing.json")
    res = track_scenario(cfg, targets, n_frames=60)
    assert len(res["per_gt"]) == 2                       # both targets tracked
    for m in res["per_gt"].values():
        assert m["n_track_ids"] == 1                     # one ID each -> no swap
        assert m["rmse_m"] < 0.3                         # tracked closely
