import matplotlib
matplotlib.use("Agg")

import pytest

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario


def test_build_gt_trajectory_first_frame_matches_initial_positions():
    """At frame 0, the GT trajectory should report each target's range
    extrapolated only by the half-frame offset — i.e., very close to its
    initial range_m."""
    from evaluation.tracker_stress import build_gt_trajectory

    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_crossing_targets.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)
    traj = build_gt_trajectory(initial_targets, gen, n_frames=5)

    assert len(traj) == 5
    frame0_idx, gt0 = traj[0]
    assert frame0_idx == 0
    assert len(gt0) == 2

    # Each GT tuple: (gt_id, range_m, velocity_mps, range_bin, doppler_bin)
    gt_a = gt0[0]
    assert gt_a[0] == 0                            # gt_id
    assert gt_a[2] == pytest.approx(1.0)           # velocity matches initial
    # range at frame 0 center ≈ initial + v * half-frame-time.
    # For target A initial=2.0 m, v=+1 m/s: should be ~2.0 within ~1 cm.
    assert gt_a[1] == pytest.approx(2.0, abs=0.01)

    gt_b = gt0[1]
    assert gt_b[0] == 1
    assert gt_b[2] == pytest.approx(-1.0)
    assert gt_b[1] == pytest.approx(6.0, abs=0.01)


def test_build_gt_trajectory_targets_meet_at_crossing_frame():
    """At frame 40 (t=2s at 20 Hz), both targets should be at range ~ 4 m."""
    from evaluation.tracker_stress import build_gt_trajectory

    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_crossing_targets.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)
    traj = build_gt_trajectory(initial_targets, gen, n_frames=41)

    _, gt_at_40 = traj[40]
    range_a = gt_at_40[0][1]
    range_b = gt_at_40[1][1]
    # They meet at ~4 m at t=2s. Allow a few cm of tolerance.
    assert abs(range_a - 4.0) < 0.05
    assert abs(range_b - 4.0) < 0.05


from collections import namedtuple
import numpy as np

# TrackLike mirrors the relevant fields of processing.tracking.Track:
#   id, state (2-array [range_bin, range_rate]), last_doppler_bin, status.
TrackLike = namedtuple("TrackLike", ["id", "state",
                                     "last_doppler_bin", "status"])


def _make_track(track_id, range_bin, doppler_bin, status="confirmed"):
    return TrackLike(
        id=track_id,
        state=np.array([range_bin, 0.0]),   # range_rate not used by eval
        last_doppler_bin=doppler_bin,
        status=status,
    )


def test_associate_empty_track_list_returns_none_for_each_gt():
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0, 0.0)]
    result = associate_per_frame(gt_list, [])
    assert result == {0: None}


def test_associate_single_close_target_matched():
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0, 0.0)]
    tracks = [_make_track(7, 40.2, 33.1)]
    result = associate_per_frame(gt_list, tracks)
    assert result == {0: 7}


def test_associate_single_far_target_not_matched():
    """If the closest track is beyond max_distance_bins, return None."""
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0, 0.0)]
    tracks = [_make_track(7, 70.0, 10.0)]
    result = associate_per_frame(gt_list, tracks, max_distance_bins=3.0)
    assert result == {0: None}


def test_associate_two_targets_obvious_mapping():
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [
        (0, 3.0, 1.0, 40.0, 33.0, 0.0),
        (1, 6.0, -1.0, 80.0, 31.0, 0.0),
    ]
    tracks = [
        _make_track(11, 80.1, 31.0),
        _make_track(10, 39.9, 33.0),
    ]
    result = associate_per_frame(gt_list, tracks)
    assert result == {0: 10, 1: 11}


def test_associate_hungarian_beats_greedy_on_crossed_positions():
    """Two GTs at close range, two tracks each near both. Hungarian's
    one-to-one constraint picks the minimum-total-cost pairing."""
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [
        (0, 3.0, 1.0, 40.0, 33.0, 0.0),
        (1, 3.1, 1.0, 41.0, 33.0, 0.0),
    ]
    tracks = [
        _make_track(100, 40.4, 33.0),
        _make_track(200, 40.6, 33.0),
    ]
    result = associate_per_frame(gt_list, tracks)
    # Hungarian min-total-cost: GT-0↔track 100 (cost 0.4), GT-1↔track 200 (cost 0.4).
    assert result == {0: 100, 1: 200}


def test_count_switches_empty_history():
    from evaluation.tracker_stress import count_id_switches
    assert count_id_switches([]) == []


def test_count_switches_constant_mapping():
    """No switch if a GT is consistently mapped to the same track."""
    from evaluation.tracker_stress import count_id_switches
    history = [
        (0, {0: 7}),
        (1, {0: 7}),
        (2, {0: 7}),
        (3, {0: 7}),
    ]
    assert count_id_switches(history) == []


def test_count_switches_persistent_switch_counted():
    """Single persistent switch is detected."""
    from evaluation.tracker_stress import count_id_switches
    history = [
        (0, {0: 7}),
        (1, {0: 7}),
        (2, {0: 9}),   # switch starts here
        (3, {0: 9}),   # persists for >= 2 frames
        (4, {0: 9}),
    ]
    events = count_id_switches(history, min_persistence_frames=2)
    assert len(events) == 1
    frame_idx, gt_id, old, new = events[0]
    assert (frame_idx, gt_id, old, new) == (2, 0, 7, 9)


def test_count_switches_one_frame_flipflop_not_counted():
    """A single-frame flip-flop is rejected by the persistence filter."""
    from evaluation.tracker_stress import count_id_switches
    history = [
        (0, {0: 7}),
        (1, {0: 7}),
        (2, {0: 9}),   # apparent switch
        (3, {0: 7}),   # back to original; flip-flop
        (4, {0: 7}),
    ]
    events = count_id_switches(history, min_persistence_frames=2)
    assert events == []


def test_count_switches_gap_then_same_id_not_a_switch():
    """If a GT goes unassigned then reacquires the same track ID, no switch."""
    from evaluation.tracker_stress import count_id_switches
    history = [
        (0, {0: 7}),
        (1, {0: 7}),
        (2, {0: None}),   # lost
        (3, {0: None}),   # still lost
        (4, {0: 7}),      # reacquired with same ID
        (5, {0: 7}),
    ]
    events = count_id_switches(history, min_persistence_frames=2)
    assert events == []


def test_count_switches_gap_then_different_id_is_a_switch():
    """If a GT is reacquired with a different track ID after a gap, that
    counts as a switch (the tracker lost identity continuity)."""
    from evaluation.tracker_stress import count_id_switches
    history = [
        (0, {0: 7}),
        (1, {0: 7}),
        (2, {0: None}),
        (3, {0: 9}),     # reacquired with different ID
        (4, {0: 9}),     # persists
    ]
    events = count_id_switches(history, min_persistence_frames=2)
    assert len(events) == 1
    frame_idx, gt_id, old, new = events[0]
    assert (gt_id, old, new) == (0, 7, 9)
    assert frame_idx == 3


def _history_all_clean(n_frames=10):
    """Helper: history where two GTs are perfectly tracked by two stable tracks."""
    return [(i, {0: 100, 1: 200}) for i in range(n_frames)]


def test_evaluate_clean_run():
    """Pd=1, 1 track ID per GT, 0 switches -> CLEAN."""
    from evaluation.tracker_stress import evaluate_run
    history = _history_all_clean(10)
    summary = evaluate_run(history, switch_events=[], tracks_history=[])
    assert summary["classification"] == "CLEAN"
    assert summary["per_gt"][0]["pd"] == 1.0
    assert summary["per_gt"][1]["pd"] == 1.0
    assert summary["per_gt"][0]["distinct_track_ids"] == 1
    assert summary["per_gt"][1]["distinct_track_ids"] == 1


def test_evaluate_id_switch_only():
    """Pd=1, >=1 switch, but no track break -> ID SWITCH."""
    from evaluation.tracker_stress import evaluate_run
    history = [
        (0, {0: 100, 1: 200}),
        (1, {0: 100, 1: 200}),
        (2, {0: 200, 1: 100}),   # swap
        (3, {0: 200, 1: 100}),
        (4, {0: 200, 1: 100}),
    ]
    switch_events = [(2, 0, 100, 200), (2, 1, 200, 100)]
    summary = evaluate_run(history, switch_events=switch_events,
                            tracks_history=[])
    assert summary["classification"] == "ID SWITCH"
    assert summary["per_gt"][0]["pd"] == 1.0
    assert summary["per_gt"][1]["pd"] == 1.0


def test_evaluate_track_break():
    """One GT's Pd < 0.5 -> TRACK BREAK."""
    from evaluation.tracker_stress import evaluate_run
    history = [
        (0, {0: 100, 1: 200}),
        (1, {0: None, 1: 200}),
        (2, {0: None, 1: 200}),
        (3, {0: None, 1: 200}),
        (4, {0: None, 1: 200}),
    ]
    summary = evaluate_run(history, switch_events=[], tracks_history=[])
    # GT-0: Pd = 1/5 = 0.2 < 0.5 -> TRACK BREAK
    assert summary["classification"] == "TRACK BREAK"
    assert summary["per_gt"][0]["pd"] == pytest.approx(0.2)
    assert summary["per_gt"][1]["pd"] == 1.0


def test_evaluate_total_confusion():
    """ID switches AND track break -> TOTAL CONFUSION."""
    from evaluation.tracker_stress import evaluate_run
    history = [
        (0, {0: 100, 1: 200}),
        (1, {0: 200, 1: None}),   # GT-0 switches, GT-1 lost
        (2, {0: 200, 1: None}),
        (3, {0: 200, 1: None}),
    ]
    switch_events = [(1, 0, 100, 200)]
    summary = evaluate_run(history, switch_events=switch_events,
                            tracks_history=[])
    assert summary["classification"] == "TOTAL CONFUSION"


def test_evaluate_pd_per_gt():
    """Pd is fraction of frames with non-None association."""
    from evaluation.tracker_stress import evaluate_run
    history = [
        (0, {0: 100}),
        (1, {0: None}),
        (2, {0: 100}),
        (3, {0: None}),
    ]
    summary = evaluate_run(history, switch_events=[], tracks_history=[])
    assert summary["per_gt"][0]["pd"] == 0.5


def test_evaluate_distinct_track_ids_count():
    """Distinct non-None track IDs that ever associated with each GT."""
    from evaluation.tracker_stress import evaluate_run
    history = [
        (0, {0: 100}),
        (1, {0: 100}),
        (2, {0: 200}),
        (3, {0: 200}),
        (4, {0: 300}),
    ]
    summary = evaluate_run(history, switch_events=[], tracks_history=[])
    assert summary["per_gt"][0]["distinct_track_ids"] == 3


def test_evaluate_coast_frames_per_gt():
    """coast_frames counts frames where a track that ever-associated with
    this GT was alive in status='coasting'."""
    from evaluation.tracker_stress import evaluate_run, TrackSnapshot
    import numpy as np

    # GT 0 is associated with track 100 on frames 0-1 and 4. On frames 2 and 3,
    # track 100 is alive but coasting (not associated). Expected: coast_frames=2.
    history = [
        (0, {0: 100}),
        (1, {0: 100}),
        (2, {0: None}),
        (3, {0: None}),
        (4, {0: 100}),
    ]
    state = np.array([10.0, 0.0])
    tracks_history = [
        (0, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="confirmed")]),
        (1, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="confirmed")]),
        (2, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="coasting")]),
        (3, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="coasting")]),
        (4, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="confirmed")]),
    ]
    summary = evaluate_run(history, switch_events=[],
                            tracks_history=tracks_history)
    assert summary["per_gt"][0]["coast_frames"] == 2


def test_evaluate_coast_frames_excludes_unassociated_tracks():
    """A track that coasts but was never associated with any GT does not
    add to anyone's coast_frames count (e.g., a false-track corpse)."""
    from evaluation.tracker_stress import evaluate_run, TrackSnapshot
    import numpy as np

    history = [
        (0, {0: 100}),
        (1, {0: 100}),
    ]
    state = np.array([10.0, 0.0])
    tracks_history = [
        (0, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="confirmed"),
             TrackSnapshot(id=999, state=state, last_doppler_bin=32.0,
                           status="coasting")]),  # never assoc with any GT
        (1, [TrackSnapshot(id=100, state=state, last_doppler_bin=32.0,
                           status="confirmed"),
             TrackSnapshot(id=999, state=state, last_doppler_bin=32.0,
                           status="coasting")]),
    ]
    summary = evaluate_run(history, switch_events=[],
                            tracks_history=tracks_history)
    assert summary["per_gt"][0]["coast_frames"] == 0


def test_pilot_smoke_run_10_frames():
    """End-to-end smoke test: run pipeline + eval for 10 frames, verify
    basic invariants. Doesn't re-test the underlying components -- just
    catches import/glue bugs."""
    from evaluation.tracker_stress import (
        build_gt_trajectory, run_pipeline, associate_per_frame,
        count_id_switches, evaluate_run,
    )

    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_crossing_targets.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(initial_targets, gen, n_frames=10)
    tracks_history = run_pipeline(radar_cfg, initial_targets, n_frames=10)

    assert len(gt_trajectory) == 10
    assert len(tracks_history) == 10

    association_history = []
    for (f_gt, gt_at_frame), (f_tr, tracks) in zip(gt_trajectory,
                                                    tracks_history):
        assert f_gt == f_tr
        mapping = associate_per_frame(gt_at_frame, tracks)
        association_history.append((f_gt, mapping))

    switch_events = count_id_switches(association_history)
    summary = evaluate_run(association_history, switch_events,
                            tracks_history)
    assert "classification" in summary
    assert "per_gt" in summary
    assert set(summary["per_gt"].keys()) == {0, 1}
    # coast_frames is reported per GT; non-negative integer.
    for gt_id in summary["per_gt"]:
        assert summary["per_gt"][gt_id]["coast_frames"] >= 0


def test_pilot_smoke_run_imm_10_frames():
    """End-to-end smoke test with IMMTracker driving the harness.

    Verifies that IMMTrack exposes the same field interface the harness
    reads off Track (id, state, last_doppler_bin, status) so the existing
    snapshot / association / plotting code 'just works'.
    """
    import numpy as np

    from processing.config import IMMConfig, MotionModelSpec
    from processing.imm import IMMTracker
    from processing.cfar import cfar_2d
    from processing.clustering import cluster
    from processing.range_doppler import range_doppler

    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_crossing_targets.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)

    imm_cfg = IMMConfig(
        models=[
            MotionModelSpec(motion_model="cv"),
            MotionModelSpec(motion_model="drag",
                            motion_params={"drag_coefficient_k": 0.05}),
        ],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        initial_mode_probs=[0.5, 0.5],
    )
    tracker = IMMTracker(imm_cfg)

    for frame_idx in range(10):
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", pfa=1e-4, window="hann",
                       dc_removal=True, clutter_removal=False)
        mask[:, 0] = False
        dets = cluster(mask, rd_db, method="peak", min_cells=2,
                       peak_neighborhood=3)
        tracks = tracker.step(dets)

        for t in tracks:
            # Fields the harness reads off any Track-like object:
            assert hasattr(t, "id")
            assert hasattr(t, "state")
            assert t.state.shape == (2,)
            assert hasattr(t, "last_doppler_bin")
            assert hasattr(t, "status")
            # IMM-specific invariant
            assert abs(sum(t.mode_probs) - 1.0) < 1e-6


def test_track_position_rmse_constant_offset():
    """RMSE equals a known constant range offset; unassociated frames are
    excluded from the average."""
    import numpy as np
    from evaluation.tracker_stress import track_position_rmse, TrackSnapshot

    range_per_bin = 0.1
    # Single GT at 4.0 m every frame (range_bin/doppler fields unused here).
    gt_trajectory = [
        (0, [(0, 4.0, 0.0, 0.0, 0.0)]),
        (1, [(0, 4.0, 0.0, 0.0, 0.0)]),
        (2, [(0, 4.0, 0.0, 0.0, 0.0)]),
    ]
    # Track id=7 sits at 45 bins -> 4.5 m -> a constant 0.5 m offset from GT.
    snap = TrackSnapshot(id=7, state=np.array([45.0, 0.0]),
                         last_doppler_bin=0.0, status="confirmed")
    tracks_history = [(0, [snap]), (1, [snap]), (2, [snap])]
    # Frame 1 has no association and must be excluded; frames 0 and 2 count.
    association_history = [(0, {0: 7}), (1, {0: None}), (2, {0: 7})]

    rmse = track_position_rmse(gt_trajectory, tracks_history,
                               association_history, range_per_bin)
    assert rmse == pytest.approx(0.5)


def test_track_position_rmse_no_associations_is_nan():
    """With no associated frames the RMSE is NaN (no data to average)."""
    import math
    from evaluation.tracker_stress import track_position_rmse

    gt_trajectory = [(0, [(0, 4.0, 0.0, 0.0, 0.0)])]
    tracks_history = [(0, [])]
    association_history = [(0, {0: None})]

    rmse = track_position_rmse(gt_trajectory, tracks_history,
                               association_history, range_per_bin=0.1)
    assert math.isnan(rmse)


def test_count_false_confirmed_tracks_counts_clutter_only():
    """A confirmed track never associated with any GT is 'false' (clutter-born);
    the real track (ever associated) and tentative tracks are excluded."""
    import numpy as np
    from evaluation.tracker_stress import (
        count_false_confirmed_tracks, TrackSnapshot,
    )

    state = np.array([10.0, 0.0])

    def snap(i, status):
        return TrackSnapshot(id=i, state=state, last_doppler_bin=0.0,
                             status=status)

    # Track 1 = the real target (it is associated below). 99 and 100 = clutter
    # tracks (confirmed, never associated). 50 = a tentative clutter track that
    # must NOT count (only confirmed tracks are false-confirmed).
    tracks_history = [
        (0, [snap(1, "confirmed"), snap(99, "confirmed"),
             snap(50, "tentative")]),
        (1, [snap(1, "confirmed"), snap(99, "confirmed"),
             snap(100, "confirmed")]),
    ]
    association_history = [(0, {0: 1}), (1, {0: 1})]

    result = count_false_confirmed_tracks(association_history, tracks_history)
    # 99 and 100 are false-confirmed; 1 is real, 50 is tentative.
    assert result["n_false_confirmed"] == 2
    # Peak simultaneous: frame 1 carries 99 and 100 together.
    assert result["max_simultaneous_false"] == 2


def test_count_false_confirmed_tracks_zero_when_all_associated():
    """No false tracks when every confirmed track is the associated GT track."""
    import numpy as np
    from evaluation.tracker_stress import (
        count_false_confirmed_tracks, TrackSnapshot,
    )

    snap = TrackSnapshot(id=1, state=np.array([10.0, 0.0]),
                         last_doppler_bin=0.0, status="confirmed")
    tracks_history = [(0, [snap]), (1, [snap])]
    association_history = [(0, {0: 1}), (1, {0: 1})]

    result = count_false_confirmed_tracks(association_history, tracks_history)
    assert result["n_false_confirmed"] == 0
    assert result["max_simultaneous_false"] == 0


def test_run_pipeline_confirm_overrides_reach_tracker_config():
    """confirm_M/confirm_N plumb through run_pipeline to TrackerConfig, whose
    validation rejects confirm_M > confirm_N -- proving the override is wired
    (it raises at tracker construction, before any frame is generated)."""
    from evaluation.tracker_stress import run_pipeline

    radar_cfg, targets = load_scenario(
        "scenarios/scenario_missed_detections.json"
    )
    with pytest.raises(ValueError):
        run_pipeline(radar_cfg, targets, n_frames=1,
                     confirm_M=5, confirm_N=3)


def test_run_pipeline_seed_override_reseeds_realization():
    """The seed override re-seeds the radar noise: passing the scenario's own
    seed reproduces the no-override run exactly, while a different seed changes
    the (marginal-SNR) realization."""
    from evaluation.tracker_stress import run_pipeline

    radar_cfg, targets = load_scenario(
        "scenarios/scenario_missed_detections.json"
    )

    def confirmed_signature(tracks_history):
        # Scalar summary sensitive to the noise realization: total confirmed
        # snapshots across all frames.
        return sum(1 for _, snaps in tracks_history for s in snaps
                   if s.status == "confirmed")

    baseline = confirmed_signature(run_pipeline(radar_cfg, targets, 80))
    same = confirmed_signature(
        run_pipeline(radar_cfg, targets, 80, seed=radar_cfg.seed)
    )
    different = confirmed_signature(
        run_pipeline(radar_cfg, targets, 80, seed=radar_cfg.seed + 4242)
    )

    assert same == baseline          # same seed -> identical realization
    assert different != baseline     # different seed -> different realization


def test_run_pipeline_fluctuation_seed_is_deterministic():
    import numpy as np
    from evaluation.tracker_stress import run_pipeline
    from simulator.beat_signal_generator import RadarConfig, Target

    cfg = RadarConfig(noise_power=1e-4, frame_period_s=0.05)
    targets = [Target(range_m=4.0, velocity_mps=0.5, amplitude=0.0009,
                      fluctuation={"model": "swerling1"})]
    h1 = run_pipeline(cfg, targets, 10, seed=3, fluctuation_seed=42)
    h2 = run_pipeline(cfg, targets, 10, seed=3, fluctuation_seed=42)
    assert len(h1) == len(h2) == 10
    for (f1, s1), (f2, s2) in zip(h1, h2):
        assert f1 == f2
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.id == b.id
            assert a.status == b.status
            np.testing.assert_array_equal(a.state, b.state)


def test_associate_per_frame_span_reduces_to_point_at_zero_width():
    """With span half-width 0, association is identical to the point matcher."""
    from evaluation.tracker_stress import associate_per_frame
    # gt tuple now: (gt_id, range_m, v, range_bin, doppler_bin, half_width_bins)
    gt = [(0, 4.0, 0.5, 50.0, 32.0, 0.0)]
    tracks = [_make_track(7, 51.0, 32.0)]  # 1 bin away -> within 3-bin gate
    out = associate_per_frame(gt, tracks)
    assert out[0] == 7


def test_associate_per_frame_span_associates_within_extent():
    """A track at the blob edge (4 bins from centre, span half-width 4)
    associates; the same track would miss a 3-bin point gate."""
    from evaluation.tracker_stress import associate_per_frame
    gt = [(0, 4.0, 0.5, 50.0, 32.0, 4.0)]
    edge = [_make_track(7, 54.0, 32.0)]   # 4 bins from centre, on span edge
    assert associate_per_frame(gt, edge)[0] == 7
    far = [_make_track(7, 58.0, 32.0)]    # 8 bins from centre -> >3 beyond edge
    assert associate_per_frame(gt, far)[0] is None


def test_track_position_rmse_cutoff_charges_missed_frames():
    """With cutoff_m set (GOSPA convention), a frame with no associated track
    contributes the cutoff instead of being excluded; an on-centre track
    contributes zero. Two frames (one on-centre, one missed) -> RMSE =
    cutoff/sqrt(2)."""
    import math
    from evaluation.tracker_stress import track_position_rmse, TrackSnapshot
    range_per_bin = 0.1
    gt_traj = [(0, [(0, 4.0, 0.5, 40.0, 32.0, 4.0)]),
               (1, [(0, 4.0, 0.5, 40.0, 32.0, 4.0)])]
    # Frame 0: track exactly on the true range (4.0 m -> 40 bins). Frame 1: none.
    snap = TrackSnapshot(id=7, state=np.array([40.0, 0.0]),
                         last_doppler_bin=32.0, status="confirmed")
    tracks_hist = [(0, [snap]), (1, [])]
    assoc = [(0, {0: 7}), (1, {0: None})]
    rmse = track_position_rmse(gt_traj, tracks_hist, assoc, range_per_bin,
                               cutoff_m=2.0)
    assert rmse == pytest.approx(2.0 / math.sqrt(2.0))


def test_run_pipeline_range_meas_var_override_is_deterministic():
    import numpy as np
    from evaluation.tracker_stress import run_pipeline
    from simulator.beat_signal_generator import RadarConfig, Target
    cfg = RadarConfig(noise_power=1e-4, frame_period_s=0.05)
    targets = [Target(range_m=4.0, velocity_mps=0.5, amplitude=0.0009,
                      extent={"n_scatterers": 10, "range_extent_m": 0.6})]
    h1 = run_pipeline(cfg, targets, 8, seed=3, range_meas_var=5.0)
    h2 = run_pipeline(cfg, targets, 8, seed=3, range_meas_var=5.0)
    assert len(h1) == len(h2) == 8
    for (_, s1), (_, s2) in zip(h1, h2):
        assert len(s1) == len(s2)


def test_run_pipeline_cluster_method_param():
    """Default clustering is unchanged (peak); dbscan is selectable. Both run
    and return n_frames of history deterministically."""
    from evaluation.tracker_stress import run_pipeline
    from simulator.beat_signal_generator import RadarConfig, Target
    cfg = RadarConfig(noise_power=1e-4, frame_period_s=0.05)
    targets = [Target(range_m=4.0, velocity_mps=0.5, amplitude=0.0009,
                      extent={"n_scatterers": 10, "range_extent_m": 0.6})]
    h_default = run_pipeline(cfg, targets, 6, seed=3)
    h_db = run_pipeline(cfg, targets, 6, seed=3, cluster_method="dbscan",
                        cluster_min_cells=1, cluster_eps=2.0,
                        cluster_min_samples=1)
    assert len(h_default) == len(h_db) == 6


def test_associate_per_frame_span_keeps_doppler_tight():
    """The span loosens RANGE only: a track inside the range span but far in
    Doppler is still rejected (span is range-only; Doppler stays tight)."""
    from evaluation.tracker_stress import associate_per_frame
    gt = [(0, 4.0, 0.5, 50.0, 32.0, 4.0)]
    # At the range centre (dist-to-span 0) but 5 Doppler bins off -> reject.
    off = [_make_track(7, 50.0, 37.0)]
    assert associate_per_frame(gt, off)[0] is None
    # Same range, Doppler within the 3-bin gate -> matched.
    on = [_make_track(7, 50.0, 33.0)]
    assert associate_per_frame(gt, on)[0] == 7


def test_track_position_rmse_cutoff_caps_large_associated_error():
    """An associated frame whose error exceeds the cutoff is capped at it
    (the min(err, cutoff) branch), so a wild estimate can't blow up the RMSE
    beyond the cutoff."""
    from evaluation.tracker_stress import track_position_rmse, TrackSnapshot
    range_per_bin = 0.1
    gt_traj = [(0, [(0, 4.0, 0.5, 40.0, 32.0, 4.0)])]
    # Track at 100 bins -> 10 m -> 6 m error, well beyond the 2 m cutoff.
    snap = TrackSnapshot(id=7, state=np.array([100.0, 0.0]),
                         last_doppler_bin=32.0, status="confirmed")
    tracks_hist = [(0, [snap])]
    assoc = [(0, {0: 7})]
    rmse = track_position_rmse(gt_traj, tracks_hist, assoc, range_per_bin,
                              cutoff_m=2.0)
    assert rmse == pytest.approx(2.0)   # single frame, error capped at cutoff
