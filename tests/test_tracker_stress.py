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
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0)]
    result = associate_per_frame(gt_list, [])
    assert result == {0: None}


def test_associate_single_close_target_matched():
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0)]
    tracks = [_make_track(7, 40.2, 33.1)]
    result = associate_per_frame(gt_list, tracks)
    assert result == {0: 7}


def test_associate_single_far_target_not_matched():
    """If the closest track is beyond max_distance_bins, return None."""
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [(0, 3.0, 1.0, 40.0, 33.0)]
    tracks = [_make_track(7, 70.0, 10.0)]
    result = associate_per_frame(gt_list, tracks, max_distance_bins=3.0)
    assert result == {0: None}


def test_associate_two_targets_obvious_mapping():
    from evaluation.tracker_stress import associate_per_frame
    gt_list = [
        (0, 3.0, 1.0, 40.0, 33.0),
        (1, 6.0, -1.0, 80.0, 31.0),
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
        (0, 3.0, 1.0, 40.0, 33.0),
        (1, 3.1, 1.0, 41.0, 33.0),
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
