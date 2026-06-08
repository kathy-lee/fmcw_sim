"""Tests for the A.4 cross-algorithm matrix harness.

aggregate() is tested on synthetic rows (fast, deterministic — the aggregation
rules are where the bugs hide). run_seed() gets one slow smoke test to lock the
row schema and the per-GT / run-level duplication.
"""

import matplotlib
matplotlib.use("Agg")


def test_aggregate_pools_per_gt_and_dedups_run_level():
    """Per-GT metrics pool over all GT-rows × seeds; run-level metrics are
    de-duplicated to one value per seed before aggregating, and classification
    rates are fractions over the seeds."""
    from evaluation.tracker_stress_matrix import aggregate

    # crossing_targets has 2 GTs. Two seeds, so 4 GT-rows for (crossing, kf).
    # Run-level metrics are duplicated across each run's 2 GT rows.
    rows = [
        # seed 0: a CLEAN run, 0 switches; per-GT Pd 1.0 and 0.8
        {"scenario": "crossing_targets", "algorithm": "kf", "seed": 0,
         "gt_id": 0, "pd": 1.0, "n_assoc_frames": 80, "distinct_track_ids": 1,
         "coast_frames": 0, "run_n_switches": 0,
         "run_false_confirmed_tracks": 0, "run_classification": "CLEAN"},
        {"scenario": "crossing_targets", "algorithm": "kf", "seed": 0,
         "gt_id": 1, "pd": 0.8, "n_assoc_frames": 64, "distinct_track_ids": 1,
         "coast_frames": 0, "run_n_switches": 0,
         "run_false_confirmed_tracks": 0, "run_classification": "CLEAN"},
        # seed 1: an ID SWITCH run, 2 switches; per-GT Pd 0.6 and 0.4
        {"scenario": "crossing_targets", "algorithm": "kf", "seed": 1,
         "gt_id": 0, "pd": 0.6, "n_assoc_frames": 48, "distinct_track_ids": 2,
         "coast_frames": 4, "run_n_switches": 2,
         "run_false_confirmed_tracks": 0, "run_classification": "ID SWITCH"},
        {"scenario": "crossing_targets", "algorithm": "kf", "seed": 1,
         "gt_id": 1, "pd": 0.4, "n_assoc_frames": 32, "distinct_track_ids": 2,
         "coast_frames": 6, "run_n_switches": 2,
         "run_false_confirmed_tracks": 0, "run_classification": "ID SWITCH"},
    ]

    summary = aggregate(rows)
    assert len(summary) == 1
    s = summary[0]
    assert s["scenario"] == "crossing_targets" and s["algorithm"] == "kf"
    assert s["n_seeds"] == 2

    # Per-GT Pd pooled over all 4 GT-rows: mean(1.0, 0.8, 0.6, 0.4) = 0.7.
    assert s["pd_mean"] == 0.7

    # Run-level switches de-duped to one per seed: mean(0, 2) = 1.0, NOT
    # mean(0,0,2,2). This is the key dedup assertion.
    assert s["switches_mean"] == 1.0

    # Classification rates are fractions over the 2 seeds.
    assert s["pct_clean"] == 0.5
    assert s["pct_id_switch"] == 0.5
    assert s["pct_track_break"] == 0.0
    assert s["pct_total_confusion"] == 0.0


def test_run_seed_returns_one_row_per_gt_with_run_level_dup():
    """run_seed on crossing_targets (2 GTs) returns 2 rows carrying the schema,
    with run-level metrics duplicated across the GT rows. Slow (full pipeline)."""
    from evaluation.tracker_stress_matrix import run_seed

    rows = run_seed("crossing_targets", "kf", seed=0, n_frames=30)
    assert len(rows) == 2
    expected_keys = {
        "scenario", "algorithm", "seed", "gt_id", "pd", "n_assoc_frames",
        "distinct_track_ids", "coast_frames", "run_n_switches",
        "run_false_confirmed_tracks", "run_classification",
    }
    for r in rows:
        assert set(r) == expected_keys
        assert r["scenario"] == "crossing_targets"
        assert r["algorithm"] == "kf"
        assert 0.0 <= r["pd"] <= 1.0
    # Run-level metrics identical across the two GT rows of one run.
    assert rows[0]["run_n_switches"] == rows[1]["run_n_switches"]
    assert rows[0]["run_classification"] == rows[1]["run_classification"]
