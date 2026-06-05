"""Smoke tests for the two 1-D missed-detection sweeps (A.1.2).

These run the full pipeline on a short frame budget, so they are slow-ish but
guard the harness glue: the public run_one() shape, the CSV-bound metric keys,
and — for the q sweep — the regression guard that q is inert on association
(n_assoc constant across q), which is the whole reason the sweeps are split.
"""

import matplotlib
matplotlib.use("Agg")

SCENARIO = "scenarios/scenario_missed_detections.json"
_VALID_CLASSES = {"CLEAN", "ID SWITCH", "TRACK BREAK", "TOTAL CONFUSION"}


def test_maxcoast_sweep_run_one_returns_discrete_row():
    from evaluation.tracker_stress_maxcoast_sweep import run_one

    row = run_one(SCENARIO, n_frames=30, max_coast=5)
    assert set(row) == {"max_coast", "pd", "distinct_tracks",
                        "coast_frames", "classification"}
    assert row["max_coast"] == 5
    assert 0.0 <= row["pd"] <= 1.0
    assert row["classification"] in _VALID_CLASSES


def test_q_sweep_run_one_returns_rmse_row():
    from evaluation.tracker_stress_q_sweep import run_one

    row = run_one(SCENARIO, n_frames=30, process_noise_q=1.0)
    assert set(row) == {"process_noise_q", "rmse_m", "n_assoc"}
    assert row["process_noise_q"] == 1.0
    assert row["rmse_m"] >= 0.0
    assert isinstance(row["n_assoc"], int)


def test_q_sweep_n_assoc_constant_across_q():
    """The defining property of the split: q changes the estimate (rmse_m) but
    not which frames associate (n_assoc). Guards against a future change that
    lets q leak into the discrete association outcome."""
    from evaluation.tracker_stress_q_sweep import run_one

    low = run_one(SCENARIO, n_frames=40, process_noise_q=0.05)
    high = run_one(SCENARIO, n_frames=40, process_noise_q=10.0)
    assert low["n_assoc"] == high["n_assoc"]
    # ... while the RMSE does differ (q is not inert on the estimate).
    assert low["rmse_m"] != high["rmse_m"]
