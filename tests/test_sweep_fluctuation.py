"""Tests for the fluctuation Pd sweep and its run_one_point refactor."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def test_run_one_point_advance_frame_index_tracks_moving_target():
    """With advance_frame_index=True the truth bins are recomputed per
    frame, so a strong moving target stays matched while it drifts
    (1 m/s * 0.05 s/frame = 0.67 range bins/frame -- it leaves its
    initial bin within 2 frames, so passing proves per-frame tracking)."""
    from evaluation.sweep_amplitude import run_one_point
    from simulator.beat_signal_generator import load_scenario

    radar_cfg, base_targets = load_scenario(
        "scenarios/scenario_two_targets.json")
    targets = [dataclasses.replace(base_targets[0], velocity_mps=1.0)]
    pd_moving, _ = run_one_point(radar_cfg, targets, "os", 10,
                                 advance_frame_index=True)
    assert pd_moving[0] > 0.9


def test_zero_velocity_cancellation_is_clutter_removal_not_dc_removal():
    """Documents which knob cancels a zero-velocity target.

    range_doppler has two mean-subtraction options that are easy to
    conflate: `dc_removal` subtracts the per-chirp FAST-TIME mean (kills
    range bin 0 only), while `clutter_removal` subtracts the SLOW-TIME
    mean across chirps -- which removes a zero-Doppler (v=0) echo
    exactly. The evaluation pipeline (sweep_amplitude / tracker_stress)
    runs with clutter_removal=False, so a strong v=0 target IS detected
    there. The fluctuation sweep reuses the baseline moving-target
    scenario anyway, for direct comparability with detection report §3.
    """
    from evaluation.sweep_amplitude import run_one_point
    from processing.range_doppler import range_doppler
    from simulator.beat_signal_generator import (
        FMCWBeatSignalGenerator, load_scenario,
    )

    radar_cfg, base_targets = load_scenario(
        "scenarios/scenario_two_targets.json")
    targets = [dataclasses.replace(base_targets[0], velocity_mps=0.0)]

    # Through the actual pipeline (clutter_removal=False): detected.
    pd_zero, _ = run_one_point(radar_cfg, targets, "os", 10,
                               advance_frame_index=True)
    assert pd_zero[0] > 0.9

    # With clutter_removal=True the v=0 echo is removed exactly: the RD
    # map of a noise-free frame collapses to (numerically) nothing.
    cfg = dataclasses.replace(radar_cfg, noise_power=0.0,
                              leakage_amplitude=0.0)
    frame = FMCWBeatSignalGenerator(cfg).generate_frame(targets,
                                                        frame_index=0)
    rd_kept = range_doppler(frame, window="hann", dc_removal=True,
                            clutter_removal=False)
    rd_cancelled = range_doppler(frame, window="hann", dc_removal=True,
                                 clutter_removal=True)
    peak_kept = np.abs(rd_kept).max()
    peak_cancelled = np.abs(rd_cancelled).max()
    assert peak_cancelled < 1e-6 * peak_kept


def test_snr_at_pd_interpolates():
    import math
    from evaluation.sweep_amplitude_fluctuation import snr_at_pd
    pts = [(-30.0, 0.0), (-25.0, 0.5), (-20.0, 1.0)]
    assert snr_at_pd(pts, 0.5) == pytest.approx(-25.0)
    assert snr_at_pd(pts, 0.75) == pytest.approx(-22.5)
    assert math.isnan(snr_at_pd([(-30.0, 0.0), (-25.0, 0.3)], 0.9))


def test_fluctuation_sweep_smoke():
    from pathlib import Path
    from evaluation.sweep_amplitude_fluctuation import run_sweep
    amplitudes = np.array([3.5e-4, 1.0e-3])
    pd_results, fa_results, base_targets = run_sweep(
        Path("scenarios/scenario_two_targets.json"),
        amplitudes, n_frames=5)
    assert len(base_targets) == 2
    for case in ("steady", "swerling1", "swerling3"):
        assert len(fa_results[case]) == 2
        for ti in range(2):
            pts = pd_results[(case, ti)]
            assert len(pts) == 2
            assert all(0.0 <= pd <= 1.0 for _, pd in pts)


def test_targets_with_corr_steady_strips_fluctuation():
    from evaluation.tracker_stress_swerling_sweep import targets_with_corr
    from simulator.beat_signal_generator import Target
    base = [Target(range_m=4.0, velocity_mps=0.5, amplitude=0.0009,
                   fluctuation={"model": "swerling1", "frame_corr": 0.9})]
    steady = targets_with_corr(base, None)
    assert steady[0].fluctuation is None
    corr = targets_with_corr(base, 0.6)
    assert corr[0].fluctuation == {"model": "swerling1", "frame_corr": 0.6}
    # base untouched
    assert base[0].fluctuation == {"model": "swerling1", "frame_corr": 0.9}


def test_swerling_sweep_run_cell_smoke():
    from evaluation.tracker_stress_swerling_sweep import run_cell
    from simulator.beat_signal_generator import load_scenario
    radar_cfg, targets = load_scenario(
        "scenarios/scenario_missed_detections.json")
    rows = run_cell(radar_cfg, targets, frame_corr=0.9, max_coast=3,
                    seeds=[1], n_frames=8)
    assert len(rows) == 1
    r = rows[0]
    assert set(r) == {"frame_corr", "max_coast", "seed", "pd", "distinct",
                      "coast", "n_switches", "classification"}
    assert 0.0 <= r["pd"] <= 1.0
    assert r["frame_corr"] == 0.9
    assert r["max_coast"] == 3
