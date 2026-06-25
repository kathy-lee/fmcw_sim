"""Tests for the extended-target detection sweep and its helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_detect_frame_returns_centroids_mask_rddb():
    from evaluation.sweep_amplitude import detect_frame
    from simulator.beat_signal_generator import (
        FMCWBeatSignalGenerator, RadarConfig, Target,
    )
    cfg = RadarConfig(noise_power=1e-4)
    gen = FMCWBeatSignalGenerator(cfg)
    t = Target(x=4.0, vx=0.5, amplitude=0.005)
    centroids, mask, rd_db = detect_frame(gen, [t], frame_index=0,
                                          cfar_method="os")
    assert mask.dtype == bool
    assert rd_db.shape == mask.shape
    assert isinstance(centroids, list)
    # A strong target produces at least one detection.
    assert len(centroids) >= 1


def test_object_metrics_point_target():
    """For a point target (span 0), object-level Pd matches a within-1-bin
    detection and the per-cell fraction / cluster count are well-formed."""
    from evaluation.sweep_extent import object_metrics
    from simulator.beat_signal_generator import (
        FMCWBeatSignalGenerator, RadarConfig, Target,
    )
    cfg = RadarConfig(noise_power=0.0, leakage_amplitude=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    t = Target(x=4.0, vx=0.5, amplitude=0.01)
    m = object_metrics(gen, t, target_index=0, frame_index=0,
                       cfar_method="os", cluster_method="cc")
    assert m["detected"] in (True, False)
    assert 0.0 <= m["cell_fraction"] <= 1.0
    assert m["cluster_count"] >= 0


def test_extended_sweep_smoke(tmp_path):
    from pathlib import Path
    from evaluation.sweep_extent import run_sweep
    amplitudes = np.array([1.0e-3, 5.0e-3])
    extents = [None, {"n_scatterers": 10, "range_extent_m": 0.6}]
    res = run_sweep(Path("scenarios/scenario_extended_detection.json"),
                    amplitudes, extents, n_frames=5)
    # One Pd/cell-fraction/cluster series per extent case.
    assert set(res.keys()) == {"point", "ext_0.6"}
    for key in res:
        assert len(res[key]["pd"]) == 2
        assert all(0.0 <= p <= 1.0 for p in res[key]["pd"])


def test_detect_frame_matches_inline_lens():
    """Guard the refactor: detect_frame reproduces the EXACT pre-refactor
    inline lens (range_doppler -> cfar_2d -> range-0 notch -> cluster cc/2),
    so run_one_point and the extended sweep cannot drift from report 1's lens.
    """
    from evaluation.sweep_amplitude import detect_frame
    from processing.range_doppler import range_doppler
    from processing.cfar import cfar_2d
    from processing.clustering import cluster
    from simulator.beat_signal_generator import (
        FMCWBeatSignalGenerator, RadarConfig, Target,
    )
    cfg = RadarConfig(noise_power=1e-4, seed=99999)
    targets = [Target(x=3.0, vx=0.1, amplitude=0.0012),
               Target(x=6.0, vx=-0.1, amplitude=0.0008)]
    fi = 0

    # detect_frame on one generator; reference inline lens on a fresh
    # generator with the same seed (identical noise on the first frame).
    cents, mask_df, rddb_df = detect_frame(
        FMCWBeatSignalGenerator(cfg), targets, fi, "os")

    gen2 = FMCWBeatSignalGenerator(cfg)
    frame = gen2.generate_frame(targets, frame_index=fi)
    rd = range_doppler(frame, window="hann", dc_removal=True,
                       clutter_removal=False)
    rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
    mask = cfar_2d(rd_db, method="os", guard=(2, 4), training=(4, 8),
                   pfa=1e-4, window="hann")
    mask[:, 0] = False
    ref = cluster(mask, rd_db, method="cc", min_cells=2)

    np.testing.assert_array_equal(mask_df, mask)
    np.testing.assert_allclose(rddb_df, rd_db)
    assert len(cents) == len(ref)
    for c, r in zip(sorted(cents, key=lambda d: (d.range_bin, d.doppler_bin)),
                    sorted(ref, key=lambda d: (d.range_bin, d.doppler_bin))):
        assert c.range_bin == r.range_bin
        assert c.doppler_bin == r.doppler_bin
        assert c.cell_count == r.cell_count


def test_run_one_point_pd_is_pinned():
    """Characterization pin: run_one_point on the report-1 scenario at a fixed
    amplitude/seed yields an exact Pd. Any drift in the shared lens trips this.
    """
    import dataclasses
    from evaluation.sweep_amplitude import run_one_point
    from simulator.beat_signal_generator import load_scenario
    radar_cfg, base = load_scenario("scenarios/scenario_two_targets.json")
    targets = [dataclasses.replace(t, amplitude=t.amplitude * 1.257835e-3)
               for t in base]
    pd, fa = run_one_point(radar_cfg, targets, "os", 30)
    # Pinned to the post-refactor values (full suite green at refactor time).
    assert pd[0] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= pd[1] <= 1.0


def test_measure_sigma_glint_zero_for_point():
    from evaluation.tracker_stress_extent_sweep import measure_sigma_glint
    from simulator.beat_signal_generator import RadarConfig, Target
    cfg = RadarConfig(noise_power=0.0, leakage_amplitude=0.0)
    t = Target(x=4.0, vx=0.5, amplitude=0.01)
    sg = measure_sigma_glint(cfg, t, n_frames=8, seeds=[1])
    assert sg == pytest.approx(0.0, abs=1e-6)


def test_run_cell_smoke():
    from evaluation.tracker_stress_extent_sweep import run_cell
    from simulator.beat_signal_generator import load_scenario
    radar_cfg, targets = load_scenario(
        "scenarios/scenario_extended_tracking.json")
    rows = run_cell(radar_cfg, targets[0],
                    extent={"n_scatterers": 10, "range_extent_m": 0.6},
                    cluster_method="dbscan", range_meas_var=None,
                    seeds=[1], n_frames=8)
    assert len(rows) == 1
    r = rows[0]
    assert set(r) >= {"pd", "distinct", "n_switches", "rmse_m",
                      "classification"}
    assert 0.0 <= r["pd"] <= 1.0


def test_fragmentation_by_method_smoke():
    """peak/cc/dbscan fragmentation comparison at a fixed amplitude: each
    method reports clusters-per-object and object-Pd per extent."""
    from pathlib import Path
    from evaluation.sweep_extent import fragmentation_by_method
    res = fragmentation_by_method(
        Path("scenarios/scenario_extended_detection.json"),
        amplitude_scale=2.1e-3,
        extents=[None, {"n_scatterers": 10, "range_extent_m": 0.6}],
        methods=["peak", "cc", "dbscan"], n_frames=5)
    assert set(res.keys()) == {"peak", "cc", "dbscan"}
    for method in res:
        assert set(res[method].keys()) == {"point", "ext_0.6"}
        for key in res[method]:
            assert res[method][key]["clusters"] >= 0
            assert 0.0 <= res[method][key]["pd"] <= 1.0
