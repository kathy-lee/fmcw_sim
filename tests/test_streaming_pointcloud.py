"""M2c: live point-cloud subscriber — smoke tests for the processing core.

The live matplotlib loop is validated by running it against the publisher; here we
test the per-frame processing (process_mimo_frame / frame_params) headlessly.
"""
import numpy as np

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.streaming_subscriber_pointcloud import (
    deadlines, format_summary, frame_params, process_mimo_frame)
from simulator.wire import build_header


def _frame_and_header(scenario="scenarios/scenario_mimo.json", frame_index=0):
    cfg, targets = load_scenario(scenario)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame_mimo(targets, frame_index=frame_index)
    header = build_header(
        frame=frame, frame_id=frame_index, topic="fmcw",
        metadata=gen.metadata(),
        ground_truth=gen.ground_truth(targets, frame_index=frame_index),
        timestamp_unix_s=0.0)
    return cfg, frame, header


def test_frame_params_from_header():
    cfg, _, header = _frame_and_header()
    p = frame_params(header)
    assert p["n_tx"] == cfg.n_tx and p["n_rx"] == cfg.n_rx
    assert p["pos"].shape == (cfg.n_tx * cfg.n_rx,)
    assert p["range_axis"].shape == (cfg.n_range // 2,)
    assert p["vel_axis"].shape == (cfg.n_chirps,)


def _near(objects, x, y, r=0.7):
    """Count clustered objects within radius r (m) of (x, y)."""
    return sum(np.hypot(o.x - x, o.y - y) < r for o in objects)


def _gt_xy(header):
    out = []
    for gt in header["ground_truth"]:
        az = np.deg2rad(gt["azimuth_deg"])
        out.append((gt["range_m"] * np.cos(az), gt["range_m"] * np.sin(az)))
    return out


def test_process_mimo_frame_produces_cloud_and_objects():
    cfg, frame, header = _frame_and_header()
    out = process_mimo_frame(frame, header, doa_method="music")
    assert out["power_db"].shape == (cfg.n_chirps, cfg.n_range // 2)
    assert out["cloud"].ndim == 2 and out["cloud"].shape[1] == 4
    # timings present for the real-time report (incl. the cluster stage)
    assert set(out["timings"]) == {
        "rd", "integrate", "cfar", "doa_pc", "cluster", "total"}
    assert out["timings"]["total"] > 0.0
    # the single target (8, 2) collapses to EXACTLY one object near it. (Far
    # objects may exist from near-DC false alarms — a detection-Pfa issue, not
    # clustering; the tracker rejects transient false tracks.)
    assert _near(out["objects"], 8.0, 2.0) == 1


def test_process_mimo_frame_two_targets_found():
    _, frame, header = _frame_and_header("scenarios/scenario_mimo_2targets.json")
    out = process_mimo_frame(frame, header, doa_method="music")
    # exactly one object near each GT target — a merged blob would sit near
    # neither, so this also verifies separation.
    for gx, gy in _gt_xy(header):
        assert _near(out["objects"], gx, gy) == 1


def test_process_mimo_frame_crossing_velocity_separates():
    # at the crossing frame (~40) both targets share ~(4, 0) but have opposite
    # radial velocity -> only the velocity feature separates them.
    _, frame, header = _frame_and_header(
        "scenarios/scenario_mimo_crossing.json", frame_index=40)
    sep = process_mimo_frame(frame, header, doa_method="music", vel_scale=1.0)
    merged = process_mimo_frame(frame, header, doa_method="music", vel_scale=0.0)
    assert _near(sep["objects"], 4.0, 0.0, r=1.0) == 2     # velocity splits
    assert _near(merged["objects"], 4.0, 0.0, r=1.0) == 1  # spatial-only merges


def test_deadlines_tdm_acquisition_scales_with_n_tx():
    _, _, header = _frame_and_header()
    stream, acq = deadlines(header)
    cfg = header["metadata"]["config"]
    assert stream == cfg["frame_period_s"]
    assert np.isclose(
        acq, cfg["n_tx"] * cfg["n_chirps"] * cfg["chirp_period_s"])


def test_format_summary_reports_realtime_verdict():
    line = format_summary(
        frame_idx=20, stage_means_ms=(1.0, 0.2, 3.0, 8.0, 0.5), total_ms=12.7,
        period_ms=50.0, acq_ms=9.6, over_stream=0, over_realtime=5,
        window=20, drops=2)
    assert "[frame 20]" in line
    assert "doa+pc=8.0" in line
    assert "cluster=0.5" in line
    assert "total=12.7" in line
    assert "REALTIME=NO" in line          # 12.7 ms total > 9.6 ms acquisition
    assert "drops=2" in line
