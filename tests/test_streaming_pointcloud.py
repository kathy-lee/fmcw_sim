"""M2c: live point-cloud subscriber — smoke tests for the processing core.

The live matplotlib loop is validated by running it against the publisher; here we
test the per-frame processing (process_mimo_frame / frame_params) headlessly.
"""
import numpy as np

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.streaming_subscriber_pointcloud import (
    deadlines, format_summary, frame_params, process_mimo_frame)
from simulator.wire import build_header


def _frame_and_header(frame_index=0):
    cfg, targets = load_scenario("scenarios/scenario_mimo_2d.json")
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


def test_process_mimo_frame_produces_point_cloud():
    cfg, frame, header = _frame_and_header()
    out = process_mimo_frame(frame, header, doa_method="music")
    assert out["power_db"].shape == (cfg.n_chirps, cfg.n_range // 2)
    assert out["mask"].shape == out["power_db"].shape
    assert out["cloud"].ndim == 2 and out["cloud"].shape[1] == 4
    # timings present for the real-time report
    assert set(out["timings"]) == {"rd", "integrate", "cfar", "doa_pc", "total"}
    assert out["timings"]["total"] > 0.0
    # the off-boresight target (8, 2) should appear somewhere in the cloud
    if out["cloud"].shape[0]:
        d = np.hypot(out["cloud"][:, 0] - 8.0, out["cloud"][:, 1] - 2.0)
        assert d.min() < 0.5


def test_deadlines_tdm_acquisition_scales_with_n_tx():
    _, _, header = _frame_and_header()
    stream, acq = deadlines(header)
    cfg = header["metadata"]["config"]
    assert stream == cfg["frame_period_s"]
    assert np.isclose(
        acq, cfg["n_tx"] * cfg["n_chirps"] * cfg["chirp_period_s"])


def test_format_summary_reports_realtime_verdict():
    line = format_summary(
        frame_idx=20, stage_means_ms=(1.0, 0.2, 3.0, 8.0), total_ms=12.2,
        period_ms=50.0, acq_ms=9.6, over_stream=0, over_realtime=5,
        window=20, drops=2)
    assert "[frame 20]" in line
    assert "doa+pc=8.0" in line
    assert "total=12.2" in line
    assert "REALTIME=NO" in line          # 12.2 ms total > 9.6 ms acquisition
    assert "drops=2" in line
