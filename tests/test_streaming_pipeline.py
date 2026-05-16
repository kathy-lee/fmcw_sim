import matplotlib
matplotlib.use("Agg")

import pytest

from simulator.streaming_pipeline import detect_gap, compute_lag, format_summary


def test_detect_gap_returns_zero_for_first_frame():
    assert detect_gap(frame_id=0, last_frame_id=None) == 0


def test_detect_gap_returns_zero_for_contiguous():
    assert detect_gap(frame_id=5, last_frame_id=4) == 0


def test_detect_gap_counts_missing_frames():
    assert detect_gap(frame_id=10, last_frame_id=5) == 4


def test_detect_gap_handles_out_of_order_returns_zero_or_negative():
    # Defensive: if frame_id decreases (publisher restart), don't blow up.
    assert detect_gap(frame_id=3, last_frame_id=10) <= 0


def test_compute_lag_zero_when_on_schedule():
    lag = compute_lag(
        frame_id=10, frame_id_first=0,
        t_first=0.0, now=0.5,
        frame_period_s=0.05,
    )
    assert abs(lag) < 1e-9


def test_compute_lag_positive_when_late():
    lag = compute_lag(
        frame_id=10, frame_id_first=0,
        t_first=0.0, now=0.7,
        frame_period_s=0.05,
    )
    assert abs(lag - 0.2) < 1e-9


def test_compute_lag_negative_when_early():
    lag = compute_lag(
        frame_id=10, frame_id_first=0,
        t_first=0.0, now=0.3,
        frame_period_s=0.05,
    )
    assert abs(lag + 0.2) < 1e-9


def test_format_summary_contains_all_keys():
    line = format_summary(
        frame_idx=250,
        stage_means_ms=(1.4, 3.2, 0.3, 0.1),  # rd, cfar, cluster, track
        viz_mean_ms=20.1,
        viz_max_ms=80.3,
        viz_skipped=8,
        stats_window=50,
        over_stream=2,
        over_realtime=47,
        drops=2,
        lag_s=0.32,
    )
    assert "[frame 250]" in line
    assert "rd=1.4" in line
    assert "cfar=3.2" in line
    assert "cluster=0.3" in line
    assert "track=0.1" in line
    assert "viz mean=20.1" in line
    assert "viz max" not in line or "max=80.3" in line  # max present in some form
    assert "max=80.3" in line
    assert "skipped=8/50" in line
    assert "over_stream=2/50" in line
    assert "over_realtime=47/50" in line
    assert "drops=2" in line
    assert "lag=+0.32s" in line


def test_format_summary_signs_negative_lag():
    line = format_summary(
        frame_idx=50, stage_means_ms=(1.0, 1.0, 1.0, 1.0),
        viz_mean_ms=10.0, viz_max_ms=20.0,
        viz_skipped=0, stats_window=50,
        over_stream=0, over_realtime=0,
        drops=0, lag_s=-0.05,
    )
    assert "lag=-0.05s" in line
