#!/usr/bin/env python3
"""Streaming consumer: range-Doppler FFT -> CFAR -> cluster -> track -> view,
with overload detection.

See docs/superpowers/specs/2026-05-15-streaming-rd-pipeline-design.md.
"""

from __future__ import annotations

from typing import Optional


def detect_gap(frame_id: int, last_frame_id: Optional[int]) -> int:
    """Return the number of frames dropped between last_frame_id and frame_id.

    Returns 0 if last_frame_id is None (first frame) or if frame_id is not
    greater than last_frame_id + 1 (contiguous or out-of-order).
    """
    if last_frame_id is None:
        return 0
    return max(0, frame_id - last_frame_id - 1)


def compute_lag(
    frame_id: int,
    frame_id_first: int,
    t_first: float,
    now: float,
    frame_period_s: float,
) -> float:
    """Cumulative lag in seconds: actual elapsed wall time minus expected."""
    expected_dt = (frame_id - frame_id_first) * frame_period_s
    actual_dt = now - t_first
    return actual_dt - expected_dt


def format_summary(
    frame_idx: int,
    stage_means_ms: tuple,         # (rd, cfar, cluster, track)
    viz_mean_ms: float,
    viz_max_ms: float,
    viz_skipped: int,
    stats_window: int,
    over_stream: int,
    over_realtime: int,
    drops: int,
    lag_s: float,
) -> str:
    """One-line periodic summary string. Format documented in the design spec."""
    rd, cfar, cluster, track = stage_means_ms
    return (
        f"[frame {frame_idx}] "
        f"rd={rd:.1f} cfar={cfar:.1f} cluster={cluster:.1f} track={track:.1f} (mean ms) "
        f"| viz mean={viz_mean_ms:.1f} max={viz_max_ms:.1f} skipped={viz_skipped}/{stats_window} "
        f"| over_stream={over_stream}/{stats_window} over_realtime={over_realtime}/{stats_window} "
        f"| drops={drops} lag={lag_s:+.2f}s"
    )
