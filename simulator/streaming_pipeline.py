#!/usr/bin/env python3
"""Streaming consumer: range-Doppler FFT -> CFAR -> cluster -> track -> view,
with overload detection.

See docs/superpowers/specs/2026-05-15-streaming-rd-pipeline-design.md.
"""

from __future__ import annotations

from typing import Optional

from processing.config import CfarConfig
from processing.viz import LayeredRDView

C_EM = 299_792_458.0


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


def init_view(header: dict, cfar_cfg: CfarConfig) -> tuple:
    """First-frame construction.

    Computes radar scalars from the header, validates the CFAR window
    against the actual RD shape, builds the view, and returns
    (view, stream_deadline_s, realtime_deadline_s).
    """
    meta = header["metadata"]
    cfg = meta["config"]

    n_chirps, n_range_raw = header["shape"]
    n_range_post = n_range_raw // 2

    bandwidth_hz = float(cfg["bandwidth_hz"])
    chirp_period_s = float(cfg["chirp_period_s"])
    frame_period_s = float(cfg["frame_period_s"])
    lambda_m = float(meta["derived"]["lambda_m"])

    range_res_m = C_EM / (2.0 * bandwidth_hz)

    G_d, G_r = cfar_cfg.guard
    T_d, T_r = cfar_cfg.training
    win_d = 2 * (G_d + T_d) + 1
    win_r = 2 * (G_r + T_r) + 1
    if win_d > n_chirps:
        raise ValueError(
            f"CFAR Doppler window {win_d} exceeds n_chirps={n_chirps}; "
            f"reduce CfarConfig.guard/training."
        )
    if win_r > n_range_post:
        raise ValueError(
            f"CFAR range window {win_r} exceeds post-FFT n_range={n_range_post}; "
            f"reduce CfarConfig.guard/training or use a larger frame."
        )

    view = LayeredRDView(
        n_doppler=n_chirps,
        n_range=n_range_post,
        range_res_m=range_res_m,
        chirp_period_s=chirp_period_s,
        lambda_m=lambda_m,
    )

    stream_deadline_s = frame_period_s
    realtime_deadline_s = n_chirps * chirp_period_s
    return view, stream_deadline_s, realtime_deadline_s
