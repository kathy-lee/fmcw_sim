#!/usr/bin/env python3
"""Streaming consumer: range-Doppler FFT -> CFAR -> cluster -> track -> view,
with overload detection.

See docs/superpowers/specs/2026-05-15-streaming-rd-pipeline-design.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import CfarConfig, ClusteringConfig, TrackerConfig
from processing.output import open_csv_writers
from processing.range_doppler import range_doppler
from processing.tracking import Tracker
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--connect", default="tcp://127.0.0.1:5555")
    p.add_argument("--topic", default="fmcw")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 means run forever")
    p.add_argument("--window", choices=["rect", "hann"], default="hann")
    p.add_argument("--no-dc-removal", dest="dc_removal",
                   action="store_false", default=True,
                   help="Disable per-chirp DC removal (default: enabled)")
    p.add_argument("--no-clutter-removal", dest="clutter_removal",
                   action="store_false", default=True,
                   help="Disable slow-time mean subtraction (default: enabled)")
    p.add_argument("--save-dir", default=None,
                   help="Optional directory for .npy RD power maps")
    p.add_argument("--stats-window", type=int, default=50,
                   help="Frames between summary prints")
    return p.parse_args()


def _cfar_kwargs(cfg: CfarConfig) -> dict:
    return dict(
        method=cfg.method,
        guard=cfg.guard,
        training=cfg.training,
        pfa=cfg.pfa,
        rank_frac=cfg.rank_frac,
    )


def _cluster_kwargs(cfg: ClusteringConfig) -> dict:
    return dict(
        method=cfg.method,
        min_cells=cfg.min_cells,
        connectivity=cfg.connectivity,
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        axis_weights=cfg.axis_weights,
    )


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    print(f"Receiving from {args.connect}, topic={args.topic!r}")

    cfar_cfg = CfarConfig()
    cluster_cfg = ClusteringConfig()
    tracker = Tracker(TrackerConfig())
    cfar_kw = _cfar_kwargs(cfar_cfg)
    cluster_kw = _cluster_kwargs(cluster_cfg)

    view: Optional[LayeredRDView] = None
    det_writer = None
    track_writer = None
    stream_deadline_s = 0.0
    realtime_deadline_s = 0.0
    last_frame_id: Optional[int] = None
    t_first = 0.0
    frame_id_first = 0

    WINDOW = args.stats_window
    stage_times = np.zeros((WINDOW, 5), dtype=np.float64)
    idx = 0
    drops = 0
    viz_skipped = 0
    over_stream_count = 0
    over_realtime_count = 0

    try:
        while args.max_frames == 0 or idx < args.max_frames:
            topic, header_raw, payload = sock.recv_multipart()
            try:
                header = json.loads(header_raw.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"WARN: bad header JSON, skipping frame: {e}")
                continue

            if header.get("dtype") != "complex64":
                raise ValueError(f"Unsupported dtype: {header.get('dtype')}")

            if view is None:
                view, stream_deadline_s, realtime_deadline_s = init_view(header, cfar_cfg)
                view.on_reset(tracker.reset)
                view.show()
                meta = header["metadata"]; cfg = meta["config"]
                det_writer, track_writer = open_csv_writers(
                    save_dir,
                    range_res_m=C_EM / (2.0 * float(cfg["bandwidth_hz"])),
                    chirp_period_s=float(cfg["chirp_period_s"]),
                    lambda_m=float(meta["derived"]["lambda_m"]),
                    n_doppler=view.n_doppler,
                )
                t_first = time.perf_counter()
                frame_id_first = int(header["frame_id"])
                print(
                    f"[init] frame_period_s = {1e3*stream_deadline_s:.3f} ms (from header)\n"
                    f"       n_chirps x chirp_period_s = {1e3*realtime_deadline_s:.3f} ms (natural radar cadence)\n"
                    f"       over_stream deadline = {1e3*stream_deadline_s:.3f} ms, "
                    f"over_realtime deadline = {1e3*realtime_deadline_s:.3f} ms"
                )

            fid = int(header["frame_id"])
            gap = detect_gap(fid, last_frame_id)
            if gap > 0:
                drops += gap
                print(f"WARN frame {fid}: gap of {gap} (last {last_frame_id})")
            last_frame_id = fid

            shape = tuple(header["shape"])
            try:
                frame = np.frombuffer(payload, dtype=np.complex64).reshape(shape)
            except ValueError as e:
                print(f"WARN frame {fid}: shape mismatch, skipping ({e})")
                continue

            t0 = time.perf_counter()
            rd = range_doppler(
                frame,
                window=args.window,
                dc_removal=args.dc_removal,
                clutter_removal=args.clutter_removal,
            )
            power_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
            t1 = time.perf_counter()
            mask = cfar_2d(power_db, window=args.window,
                           dc_removal=args.dc_removal,
                           clutter_removal=args.clutter_removal, **cfar_kw)
            mask[:, 0] = False   # range-0 notch: suppress TX→RX leakage column
            t2 = time.perf_counter()
            detections = cluster(mask, power_db, **cluster_kw)
            t3 = time.perf_counter()
            tracks = tracker.step(detections)
            t4 = time.perf_counter()

            has_backlog = sock.poll(0, zmq.POLLIN) != 0
            if not has_backlog:
                view.update(power_db, mask, detections, tracks, frame_id=fid)
                t5 = time.perf_counter()
            else:
                t5 = t4
                viz_skipped += 1

            proc_time = t4 - t0
            total_time = t5 - t0
            if total_time > stream_deadline_s:
                over_stream_count += 1
            if proc_time > realtime_deadline_s:
                over_realtime_count += 1

            stage_times[idx % WINDOW] = (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
            idx += 1

            if save_dir is not None:
                # np.save(save_dir / f"rd_power_db_{fid:06d}.npy", power_db)
                det_writer.write(fid, detections)
                track_writer.write(fid, tracks)

            if idx % WINDOW == 0:
                rd_ms = stage_times[:, 0].mean() * 1e3
                cfar_ms = stage_times[:, 1].mean() * 1e3
                cluster_ms = stage_times[:, 2].mean() * 1e3
                track_ms = stage_times[:, 3].mean() * 1e3
                viz_col = stage_times[:, 4]
                rendered = viz_col[viz_col > 0]
                viz_mean = (rendered.mean() * 1e3) if rendered.size else 0.0
                viz_max = (viz_col.max() * 1e3)
                lag = compute_lag(
                    frame_id=fid,
                    frame_id_first=frame_id_first,
                    t_first=t_first,
                    now=time.perf_counter(),
                    frame_period_s=stream_deadline_s,
                )
                print(format_summary(
                    frame_idx=idx,
                    stage_means_ms=(rd_ms, cfar_ms, cluster_ms, track_ms),
                    viz_mean_ms=viz_mean, viz_max_ms=viz_max,
                    viz_skipped=viz_skipped, stats_window=WINDOW,
                    over_stream=over_stream_count,
                    over_realtime=over_realtime_count,
                    drops=drops, lag_s=lag,
                ))
                viz_skipped = 0
                over_stream_count = 0
                over_realtime_count = 0

    except KeyboardInterrupt:
        print("\nSubscriber stopped.")
    finally:
        if save_dir is not None and det_writer is not None:
            det_writer.close()
            track_writer.close()
        sock.close(linger=0)


if __name__ == "__main__":
    main()
