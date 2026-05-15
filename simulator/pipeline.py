"""Receive range-Doppler frames over ZMQ, run CFAR → cluster → track, and render.

Usage:
    python receive_with_tracking.py

Hotkeys (focused matplotlib window):
    c — toggle CFAR mask overlay
    k — toggle cluster centroids
    t — toggle track overlay
    r — reset the tracker (drops all tracks)
    q — close the window
"""

from __future__ import annotations

import time

import numpy as np
import zmq

from radar_postproc.config import PipelineConfig
from radar_postproc.cfar import cfar_2d
from radar_postproc.clustering import cluster
from radar_postproc.tracking import Tracker
from radar_postproc.viz import LayeredRDView


def _cfar_kwargs(cfg):
    return dict(
        method=cfg.method,
        guard=cfg.guard,
        training=cfg.training,
        pfa=cfg.pfa,
        rank_frac=cfg.rank_frac,
    )


def _cluster_kwargs(cfg):
    return dict(
        method=cfg.method,
        min_cells=cfg.min_cells,
        connectivity=cfg.connectivity,
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        axis_weights=cfg.axis_weights,
    )


def build_pipeline(cfg: PipelineConfig | None = None):
    """Construct tracker + view from a config; useful for tests and reuse."""
    cfg = cfg or PipelineConfig()
    tracker = Tracker(cfg.tracker)
    view = LayeredRDView(cfg.radar.n_doppler, cfg.radar.n_range, cfg.radar)
    view.on_reset(tracker.reset)
    return cfg, tracker, view


def main(zmq_endpoint: str = "tcp://127.0.0.1:5555") -> None:
    cfg, tracker, view = build_pipeline()
    N = cfg.radar.n_doppler * cfg.radar.n_range

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(zmq_endpoint)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    view.show()

    # Accumulate per-stage timings and print mean / p95 / max every N=WINDOW frames
    WINDOW = 50
    stage_times = np.zeros((WINDOW, 4), dtype=np.float64)  # cfar, cluster, track, viz
    frame_idx = 0

    buffer = np.empty(0, dtype=np.float32)
    while True:
        msg = sock.recv()
        chunk = np.frombuffer(msg, dtype=np.float32)
        buffer = np.concatenate([buffer, chunk])

        while buffer.size >= N:
            frame = buffer[:N]
            buffer = buffer[N:]

            rd = frame.reshape(cfg.radar.n_doppler, cfg.radar.n_range)
            t0 = time.perf_counter()
            mask = cfar_2d(rd, **_cfar_kwargs(cfg.cfar))
            t1 = time.perf_counter()
            detections = cluster(mask, rd, **_cluster_kwargs(cfg.clustering))
            t2 = time.perf_counter()
            tracks = tracker.step(detections)
            t3 = time.perf_counter()
            view.update(rd, mask, detections, tracks)
            t4 = time.perf_counter()

            stage_times[frame_idx % WINDOW] = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
            frame_idx += 1

            # Show timing stat info
            # 'backlog' : Local Python buffer, how many complete frames are sitting in Python memory, fully received but not yet processed.
            #  - backlog=0f — after slicing this frame, less than one full frame remains. The pipeline finished before the next frame arrived. Keeping up.                                                 
            #  - backlog=1f — one full frame has already arrived while we were processing this one. Running right at the edge — that single frame took ≥640 ms total (DSP + viz), so the next one queued up. Not a problem if it stays at 0/1 and oscillates; a problem if it monotonically grows (2, 3, 5, 10 → falling behind, latency accumulating).                                              
            #  - backlog=Nf and growing — pipeline is too slow; you'll eventually OOM or lag visibly behind reality.
            if frame_idx % WINDOW == 0:
                proc = stage_times[:, :3].sum(axis=1)
                viz = stage_times[:, 3]
                backlog = buffer.size // N
                print(
                    f"[{frame_idx}] proc ms mean={1e3*proc.mean():.1f} "
                    f"p95={1e3*np.percentile(proc, 95):.1f} "
                    f"max={1e3*proc.max():.1f} | "
                    f"viz mean={1e3*viz.mean():.1f} max={1e3*viz.max():.1f} | "
                    f"backlog={backlog}f"
                )


if __name__ == "__main__":
    main()
