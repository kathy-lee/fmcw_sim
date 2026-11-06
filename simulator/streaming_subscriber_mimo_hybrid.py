#!/usr/bin/env python3
"""Live MIMO tracking subscriber with a switchable front end.

Same pipeline and output as streaming_subscriber_mimo (3-panel live view,
per-stage real-time summary): the front end is chosen at startup and
everything downstream — Cartesian clustering, CTRV-EKF tracking, drawing —
is shared, so flipping the flag between two runs against the same seeded
publisher is a live A/B of the two implementations.

  --frontend python   NumPy reference chain (process_mimo_frame)
  --frontend c        the C library via processing.c_frontend.CFrontend,
                      called per stage so the summary line keeps its true
                      per-stage breakdown; requires --params <params.cfg>
                      exported for the same scenario the publisher streams
                      (scripts/export_for_c.py), and the library built
                      (make -C c_frontend lib)

DoA is Bartlett in both modes (the C library implements no other method,
and an asymmetric A/B would be meaningless).

Usage:
    python -m simulator.streaming_subscriber_mimo_hybrid --frontend c \
        --params data/c_export_bench/params.cfg
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from processing.clustering import cluster_point_cloud
from processing.mimo_frame import frame_params, process_mimo_frame
from processing.tracking_cartesian import CartesianTracker, CartesianTrackerConfig
from simulator.streaming_subscriber import detect_gap
from simulator.streaming_subscriber_mimo import (
    _draw, _prewarm_cfar, deadlines, format_summary,
)
from simulator.wire import deserialize_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--connect", default="tcp://127.0.0.1:5555")
    p.add_argument("--topic", default="fmcw")
    p.add_argument("--max-frames", type=int, default=0, help="0 = run forever")
    p.add_argument("--frontend", choices=["python", "c"], default="c")
    p.add_argument("--params", default=None,
                   help="params.cfg for the C front end (required with "
                        "--frontend c)")
    p.add_argument("--stats-window", type=int, default=20,
                   help="frames between real-time summary prints")
    args = p.parse_args()
    if args.frontend == "c" and not args.params:
        p.error("--frontend c requires --params <params.cfg>")
    return args


def _python_out(frame, header):
    """Reference front end + clustering, exactly the sibling's block."""
    return process_mimo_frame(frame, header, doa_method="bartlett")


def _c_out(fe, frame, header, p):
    """C front end per stage (timed like the reference timings dict), then
    the shared Python clustering. Returns the out-dict shape _draw and the
    summary line consume."""
    t0 = time.perf_counter()
    rd = fe.rd(frame)
    t1 = time.perf_counter()
    power = fe.nci(rd)
    t2 = time.perf_counter()
    mask = fe.cfar(power)
    t3 = time.perf_counter()
    rows = fe.pointcloud(rd, mask)
    t4 = time.perf_counter()
    cloud = rows[:, 3:7] if rows.size else np.zeros((0, 4))
    objects = cluster_point_cloud(cloud, eps_m=0.8, min_samples=2,
                                  vel_scale=1.0)
    t5 = time.perf_counter()
    return {
        "rd": rd,
        "power_db": 10.0 * np.log10(power + 1e-12),
        "mask": mask,
        "cloud": cloud,
        "objects": objects,
        "range_axis": p["range_axis"],
        "vel_axis": p["vel_axis"],
        "pos": p["pos"],
        "n_tx": p["n_tx"],
        "n_rx": p["n_rx"],
        "timings": {"rd": t1 - t0, "integrate": t2 - t1, "cfar": t3 - t2,
                    "doa_pc": t4 - t3, "cluster": t5 - t4, "total": t5 - t0},
    }


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import zmq

    fe = None
    if args.frontend == "c":
        # Imported lazily so --frontend python never needs the library built.
        from processing.c_frontend import CFrontend
        fe = CFrontend(args.params)
        print(f"C front end: {args.params}, dims={fe.dims}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    print(f"Receiving from {args.connect}, topic={args.topic!r}, "
          f"frontend={args.frontend}")

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    W = args.stats_window
    # columns: rd, integ, cfar, doa+pc, cluster, track, total
    stage_times = np.zeros((W, 7), dtype=np.float64)
    idx = 0
    drops = 0
    over_stream = over_realtime = 0
    last_frame_id = None
    stream_dl = acq_dl = 0.0
    ready = False
    tracker = None
    params = None

    try:
        while args.max_frames == 0 or idx < args.max_frames:
            if not sock.poll(1000, zmq.POLLIN):
                plt.pause(0.001)                     # stay responsive while idle
                continue
            topic, header_raw, payload = sock.recv_multipart()
            header = json.loads(header_raw.decode("utf-8"))
            if len(header["shape"]) != 3:
                print(f"skip non-MIMO frame shape={header['shape']}")
                continue

            if not ready:
                stream_dl, acq_dl = deadlines(header)
                params = frame_params(header)
                if fe is not None:
                    if list(header["shape"]) != list(fe.dims):
                        raise SystemExit(
                            f"stream shape {header['shape']} does not match "
                            f"params.cfg dims {list(fe.dims)} — re-export "
                            f"params for this scenario")
                else:
                    print("Calibrating CFAR (one-time)...", flush=True)
                    _prewarm_cfar(header, "ca")
                # Predict timestep = frame period, so the tracker's state
                # velocity is m/s, consistent with the Doppler measurement.
                tracker = CartesianTracker(CartesianTrackerConfig(dt=stream_dl))
                ready = True

            fid = int(header["frame_id"])
            drops += detect_gap(fid, last_frame_id)
            last_frame_id = fid

            frame = deserialize_frame(header, payload)
            if fe is not None:
                out = _c_out(fe, frame, header, params)
            else:
                out = _python_out(frame, header)
            t_track0 = time.perf_counter()
            tracks = tracker.step(out["objects"])
            track_s = time.perf_counter() - t_track0
            _draw(axes, out, header, "bartlett", tracks)

            t = out["timings"]
            total_s = t["total"] + track_s
            stage_times[idx % W] = (t["rd"], t["integrate"], t["cfar"],
                                    t["doa_pc"], t["cluster"], track_s, total_s)
            if total_s > stream_dl:
                over_stream += 1
            if total_s > acq_dl:
                over_realtime += 1
            idx += 1

            if idx % W == 0:
                means_ms = tuple(stage_times[:, k].mean() * 1e3 for k in range(6))
                total_ms = stage_times[:, 6].mean() * 1e3
                print(format_summary(
                    idx, means_ms, total_ms, stream_dl * 1e3, acq_dl * 1e3,
                    over_stream, over_realtime, W, drops))
                over_stream = over_realtime = 0
    except KeyboardInterrupt:
        print("\nSubscriber stopped.")
    finally:
        sock.close(linger=0)
        if fe is not None:
            fe.close()


if __name__ == "__main__":
    main()
