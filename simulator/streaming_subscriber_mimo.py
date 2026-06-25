#!/usr/bin/env python3
"""Live MIMO point-cloud + tracking subscriber.

Consumes the MIMO beat-signal frames streamed by streaming_publisher.py and runs
the full detect-then-angle-then-track pipeline per frame (per-channel range-
Doppler -> non-coherent integration -> CFAR -> DoA at detected cells -> Cartesian
point cloud -> DBSCAN clustering -> CTRV-EKF CartesianTracker), live-updating a
3-panel view: integrated RD + CFAR | DoA spectra (4 methods) | BEV point cloud
with confirmed tracks (id-labelled, with trails).

It also reports whether real-time processing is achieved, like
streaming_subscriber.py: per-stage timing (range-Doppler -> tracking) against two
deadlines -- the stream cadence (frame_period_s) and the TDM acquisition time
(n_tx * n_chirps * chirp_period_s).


Usage:  python3 -m simulator.streaming_subscriber_mimo [--doa music]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from processing.cfar import cfar_2d
from processing.doa import estimate_doa, spectrum
from processing.mimo_frame import _CFAR_KW, process_mimo_frame
from processing.point_cloud import tdm_doppler_compensation
from processing.tracking_cartesian import CartesianTracker, CartesianTrackerConfig
from simulator.streaming_subscriber import detect_gap
from simulator.wire import deserialize_frame

_SPECTRAL = ["bartlett", "capon", "music"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--connect", default="tcp://127.0.0.1:5555")
    p.add_argument("--topic", default="fmcw")
    p.add_argument("--max-frames", type=int, default=0, help="0 = run forever")
    p.add_argument("--doa", choices=_SPECTRAL + ["esprit"], default="music")
    p.add_argument("--cfar", choices=["ca", "os"], default="ca",
                   help="CFAR method: ca (fast, real-time) or os (robust, slow)")
    p.add_argument("--stats-window", type=int, default=20,
                   help="frames between real-time summary prints")
    return p.parse_args()


def deadlines(header: dict) -> tuple[float, float]:
    """(stream cadence, TDM acquisition time) in seconds, from the header."""
    cfg = header["metadata"]["config"]
    stream = float(cfg["frame_period_s"])
    acq = int(cfg["n_tx"]) * int(cfg["n_chirps"]) * float(cfg["chirp_period_s"])
    return stream, acq


def format_summary(frame_idx: int, stage_means_ms: tuple, total_ms: float,
                   period_ms: float, acq_ms: float, over_stream: int,
                   over_realtime: int, window: int, drops: int) -> str:
    """One-line real-time summary (range-Doppler -> tracks)."""
    rd, integ, cfar, doa, clust, track = stage_means_ms
    realtime = "yes" if over_realtime == 0 else "NO"
    return (f"[frame {frame_idx}] rd={rd:.1f} integ={integ:.1f} cfar={cfar:.1f} "
            f"doa+pc={doa:.1f} cluster={clust:.1f} track={track:.1f} "
            f"total={total_ms:.1f} (mean ms) | "
            f"period={period_ms:.1f} acq={acq_ms:.1f} ms | "
            f"over_stream={over_stream}/{window} "
            f"over_realtime={over_realtime}/{window} drops={drops} "
            f"| REALTIME={realtime}")


def _prewarm_cfar(header: dict, cfar_method: str = "ca") -> None:
    """Trigger the one-time CFAR alpha calibration (cached) so the timed loop is
    representative and the first frame does not look like a hang."""
    cfg = header["metadata"]["config"]
    shape = (int(cfg["n_chirps"]), int(cfg["n_range"]) // 2)
    n_virtual = int(cfg["n_tx"]) * int(cfg["n_rx"])
    cfar_2d(np.zeros(shape), method=cfar_method, n_integrate=n_virtual,
            **_CFAR_KW)


def _draw(axes, out: dict, header: dict, doa_method: str, tracks: list) -> None:
    import matplotlib.pyplot as plt  # local: only needed for the live view
    ax0, ax1, ax2 = axes
    for a in axes:
        a.clear()
    ra, va = out["range_axis"], out["vel_axis"]
    ax0.imshow(out["power_db"], aspect="auto", origin="lower", cmap="viridis",
               extent=[ra[0], ra[-1], va[0], va[-1]])
    dd, rr = np.where(out["mask"])
    ax0.scatter(ra[rr], va[dd], s=10, facecolors="none", edgecolors="red")
    ax0.set(xlabel="range [m]", ylabel="v [m/s]", title="integrated RD + CFAR")

    power = 10 ** (out["power_db"] / 10.0)
    d, r = np.unravel_index(np.argmax(power), power.shape)
    snap = tdm_doppler_compensation(out["rd"][:, d, r], d,
                                    out["power_db"].shape[0], out["n_tx"],
                                    out["n_rx"])
    for m in _SPECTRAL:
        g, pw = spectrum(snap, out["pos"], m, n_sources=1)
        ax1.plot(g, 10 * np.log10(pw / np.max(pw) + 1e-12), label=m)
    for az in estimate_doa(snap, out["pos"], method="esprit", n_sources=1):
        ax1.axvline(az, color="purple", ls=":")
    for gt in header.get("ground_truth", []):
        ax1.axvline(gt.get("azimuth_deg", 0.0), color="k", ls="--", lw=1)
    ax1.set(xlabel="azimuth [deg]", ylabel="dB", title="DoA spectra", xlim=(-90, 90))
    ax1.legend(fontsize=7)

    cloud = out["cloud"]
    if cloud.shape[0]:
        ax2.scatter(cloud[:, 0], cloud[:, 1], c="0.7", s=8, label="points")
    for obj in out.get("objects", []):
        ax2.scatter([obj.x], [obj.y], marker="o", facecolors="none",
                    edgecolors="C1", s=90, linewidths=2.0, label="object")
        ax2.add_patch(plt.Circle((obj.x, obj.y), max(obj.extent_m, 0.2),
                                 fill=False, color="C1", ls=":", lw=0.8))
    n_confirmed = 0
    for tr in tracks:
        if tr.status != "confirmed":
            continue
        n_confirmed += 1
        x, y = float(tr.state[0]), float(tr.state[1])
        if tr.history is not None and len(tr.history) > 1:
            hx = [p[0] for p in tr.history]
            hy = [p[1] for p in tr.history]
            ax2.plot(hx, hy, "-", color="C3", lw=1.2, alpha=0.7)
        ax2.scatter([x], [y], marker="s", c="C3", s=45, zorder=5)
        ax2.annotate(f"T{tr.id}", (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color="C3")
    for gt in header.get("ground_truth", []):
        rng, az = gt["range_m"], np.deg2rad(gt.get("azimuth_deg", 0.0))
        ax2.scatter([rng * np.cos(az)], [rng * np.sin(az)], marker="x", c="k", s=60)
    ax2.set(xlabel="x [m]", ylabel="y [m]",
            title=f"BEV: {len(out.get('objects', []))} objects, "
                  f"{n_confirmed} tracks ({doa_method})")
    ax2.axis("equal")
    plt.pause(0.001)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    print(f"Receiving from {args.connect}, topic={args.topic!r}")

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
    calibrated = False
    tracker = None

    try:
        while args.max_frames == 0 or idx < args.max_frames:
            if not sock.poll(1000, zmq.POLLIN):
                plt.pause(0.001)                       # stay responsive while idle
                continue
            topic, header_raw, payload = sock.recv_multipart()
            header = json.loads(header_raw.decode("utf-8"))
            if len(header["shape"]) != 3:
                print(f"skip non-MIMO frame shape={header['shape']}")
                continue

            if not calibrated:
                stream_dl, acq_dl = deadlines(header)
                print(f"Calibrating CFAR ({args.cfar}, one-time)...", flush=True)
                _prewarm_cfar(header, args.cfar)
                # Predict timestep = frame period, so the tracker's state
                # velocity is m/s, consistent with the m/s Doppler measurement.
                tracker = CartesianTracker(CartesianTrackerConfig(dt=stream_dl))
                calibrated = True

            fid = int(header["frame_id"])
            drops += detect_gap(fid, last_frame_id)
            last_frame_id = fid

            frame = deserialize_frame(header, payload)
            out = process_mimo_frame(frame, header, doa_method=args.doa,
                                     cfar_method=args.cfar)
            t_track0 = time.perf_counter()
            tracks = tracker.step(out["objects"])
            track_s = time.perf_counter() - t_track0
            _draw(axes, out, header, args.doa, tracks)

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


if __name__ == "__main__":
    main()
