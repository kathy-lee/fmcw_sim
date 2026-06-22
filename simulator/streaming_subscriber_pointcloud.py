#!/usr/bin/env python3
"""Live MIMO point-cloud subscriber (Part II / M2c).

Consumes the MIMO beat-signal frames streamed by streaming_publisher.py and runs
the M2 detect-then-angle pipeline per frame (per-channel range-Doppler -> non-
coherent integration -> CFAR -> DoA at detected cells -> Cartesian point cloud),
live-updating a 3-panel view: integrated RD + CFAR | DoA spectra (4 methods) | BEV
point cloud. The dynamic analogue of streaming_subscriber_RD.py.

It also reports whether real-time processing is achieved, like
streaming_pipeline.py: per-stage timing (range-Doppler -> point cloud for now;
clustering/tracking can be added later) against two deadlines -- the stream cadence
(frame_period_s) and the TDM acquisition time (n_tx * n_chirps * chirp_period_s).

First production user of wire.deserialize_frame. No tracker (that is the M6 full
real-time pipeline, which extends this).

Usage:  python3 -m simulator.streaming_subscriber_pointcloud [--doa music]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from processing.cfar import cfar_2d
from processing.doa import estimate_doa, spectrum
from processing.point_cloud import (
    integrate, mimo_point_cloud, tdm_doppler_compensation)
from processing.range_doppler import range_doppler
from simulator.streaming_pipeline import compute_lag, detect_gap
from simulator.wire import deserialize_frame

C = 299_792_458.0
_SPECTRAL = ["bartlett", "capon", "music"]
# CFAR window params (method chosen per call). Default method is "ca"
# (cell-averaging) — fast and vectorized, like streaming_pipeline.py, which is
# what keeps the live viewer real-time; "os" is more robust but ~30x slower.
_CFAR_KW = dict(guard=(2, 4), training=(4, 8), pfa=1e-4)


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


def frame_params(header: dict) -> dict:
    """Derive the array/axis parameters needed by the M2 pipeline from a header."""
    cfg = header["metadata"]["config"]
    der = header["metadata"]["derived"]
    n_tx, n_rx = int(cfg["n_tx"]), int(cfg["n_rx"])
    n_virtual = n_tx * n_rx
    n_range, n_chirps = int(cfg["n_range"]), int(cfg["n_chirps"])

    pos = float(cfg["rx_spacing_wl"]) * np.arange(n_virtual)
    fs, slope = float(der["fs_hz"]), float(der["slope_hz_per_s"])
    range_axis = (C * np.fft.fftfreq(n_range, d=1.0 / fs)
                  / (2.0 * slope))[: n_range // 2]
    lam = float(der["lambda_m"])
    dop = np.fft.fftshift(np.fft.fftfreq(n_chirps, d=float(cfg["chirp_period_s"])))
    vel_axis = (lam * dop / 2.0) / n_tx          # TDM velocity scaling
    return dict(n_tx=n_tx, n_rx=n_rx, pos=pos,
                range_axis=range_axis, vel_axis=vel_axis)


def deadlines(header: dict) -> tuple[float, float]:
    """(stream cadence, TDM acquisition time) in seconds, from the header."""
    cfg = header["metadata"]["config"]
    stream = float(cfg["frame_period_s"])
    acq = int(cfg["n_tx"]) * int(cfg["n_chirps"]) * float(cfg["chirp_period_s"])
    return stream, acq


def process_mimo_frame(frame: np.ndarray, header: dict,
                       doa_method: str = "music", n_sources: int = 1,
                       cfar_method: str = "ca") -> dict:
    """Run the M2 detect-then-angle pipeline on one MIMO frame (testable core).

    Returns the stage outputs plus a `timings` dict (seconds) for rd / integrate
    / cfar / doa_pc, and `total` (range-Doppler -> point cloud).
    """
    p = frame_params(header)
    t0 = time.perf_counter()
    rd = np.stack([range_doppler(frame[v]) for v in range(frame.shape[0])])
    t1 = time.perf_counter()
    power = integrate(rd)
    power_db = 10.0 * np.log10(power + 1e-12)
    t2 = time.perf_counter()
    # n_integrate = n_virtual: α is recalibrated for the non-coherently
    # integrated (Gamma-V) noise so the realized Pfa stays nominal.
    mask = cfar_2d(power_db, method=cfar_method, n_integrate=frame.shape[0],
                   **_CFAR_KW)
    mask[:, 0] = False
    t3 = time.perf_counter()
    cloud = mimo_point_cloud(
        rd, mask, p["pos"], float(p["range_axis"][1]), p["vel_axis"],
        p["n_tx"], p["n_rx"], doa_method=doa_method, n_sources=n_sources)
    t4 = time.perf_counter()
    timings = {"rd": t1 - t0, "integrate": t2 - t1, "cfar": t3 - t2,
               "doa_pc": t4 - t3, "total": t4 - t0}
    return dict(rd=rd, power_db=power_db, mask=mask, cloud=cloud,
                timings=timings, **p)


def format_summary(frame_idx: int, stage_means_ms: tuple, total_ms: float,
                   period_ms: float, acq_ms: float, over_stream: int,
                   over_realtime: int, window: int, drops: int) -> str:
    """One-line real-time summary (range-Doppler -> point cloud)."""
    rd, integ, cfar, doa = stage_means_ms
    realtime = "yes" if over_realtime == 0 else "NO"
    return (f"[frame {frame_idx}] rd={rd:.1f} integ={integ:.1f} cfar={cfar:.1f} "
            f"doa+pc={doa:.1f} total={total_ms:.1f} (mean ms) | "
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


def _draw(axes, out: dict, header: dict, doa_method: str) -> None:
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
        ax2.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], cmap="coolwarm", s=16)
    for gt in header.get("ground_truth", []):
        rng, az = gt["range_m"], np.deg2rad(gt.get("azimuth_deg", 0.0))
        ax2.scatter([rng * np.cos(az)], [rng * np.sin(az)], marker="x", c="k", s=60)
    ax2.set(xlabel="x [m]", ylabel="y [m]",
            title=f"BEV point cloud ({doa_method})")
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
    stage_times = np.zeros((W, 5), dtype=np.float64)   # rd, integ, cfar, doa, total
    idx = 0
    drops = 0
    over_stream = over_realtime = 0
    last_frame_id = None
    stream_dl = acq_dl = 0.0
    calibrated = False

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
                calibrated = True

            fid = int(header["frame_id"])
            drops += detect_gap(fid, last_frame_id)
            last_frame_id = fid

            frame = deserialize_frame(header, payload)
            out = process_mimo_frame(frame, header, doa_method=args.doa,
                                     cfar_method=args.cfar)
            _draw(axes, out, header, args.doa)

            t = out["timings"]
            stage_times[idx % W] = (t["rd"], t["integrate"], t["cfar"],
                                    t["doa_pc"], t["total"])
            if t["total"] > stream_dl:
                over_stream += 1
            if t["total"] > acq_dl:
                over_realtime += 1
            idx += 1

            if idx % W == 0:
                means_ms = tuple(stage_times[:, k].mean() * 1e3 for k in range(4))
                total_ms = stage_times[:, 4].mean() * 1e3
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
