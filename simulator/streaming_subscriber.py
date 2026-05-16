#!/usr/bin/env python3
"""Receive FMCW beat-signal frames and compute range-Doppler maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from processing.range_doppler import range_doppler

C = 299_792_458.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--connect", default="tcp://127.0.0.1:5555")
    p.add_argument("--topic", default="fmcw")
    p.add_argument("--max-frames", type=int, default=0, help="0 means run forever")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--window", choices=["rect", "hann"], default="hann")
    p.add_argument("--save-dir", default=None, help="Optional directory for .npy RD maps")
    return p.parse_args()


def get_axes(header: dict[str, Any], n_range_fft: int, n_doppler_fft: int) -> tuple[np.ndarray, np.ndarray]:
    meta = header["metadata"]
    cfg = meta["config"]
    derived = meta["derived"]

    fs_hz = float(derived["fs_hz"])
    slope = float(derived["slope_hz_per_s"])
    lambda_m = float(derived["lambda_m"])
    chirp_period_s = float(cfg["chirp_period_s"])

    range_freqs = np.fft.fftfreq(n_range_fft, d=1.0 / fs_hz)
    range_axis_m = C * range_freqs / (2.0 * slope)
    range_axis_m = range_axis_m[: n_range_fft // 2]

    doppler_freqs = np.fft.fftfreq(n_doppler_fft, d=chirp_period_s)
    doppler_freqs = np.fft.fftshift(doppler_freqs)
    velocity_axis_mps = lambda_m * doppler_freqs / 2.0

    return range_axis_m, velocity_axis_mps


# Original local implementation, kept for reference. The function now lives at
# processing/range_doppler.py (imported above) with added DC removal and
# slow-time clutter suppression.
#
# def range_doppler(frame: np.ndarray, window: str = "hann") -> np.ndarray:
#     n_chirps, n_range = frame.shape
#
#     if window == "hann":
#         range_win = np.hanning(n_range)[None, :]
#         doppler_win = np.hanning(n_chirps)[:, None]
#     else:
#         range_win = 1.0
#         doppler_win = 1.0
#
#     x = frame * range_win
#     rfft = np.fft.fft(x, axis=1)
#
#     # Keep only positive beat-frequency range bins.
#     rfft = rfft[:, : n_range // 2]
#
#     rfft = rfft * doppler_win
#     rd = np.fft.fftshift(np.fft.fft(rfft, axis=0), axes=0)
#     return rd


def top_peaks(power_db: np.ndarray, range_axis_m: np.ndarray, velocity_axis_mps: np.ndarray, k: int) -> list[dict[str, float]]:
    if k <= 0:
        return []

    flat = power_db.ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    peaks = []
    for flat_idx in idx:
        d_bin, r_bin = np.unravel_index(flat_idx, power_db.shape)
        peaks.append({
            "power_db": float(power_db[d_bin, r_bin]),
            "range_bin": int(r_bin),
            "doppler_bin_shifted": int(d_bin),
            "range_m": float(range_axis_m[r_bin]),
            "velocity_mps": float(velocity_axis_mps[d_bin]),
        })
    return peaks


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

    received = 0
    try:
        while args.max_frames == 0 or received < args.max_frames:
            topic, header_raw, payload = sock.recv_multipart()
            header = json.loads(header_raw.decode("utf-8"))

            shape = tuple(header["shape"])
            if header["dtype"] != "complex64":
                raise ValueError(f"Unsupported dtype: {header['dtype']}")

            frame = np.frombuffer(payload, dtype=np.complex64).reshape(shape)
            rd = range_doppler(frame, window=args.window)
            power_db = 20.0 * np.log10(np.abs(rd) + 1e-12)

            range_axis_m, velocity_axis_mps = get_axes(header, frame.shape[1], frame.shape[0])
            peaks = top_peaks(power_db, range_axis_m, velocity_axis_mps, args.top_k)

            print(f"\nframe {header['frame_id']}  shape={shape}")
            print("ground truth:")
            for gt in header.get("ground_truth", []):
                print(
                    f"  R={gt['range_m']:.2f} m, "
                    f"v={gt['velocity_mps']:.2f} m/s, "
                    f"range_bin={gt['range_bin']:.2f}, "
                    f"doppler_bin_shifted={gt['doppler_bin_shifted']:.2f}"
                )
            print("top peaks:")
            for p in peaks:
                print(
                    f"  {p['power_db']:7.2f} dB  "
                    f"R={p['range_m']:7.2f} m  "
                    f"v={p['velocity_mps']:7.2f} m/s  "
                    f"bins=({p['doppler_bin_shifted']}, {p['range_bin']})"
                )

            if save_dir:
                np.save(save_dir / f"rd_power_db_{header['frame_id']:06d}.npy", power_db)

            received += 1

    except KeyboardInterrupt:
        print("\nSubscriber stopped.")
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
