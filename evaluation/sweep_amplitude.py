"""Sweep target amplitude at fixed Pfa to get Pd-vs-SNR curves.

For each CFAR method × amplitude scale, runs N frames of an in-memory FMCW
pipeline (no ZMQ): generate → 2D-FFT → CFAR → cluster, then matches cluster
centroids to ground-truth targets within ±1 range bin AND ±1 Doppler bin.

Outputs: per-target Pd vs input SNR, one curve per CFAR method, plus the
per-frame false-alarm rate.

Usage:
    python -m evaluation.sweep_amplitude \
        --scenario scenarios/scenario_two_targets.json \
        --frames 100 \
        --out docs/sweep_amplitude.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.range_doppler import range_doppler
from simulator.beat_signal_generator import (
    C, FMCWBeatSignalGenerator, RadarConfig, Target, load_scenario,
)


TOL_BIN = 1.0   # match tolerance: ±1 range bin AND ±1 Doppler bin


def predicted_bins(target: Target, gen: FMCWBeatSignalGenerator) -> tuple[float, float]:
    """Return (range_bin, doppler_bin) — Doppler bin is post-fftshift."""
    f_beat = 2.0 * gen.slope_hz_per_s * target.range_m / C
    range_bin = f_beat / gen.fs_hz * gen.cfg.n_range
    v_res = 2.0 * gen.max_unambiguous_velocity_mps / gen.cfg.n_chirps
    doppler_bin = gen.cfg.n_chirps / 2.0 + target.velocity_mps / v_res
    return range_bin, doppler_bin


def run_one_point(
    radar_cfg: RadarConfig,
    targets: list[Target],
    cfar_method: str,
    n_frames: int,
) -> tuple[np.ndarray, float]:
    """Return (per-target Pd array, FA per frame) for one (method, scale)."""
    gen = FMCWBeatSignalGenerator(radar_cfg)
    truth_bins = [predicted_bins(t, gen) for t in targets]
    n_targets = len(targets)
    hits = np.zeros(n_targets, dtype=int)
    fp_total = 0

    # frame_index=0 every iteration: holds the target's position fixed
    # (no inter-frame motion drift), while gen.rng still advances so each
    # frame gets an independent noise realization.
    for _ in range(n_frames):
        frame = gen.generate_frame(targets, frame_index=0)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method=cfar_method,
                       guard=(2, 4), training=(4, 8), pfa=1e-4)
        mask[:, 0] = False   # range-0 leakage notch
        centroids = cluster(mask, rd_db, method="cc", min_cells=2)

        # Mark each target as detected if any centroid is within tolerance;
        # centroids that don't match any target are false positives.
        detected = [False] * n_targets
        for c in centroids:
            matched = False
            for ti, (r_b, d_b) in enumerate(truth_bins):
                if (abs(c.range_bin - r_b) <= TOL_BIN and
                        abs(c.doppler_bin - d_b) <= TOL_BIN):
                    detected[ti] = True
                    matched = True
            if not matched:
                fp_total += 1
        for ti, d in enumerate(detected):
            if d:
                hits[ti] += 1

    pd = hits / n_frames
    return pd, fp_total / n_frames


def run_sweep(
    scenario: Path,
    amplitudes: np.ndarray,
    methods: list[str],
    n_frames: int,
) -> tuple[dict, dict, list[Target]]:
    radar_cfg, base_targets = load_scenario(str(scenario))
    n_targets = len(base_targets)

    pd_results = {(m, ti): [] for m in methods for ti in range(n_targets)}
    fa_results = {m: [] for m in methods}

    for method in methods:
        for scale in amplitudes:
            scaled = [
                Target(range_m=t.range_m,
                       velocity_mps=t.velocity_mps,
                       amplitude=t.amplitude * scale,
                       phase_rad=t.phase_rad)
                for t in base_targets
            ]
            pd, fa = run_one_point(radar_cfg, scaled, method, n_frames)
            for ti, t in enumerate(scaled):
                snr_db = (20.0 * np.log10(t.amplitude)
                          - 10.0 * np.log10(radar_cfg.noise_power))
                pd_results[(method, ti)].append((snr_db, pd[ti]))
            avg_snr = float(np.mean([
                20.0 * np.log10(t.amplitude)
                - 10.0 * np.log10(radar_cfg.noise_power)
                for t in scaled
            ]))
            fa_results[method].append((avg_snr, fa))
            print(f"  {method:3s}  scale={scale:8.2e}  "
                  f"Pd={pd.round(2).tolist()}  FA/frame={fa:.3f}")

    return pd_results, fa_results, base_targets


def plot(
    pd_results: dict,
    fa_results: dict,
    base_targets: list[Target],
    methods: list[str],
    out: Path,
) -> None:
    fig, (ax_pd, ax_fa) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", ":", "-."]

    for mi, method in enumerate(methods):
        color = cmap(mi % 10)
        for ti, t in enumerate(base_targets):
            pts = sorted(pd_results[(method, ti)])
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            label = (f"{method.upper()}  target@{t.range_m:.1f}m"
                     f"(A0={t.amplitude:.2f})")
            ax_pd.plot(x, y, color=color, linestyle=linestyles[ti % 4],
                       marker="o", markersize=4, linewidth=1.4, label=label)

        pts = sorted(fa_results[method])
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax_fa.plot(x, y, color=color, marker="s", markersize=4,
                   linewidth=1.4, label=method.upper())

    ax_pd.set_ylabel("Pd")
    ax_pd.set_ylim(-0.05, 1.05)
    ax_pd.set_title("Detection probability vs input SNR (Pfa=1e-4, min_cells=2)")
    ax_pd.grid(True, alpha=0.3)
    ax_pd.legend(fontsize=7, ncol=2, loc="lower right")

    ax_fa.set_xlabel("input SNR per sample [dB]")
    ax_fa.set_ylabel("false alarms / frame")
    ax_fa.grid(True, alpha=0.3)
    ax_fa.legend(fontsize=8, loc="best")
    ax_fa.axhline(y=0, color="k", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=Path,
                   default=Path("scenarios/scenario_two_targets.json"))
    p.add_argument("--frames", type=int, default=100,
                   help="frames per (method, amplitude) point")
    p.add_argument("--methods", nargs="+",
                   default=["ca", "os", "go", "so"])
    p.add_argument("--out", type=Path,
                   default=Path("docs/sweep_amplitude.png"))
    args = p.parse_args()

    # Option B: ±20 dB around predicted A_50% = 3.5e-4
    amplitudes = np.logspace(np.log10(3.5e-5), np.log10(3.5e-3), 10)
    print(f"Amplitude grid (10 points, ±20 dB around A_50%):\n  {amplitudes}\n")

    pd_results, fa_results, base_targets = run_sweep(
        args.scenario, amplitudes, args.methods, args.frames,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(pd_results, fa_results, base_targets, args.methods, args.out)


if __name__ == "__main__":
    main()
