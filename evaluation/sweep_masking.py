"""Target-masking sweep — measures each CFAR method's ability to detect a
weak target sitting close to a strong masker.

Setup (range only; both targets at v=0):
    masker:  amplitude 1.0, range 5.0 m
    weak:    amplitude 0.3, range 5.0 + separation_m
For each separation, runs N frames per CFAR method and reports:
    - Pd of the WEAK target (the only one being tested for masking)
    - Pd of the masker (sanity check: should be ~1 everywhere)
    - False-alarm count per frame (excludes detections matching either target)

Usage:
    python -m evaluation.sweep_masking --frames 100
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


TOL_BIN = 1.0

MASKER_RANGE_M = 5.0
MASKER_AMPLITUDE = 1.0
WEAK_AMPLITUDE = 0.3


def predicted_bins(target: Target, gen: FMCWBeatSignalGenerator) -> tuple[float, float]:
    f_beat = 2.0 * gen.slope_hz_per_s * target.range_m / C
    range_bin = f_beat / gen.fs_hz * gen.cfg.n_range
    v_res = 2.0 * gen.max_unambiguous_velocity_mps / gen.cfg.n_chirps
    doppler_bin = gen.cfg.n_chirps / 2.0 + target.velocity_mps / v_res
    return range_bin, doppler_bin


def _cell_detected(mask: np.ndarray, r_bin: float, d_bin: float) -> bool:
    """True if any cell within ±1 bin of (r_bin, d_bin) is set in mask."""
    n_d, n_r = mask.shape
    r0 = max(0, int(np.floor(r_bin - TOL_BIN)))
    r1 = min(n_r, int(np.ceil(r_bin + TOL_BIN)) + 1)
    d0 = max(0, int(np.floor(d_bin - TOL_BIN)))
    d1 = min(n_d, int(np.ceil(d_bin + TOL_BIN)) + 1)
    return bool(mask[d0:d1, r0:r1].any())


def run_one_point(
    radar_cfg: RadarConfig,
    targets: list[Target],
    cfar_method: str,
    cluster_kwargs: dict,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (centroid-level Pd, cell-level Pd, FA per frame)."""
    gen = FMCWBeatSignalGenerator(radar_cfg)
    truth_bins = [predicted_bins(t, gen) for t in targets]
    n_targets = len(targets)
    hits_centroid = np.zeros(n_targets, dtype=int)
    hits_cell = np.zeros(n_targets, dtype=int)
    fp_total = 0

    for _ in range(n_frames):
        frame = gen.generate_frame(targets, frame_index=0)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method=cfar_method,
                       guard=(2, 4), training=(4, 8), pfa=1e-4)
        mask[:, 0] = False   # range-0 leakage notch

        # Cell-level: does CFAR alone fire anywhere near each target's peak?
        for ti, (r_b, d_b) in enumerate(truth_bins):
            if _cell_detected(mask, r_b, d_b):
                hits_cell[ti] += 1

        # Centroid-level: full pipeline including clustering.
        centroids = cluster(mask, rd_db, **cluster_kwargs)
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
                hits_centroid[ti] += 1

    return (hits_centroid / n_frames,
            hits_cell / n_frames,
            fp_total / n_frames)


def run_sweep(
    radar_cfg: RadarConfig,
    separations_m: np.ndarray,
    methods: list[str],
    cluster_kwargs: dict,
    n_frames: int,
) -> tuple[dict, dict, dict]:
    # Keys: (method, stage, who) where stage in {"centroid", "cell"} and
    # who in {"weak", "masker"}.
    pd_results = {
        (m, stage, who): []
        for m in methods
        for stage in ("centroid", "cell")
        for who in ("weak", "masker")
    }
    fa_results = {m: [] for m in methods}

    for method in methods:
        for sep in separations_m:
            targets = [
                Target(range_m=MASKER_RANGE_M, velocity_mps=0.0,
                       amplitude=MASKER_AMPLITUDE),
                Target(range_m=MASKER_RANGE_M + sep, velocity_mps=0.0,
                       amplitude=WEAK_AMPLITUDE),
            ]
            pd_cen, pd_cel, fa = run_one_point(
                radar_cfg, targets, method, cluster_kwargs, n_frames
            )
            pd_results[(method, "centroid", "masker")].append((sep, pd_cen[0]))
            pd_results[(method, "centroid", "weak")].append((sep, pd_cen[1]))
            pd_results[(method, "cell", "masker")].append((sep, pd_cel[0]))
            pd_results[(method, "cell", "weak")].append((sep, pd_cel[1]))
            fa_results[method].append((sep, fa))
            print(f"  {method:3s}  sep={sep:.2f}m  "
                  f"cell:Pd_weak={pd_cel[1]:.2f}  "
                  f"centroid:Pd_weak={pd_cen[1]:.2f}  "
                  f"FA/frame={fa:.2f}")

    return pd_results, fa_results


def plot(
    pd_results: dict,
    fa_results: dict,
    methods: list[str],
    out: Path,
    config_label: str = "cc",
) -> None:
    fig, (ax_cell, ax_cen, ax_fa) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True
    )
    cmap = plt.get_cmap("tab10")
    # Distinct markers per method so overlapping flat lines remain readable.
    method_markers = {"ca": "o", "os": "s", "go": "^", "so": "D"}

    for mi, method in enumerate(methods):
        color = cmap(mi % 10)
        marker = method_markers.get(method, "o")

        pts = sorted(pd_results[(method, "cell", "weak")])
        ax_cell.plot([p[0] for p in pts], [p[1] for p in pts],
                     color=color, marker=marker, linewidth=1.6, markersize=9,
                     markerfacecolor="none", markeredgewidth=1.5,
                     alpha=0.85, label=method.upper())

        pts = sorted(pd_results[(method, "centroid", "weak")])
        ax_cen.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, marker=marker, linewidth=1.6, markersize=9,
                    markerfacecolor="none", markeredgewidth=1.5,
                    alpha=0.85, label=method.upper())

        pts = sorted(fa_results[method])
        ax_fa.plot([p[0] for p in pts], [p[1] for p in pts],
                   color=color, marker=marker, markersize=8, linewidth=1.4,
                   markerfacecolor="none", markeredgewidth=1.5,
                   alpha=0.85, label=method.upper())

    # Mark CFAR ring boundaries (range-axis only).
    # guard extends ±4 bins ≈ 0.30 m; training extends ±12 bins ≈ 0.90 m.
    for ax in (ax_cell, ax_cen):
        ax.axvline(0.30, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_cell.text(0.30, 1.05, "guard edge", fontsize=7, color="gray",
                 ha="center")
    ax_cell.text(0.90, 1.05, "training edge", fontsize=7, color="gray",
                 ha="center")

    ax_cell.set_ylabel("Pd (cell-level)")
    ax_cell.set_ylim(-0.05, 1.15)
    ax_cell.set_title(
        f"CFAR target masking [clustering={config_label}]: "
        f"masker (A={MASKER_AMPLITUDE}) fixed at {MASKER_RANGE_M} m; "
        f"weak (A={WEAK_AMPLITUDE}) at {MASKER_RANGE_M} m + separation"
    )
    ax_cell.grid(True, alpha=0.3)
    ax_cell.legend(fontsize=8, loc="lower right")

    ax_cen.set_ylabel("Pd (centroid-level)")
    ax_cen.set_ylim(-0.05, 1.15)
    ax_cen.grid(True, alpha=0.3)
    ax_cen.legend(fontsize=8, loc="lower right")

    ax_fa.set_xlabel("range separation [m]")
    ax_fa.set_ylabel("false alarms / frame")
    ax_fa.grid(True, alpha=0.3)
    ax_fa.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=Path,
                   default=Path("scenarios/scenario_two_targets.json"),
                   help="used only to pull the RadarConfig; targets are overridden")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--methods", nargs="+",
                   default=["ca", "os", "go", "so"],
                   help="CFAR methods to compare")
    p.add_argument("--clustering", choices=["cc", "dbscan"], default="cc",
                   help="clustering method")
    p.add_argument("--eps", type=float, default=1.0,
                   help="DBSCAN eps (ignored when clustering=cc)")
    p.add_argument("--min-samples", type=int, default=2,
                   help="DBSCAN min_samples (ignored when clustering=cc)")
    p.add_argument("--min-cells", type=int, default=2,
                   help="minimum cells per cluster (both methods)")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path; if omitted, derived from clustering config")
    args = p.parse_args()

    if args.clustering == "cc":
        cluster_kwargs = {"method": "cc", "min_cells": args.min_cells,
                          "connectivity": 8}
        config_label = "cc"
    else:
        cluster_kwargs = {"method": "dbscan", "min_cells": args.min_cells,
                          "eps": args.eps, "min_samples": args.min_samples}
        config_label = f"dbscan_eps{args.eps:g}"

    out = args.out or Path(f"docs/sweep_masking_{config_label}.png")

    # Span from just-outside-guard to well-outside-training (~12 bins = 0.9 m).
    # Bin resolution ≈ 0.075 m, so 0.3 m = 4 bins (guard edge).
    separations_m = np.array([0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0])
    print(f"Clustering: {cluster_kwargs}")
    print(f"Separation grid (m): {separations_m.tolist()}\n")

    radar_cfg, _ = load_scenario(str(args.scenario))
    pd_results, fa_results = run_sweep(
        radar_cfg, separations_m, args.methods, cluster_kwargs, args.frames,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(pd_results, fa_results, args.methods, out, config_label)


if __name__ == "__main__":
    main()
