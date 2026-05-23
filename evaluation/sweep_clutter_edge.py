"""Clutter-edge CFAR evaluation: sweep target position against a fixed
clutter region, measure Pd per CFAR method at both cell and centroid level.

Level 1: clutter spans all Doppler in a range band. Target Doppler is fixed;
target range sweeps across the clutter boundaries.

Level 2: clutter is restricted to a (range × near-zero-Doppler) rectangle.
Cross-pattern sweep: target moves in range at v=0 (Slice A) and in velocity
at r=5 m (Slice B).

Usage:
    python -m evaluation.sweep_clutter_edge --level {1,2}
                                            [--clutter-removal]
                                            [--frames N]
                                            [--out PATH]
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
    C, ClutterRegion, FMCWBeatSignalGenerator, RadarConfig, Target,
    load_scenario,
)


TOL_BIN = 1.0
CNR_DB = 10.0
SCENARIO_PATH = "scenarios/scenario_clutter_edge.json"


def predicted_bins(target: Target, gen: FMCWBeatSignalGenerator) -> tuple[float, float]:
    f_beat = 2.0 * gen.slope_hz_per_s * target.range_m / C
    range_bin = f_beat / gen.fs_hz * gen.cfg.n_range
    v_res = 2.0 * gen.max_unambiguous_velocity_mps / gen.cfg.n_chirps
    doppler_bin = gen.cfg.n_chirps / 2.0 + target.velocity_mps / v_res
    return range_bin, doppler_bin


def _cell_detected(mask: np.ndarray, r_bin: float, d_bin: float) -> bool:
    n_d, n_r = mask.shape
    r0 = max(0, int(np.floor(r_bin - TOL_BIN)))
    r1 = min(n_r, int(np.ceil(r_bin + TOL_BIN)) + 1)
    d0 = max(0, int(np.floor(d_bin - TOL_BIN)))
    d1 = min(n_d, int(np.ceil(d_bin + TOL_BIN)) + 1)
    return bool(mask[d0:d1, r0:r1].any())


def run_one_point(
    radar_cfg: RadarConfig,
    target_r: float,
    target_v: float,
    target_amplitude: float,
    cfar_method: str,
    cluster_kwargs: dict,
    n_frames: int,
    clutter_removal: bool,
) -> tuple[float, float, float]:
    """Returns (cell-level Pd, centroid-level Pd, FA per frame)."""
    gen = FMCWBeatSignalGenerator(radar_cfg)
    target = Target(range_m=target_r, velocity_mps=target_v,
                    amplitude=target_amplitude)
    r_b, d_b = predicted_bins(target, gen)

    hits_centroid = 0
    hits_cell = 0
    fp_total = 0

    for _ in range(n_frames):
        frame = gen.generate_frame([target], frame_index=0)
        rd = range_doppler(frame, window="hann",
                           dc_removal=True,
                           clutter_removal=clutter_removal)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method=cfar_method,
                       guard=(2, 4), training=(4, 8), pfa=1e-4,
                       window="hann",
                       dc_removal=True,
                       clutter_removal=clutter_removal)
        mask[:, 0] = False   # range-0 leakage notch (convention)

        if _cell_detected(mask, r_b, d_b):
            hits_cell += 1

        centroids = cluster(mask, rd_db, **cluster_kwargs)
        target_matched = False
        for c in centroids:
            if (abs(c.range_bin - r_b) <= TOL_BIN and
                    abs(c.doppler_bin - d_b) <= TOL_BIN):
                target_matched = True
            else:
                fp_total += 1
        if target_matched:
            hits_centroid += 1

    return (hits_cell / n_frames,
            hits_centroid / n_frames,
            fp_total / n_frames)


_RANGE_GRID_L1 = [1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.5]
_FIXED_TARGET_VELOCITY_L1 = 2.0   # m/s, well outside zero-Doppler
_CLUTTER_RANGE_BAND = (4.0, 6.0)
_CLUTTER_DOPPLER_BAND = (-0.5, 0.5)


def _load_scenario_for_clutter(scenario_path: str) -> tuple[RadarConfig, float]:
    """Load the clutter-edge scenario. The targets list in the JSON is a
    template — only its first entry's amplitude is used (range/velocity are
    overridden per sweep position). Clutter regions in the JSON are ignored;
    the script sets them per --level.
    """
    radar_cfg, targets = load_scenario(scenario_path)
    if not targets:
        raise ValueError(
            f"{scenario_path} must contain at least one template target "
            f"(only its amplitude is used)."
        )
    return radar_cfg, targets[0].amplitude


def _build_radar_cfg_l1(radar_cfg: RadarConfig) -> RadarConfig:
    radar_cfg.clutter = [ClutterRegion(
        range_m_min=_CLUTTER_RANGE_BAND[0],
        range_m_max=_CLUTTER_RANGE_BAND[1],
        cnr_db=CNR_DB,
    )]
    return radar_cfg


def _positions_l1() -> list[tuple[float, float]]:
    return [(r, _FIXED_TARGET_VELOCITY_L1) for r in _RANGE_GRID_L1]


def plot_level1(results: dict, out: Path, clutter_removal: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    method_markers = {"ca": "o", "os": "s", "go": "^", "so": "D"}

    methods = list(results.keys())
    for mi, method in enumerate(methods):
        color = cmap(mi % 10)
        marker = method_markers.get(method, "o")
        items = sorted(results[method].items())
        x = [r for ((r, _v), _) in items]
        pd_cell = [vals[0] for (_, vals) in items]
        pd_cen = [vals[1] for (_, vals) in items]
        ax.plot(x, pd_cell, color=color, marker=marker, linewidth=1.6,
                markersize=9, markerfacecolor="none", markeredgewidth=1.5,
                alpha=0.85, label=f"{method.upper()} cell")
        ax.plot(x, pd_cen, color=color, marker=marker, linewidth=1.0,
                linestyle="--", markersize=7, markerfacecolor=color,
                alpha=0.85, label=f"{method.upper()} centroid")

    ax.axvspan(_CLUTTER_RANGE_BAND[0], _CLUTTER_RANGE_BAND[1],
               color="gray", alpha=0.15, label="clutter range")
    ax.set_xlabel(
        f"target range [m]  (target velocity fixed at "
        f"{_FIXED_TARGET_VELOCITY_L1} m/s)"
    )
    ax.set_ylabel("Pd of target")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(
        f"Level 1 — broad clutter (CNR={CNR_DB:g} dB across all Doppler); "
        f"clutter_removal={clutter_removal}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def run_sweep(
    radar_cfg: RadarConfig,
    positions: list[tuple[float, float]],
    target_amplitude: float,
    methods: list[str],
    cluster_kwargs: dict,
    n_frames: int,
    clutter_removal: bool,
) -> dict:
    """Generic sweep over target (range, velocity) positions.

    Returns: {method: {(r, v): (pd_cell, pd_centroid, fa)}} — same shape for
    any sweep pattern (1D line, cross, or future 2D grid).
    """
    results: dict = {m: {} for m in methods}
    for method in methods:
        for (r, v) in positions:
            pd_cell, pd_cen, fa = run_one_point(
                radar_cfg, r, v, target_amplitude, method, cluster_kwargs,
                n_frames, clutter_removal,
            )
            results[method][(r, v)] = (pd_cell, pd_cen, fa)
            print(f"  {method:3s}  r={r:.2f}m v={v:+.2f}m/s  "
                  f"cell={pd_cell:.2f}  cen={pd_cen:.2f}  fa={fa:.2f}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, choices=[1, 2], required=True)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--clutter-removal", action="store_true",
                   help="Enable slow-time clutter removal in range_doppler. "
                        "Off by default — turning on partially erases the "
                        "static-clutter component (only meaningful for "
                        "Level 2 as a follow-up).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    methods = ["ca", "os", "go", "so"]
    cluster_kwargs = {"method": "peak", "min_cells": 2,
                      "peak_neighborhood": 3}
    radar_cfg, target_amplitude = _load_scenario_for_clutter(SCENARIO_PATH)
    print(f"loaded {SCENARIO_PATH}; target amplitude = {target_amplitude}")

    if args.level == 1:
        radar_cfg = _build_radar_cfg_l1(radar_cfg)
        positions = _positions_l1()
        suffix = "_clutterRemoval" if args.clutter_removal else ""
        out = args.out or Path(f"docs/sweep_clutter_edge_level1{suffix}.png")
        print(f"Level 1 sweep, clutter_removal={args.clutter_removal}, "
              f"{len(positions)} positions × {len(methods)} methods × "
              f"{args.frames} frames")
        results = run_sweep(radar_cfg, positions, target_amplitude,
                            methods, cluster_kwargs,
                            args.frames, args.clutter_removal)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_level1(results, out, args.clutter_removal)
        return

    raise NotImplementedError("Level 2 not yet wired (Task 8).")


if __name__ == "__main__":
    main()
