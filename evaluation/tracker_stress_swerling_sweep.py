"""Fade-correlation × max_coast sweep on the Swerling missed-detection scenario.

Target realism report (Part I §4) extension of tracker report §3.1.2. The
scenario is the published missed-detection baseline (steady marginal-SNR
target, amplitude 0.0006) plus unit-mean-power Swerling-1 fading, so:

  - the STEADY rows (frame_corr=None strips the fluctuation) literally
    re-run the published §3.1.2 max_coast sweep, multi-seed — a direct
    anchor against the tracker report;
  - the rho rows all share the same marginal fade distribution (hence the
    same average detector Pd), differing only in burst structure: voltage
    corr rho -> power corr rho^2 -> fade decorrelation time -1/ln(rho^2)
    frames (rho 0.6 ~ 1 frame, rho 0.9 ~ 4.7 frames). Differences across
    rho therefore isolate fade DURATION from fade rate.

Each cell runs over several (radar seed, fluctuation seed) pairs — BOTH
stochastic sources redrawn per run (radar seed s, fluctuation seed 1000+s),
honest Monte Carlo over noise and fades together.

Outputs:
  report/tracker_stress_swerling_raw.csv     — one row per (cell, seed)
  report/tracker_stress_swerling_summary.csv — one row per cell

Usage:
    python -m evaluation.tracker_stress_swerling_sweep [--frames N] [--seeds K]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_id_switches,
    evaluate_run, run_pipeline,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)

# The PUBLISHED baseline scenario, loaded directly: fading is injected per
# grid cell by targets_with_corr(), so the steady rows are literally the
# §3.1.2 experiment. The _swerling scenario file exists only for the
# single-run CLI illustration, not for this sweep.
SCENARIO = "scenarios/scenario_missed_detections.json"
GRID_FRAME_CORR = [None, 0.0, 0.6, 0.9]   # None = steady (no fluctuation)
GRID_MAX_COAST = [1, 3, 5, 8]
DEFAULT_SEEDS = list(range(1, 7))          # radar seeds; fades use 1000+s
FLUCT_SEED_BASE = 1000
DEFAULT_FRAMES = 80
RAW_CSV = Path("report/tracker_stress_swerling_raw.csv")
SUMMARY_CSV = Path("report/tracker_stress_swerling_summary.csv")


def targets_with_corr(base_targets: list, frame_corr) -> list:
    """Copy of base_targets with fluctuation overridden.

    frame_corr=None strips fluctuation entirely (steady control at the same
    mean power); a float sets swerling1 with that voltage correlation.
    """
    if frame_corr is None:
        return [dataclasses.replace(t, fluctuation=None)
                for t in base_targets]
    return [
        dataclasses.replace(
            t, fluctuation={"model": "swerling1",
                            "frame_corr": float(frame_corr)})
        for t in base_targets
    ]


def run_cell(radar_cfg, base_targets, frame_corr, max_coast,
             seeds: list, n_frames: int) -> list:
    """Run one (frame_corr, max_coast) cell over the seed list.

    Returns one row dict per seed. Single-GT scenario: per-GT metrics are
    read from gt 0.
    """
    targets = targets_with_corr(base_targets, frame_corr)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(targets, gen, n_frames)
    rows = []
    for s in seeds:
        tracks_history = run_pipeline(
            radar_cfg, targets, n_frames, max_coast=max_coast,
            seed=s, fluctuation_seed=FLUCT_SEED_BASE + s)
        association_history = []
        for (f_gt, gt_at_frame), (_, tracks) in zip(gt_trajectory,
                                                    tracks_history):
            association_history.append(
                (f_gt, associate_per_frame(gt_at_frame, tracks)))
        switch_events = count_id_switches(association_history)
        summary = evaluate_run(association_history, switch_events,
                               tracks_history)
        m = summary["per_gt"][0]
        rows.append({
            "frame_corr": "steady" if frame_corr is None else frame_corr,
            "max_coast": max_coast,
            "seed": s,
            "pd": m["pd"],
            "distinct": m["distinct_track_ids"],
            "coast": m["coast_frames"],
            "n_switches": summary["n_switches"],
            "classification": summary["classification"],
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                   help="number of (radar, fluctuation) seed pairs per cell")
    p.add_argument("--scenario", default=SCENARIO,
                   help="scenario JSON to sweep. Default is the published "
                        "report-2 baseline, whose steady rows anchor against "
                        "report 2 §3.1.2; any other scenario has no published "
                        "anchor, so its steady rows are an in-experiment "
                        "control only. Non-default scenarios get their stem "
                        "suffixed onto the output CSV names, keeping the "
                        "baseline artifacts intact.")
    args = p.parse_args()
    seeds = list(range(1, args.seeds + 1))

    radar_cfg, base_targets = load_scenario(args.scenario)
    raw_csv, summary_csv = RAW_CSV, SUMMARY_CSV
    if args.scenario != SCENARIO:
        suffix = "_" + Path(args.scenario).stem.removeprefix(
            "scenario_missed_detections_")
        raw_csv = RAW_CSV.with_name(RAW_CSV.stem + suffix + RAW_CSV.suffix)
        summary_csv = SUMMARY_CSV.with_name(
            SUMMARY_CSV.stem + suffix + SUMMARY_CSV.suffix)
    print(f"loaded {args.scenario}: grid {len(GRID_FRAME_CORR)} frame_corr x "
          f"{len(GRID_MAX_COAST)} max_coast x {len(seeds)} seeds, "
          f"{args.frames} frames")

    raw_rows: list = []
    summary_rows: list = []
    header = (f"{'frame_corr':>10s} {'max_coast':>9s} {'Pd':>11s} "
              f"{'distinct':>8s} {'coast':>6s} {'switch':>6s}  classes")
    print("\n" + header)
    print("-" * len(header))
    for frame_corr in GRID_FRAME_CORR:
        for max_coast in GRID_MAX_COAST:
            rows = run_cell(radar_cfg, base_targets, frame_corr, max_coast,
                            seeds, args.frames)
            raw_rows += rows
            pds = np.array([r["pd"] for r in rows])
            classes = Counter(r["classification"] for r in rows)
            class_str = ", ".join(
                f"{n} {c}" for c, n in sorted(classes.items()))
            label = "steady" if frame_corr is None else f"{frame_corr:.1f}"
            summary_rows.append({
                "frame_corr": label,
                "max_coast": max_coast,
                "pd_mean": round(float(pds.mean()), 4),
                "pd_std": round(float(pds.std()), 4),
                "distinct_mean": round(float(np.mean(
                    [r["distinct"] for r in rows])), 2),
                "coast_mean": round(float(np.mean(
                    [r["coast"] for r in rows])), 2),
                "switches_mean": round(float(np.mean(
                    [r["n_switches"] for r in rows])), 2),
                "classes": class_str,
            })
            print(f"{label:>10s} {max_coast:>9d} "
                  f"{pds.mean():>6.2f}±{pds.std():<4.2f} "
                  f"{np.mean([r['distinct'] for r in rows]):>8.1f} "
                  f"{np.mean([r['coast'] for r in rows]):>6.1f} "
                  f"{np.mean([r['n_switches'] for r in rows]):>6.2f}  "
                  f"{class_str}")

    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"\nwrote {raw_csv}")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
