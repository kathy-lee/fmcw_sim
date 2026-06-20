"""Extended-target tracking sweep (target realism report, Part II §10).

Extends report 2 §3.1 to extended objects, point-tracked. Two parts:

  (i) Glint characterisation: measure sigma_glint = std of the merged-cluster
      range centroid about the true centre (range bins), as a function of
      extent. This is a deliverable law AND the calibration that sets matched
      R = R_resolution + sigma_glint^2 (a fixed config value; report-1 CFAR-
      alpha hygiene -- calibrate on a held-out seed set, evaluate out-of-sample).

  (ii) Headline matrix: extent x {cc + default R, dbscan + default R,
       dbscan + matched R}. cc->dbscan isolates the gap-bridging merger;
       default->matched R isolates the glint benefit. Cells report
       Pd / distinct IDs / localization RMSE (metres, GOSPA cutoff) /
       classification over seeds. extent=0 reproduces report 2
       (clustering-independent).

Outputs:
  report/tracker_stress_extent_raw.csv     -- one row per (cell, seed)
  report/tracker_stress_extent_summary.csv -- one row per cell (+ sigma_glint)

Usage:
    python -m evaluation.tracker_stress_extent_sweep [--frames N] [--seeds K]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.sweep_extent import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_id_switches,
    evaluate_run, run_pipeline, track_position_rmse,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)
from processing.range_doppler import range_doppler
from processing.cfar import cfar_2d
from processing.clustering import cluster

SCENARIO = "scenarios/scenario_extended_tracking.json"
GRID_EXTENT_M = [None, 0.3, 0.6, 1.2]
DEFAULT_SEEDS = list(range(1, 7))
CALIB_SEEDS = [101, 102, 103]      # held-out, for sigma_glint calibration
FLUCT_SEED_BASE = 1000
RANGE_RESOLUTION_VAR = 0.5         # R_resolution (matches TrackerConfig default)
CUTOFF_M = 1.0                     # GOSPA localization cutoff (metres)
DEFAULT_FRAMES = 80
RAW_CSV = Path("report/tracker_stress_extent_raw.csv")
SUMMARY_CSV = Path("report/tracker_stress_extent_summary.csv")


def _extent_dict(extent_m):
    if extent_m is None:
        return None
    return {"n_scatterers": 10, "range_extent_m": float(extent_m)}


def _range_per_bin(gen, cfg) -> float:
    return gen.max_unambiguous_range_m / (cfg.n_range / 2)


def measure_sigma_glint(radar_cfg, target, n_frames, seeds) -> float:
    """Std (in range bins) of the merged dbscan centroid about the true centre,
    over the given (held-out) seeds. Point target -> 0."""
    errs = []
    for s in seeds:
        cfg = dataclasses.replace(radar_cfg, seed=s)
        gen = FMCWBeatSignalGenerator(cfg)
        for i in range(n_frames):
            frame = gen.generate_frame([target], frame_index=i)
            rd = range_doppler(frame, window="hann", dc_removal=True,
                               clutter_removal=False)
            rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
            mask = cfar_2d(rd_db, method="os", guard=(2, 4),
                           training=(4, 8), pfa=1e-4, window="hann")
            mask[:, 0] = False
            # min_cells=2 to match the tracking pipeline (run_cell) this
            # sigma_glint calibrates: measure the wander of the centroid the
            # tracker actually receives, not of singleton noise cells.
            cents = cluster(mask, rd_db, method="dbscan", min_cells=2,
                            eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            g = gen.ground_truth([target], frame_index=i)[0]
            r_c, d_c, w = (g["range_bin"], g["doppler_bin_shifted"],
                           g["range_half_width_bins"])
            in_span = [c for c in cents
                       if abs(c.range_bin - r_c) <= w + 1.0
                       and abs(c.doppler_bin - d_c) <= 1.0]
            if in_span:
                nearest = min(in_span, key=lambda c: abs(c.range_bin - r_c))
                errs.append(nearest.range_bin - r_c)
    if not errs:
        return 0.0
    return float(np.std(errs))


def run_cell(radar_cfg, base_target, extent, cluster_method, range_meas_var,
             seeds, n_frames) -> list:
    """One (extent, config) cell over the seed list."""
    target = dataclasses.replace(base_target, extent=extent)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    range_per_bin = _range_per_bin(gen, radar_cfg)
    gt_trajectory = build_gt_trajectory([target], gen, n_frames)
    # Both clusterers use min_cells=2 (report-2 default; filters singleton
    # noise) so the cc-vs-dbscan comparison is fair -- it isolates the
    # gap-bridging merger from any min_cells difference. dbscan additionally
    # gets its gap-bridging eps/min_samples; cc ignores them.
    cluster_kwargs = dict(cluster_method=cluster_method, cluster_min_cells=2)
    if cluster_method == "dbscan":
        cluster_kwargs.update(cluster_eps=DBSCAN_EPS,
                              cluster_min_samples=DBSCAN_MIN_SAMPLES)
    rows = []
    for s in seeds:
        tracks_history = run_pipeline(
            radar_cfg, [target], n_frames, seed=s,
            fluctuation_seed=FLUCT_SEED_BASE + s,
            range_meas_var=range_meas_var, **cluster_kwargs)
        association_history = []
        for (f_gt, gt_at_frame), (_, tracks) in zip(gt_trajectory,
                                                    tracks_history):
            association_history.append(
                (f_gt, associate_per_frame(gt_at_frame, tracks)))
        switch_events = count_id_switches(association_history)
        summary = evaluate_run(association_history, switch_events,
                               tracks_history)
        rmse_m = track_position_rmse(gt_trajectory, tracks_history,
                                     association_history, range_per_bin,
                                     cutoff_m=CUTOFF_M)
        m = summary["per_gt"][0]
        rows.append({
            "extent_m": 0.0 if extent is None else extent["range_extent_m"],
            "cluster": cluster_method,
            "matched_R": range_meas_var is not None,
            "seed": s,
            "pd": m["pd"],
            "distinct": m["distinct_track_ids"],
            "coast": m["coast_frames"],
            "n_switches": summary["n_switches"],
            "rmse_m": rmse_m,
            "classification": summary["classification"],
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = p.parse_args()
    seeds = list(range(1, args.seeds + 1))

    radar_cfg, base_targets = load_scenario(SCENARIO)
    base = base_targets[0]
    print(f"loaded {SCENARIO}: extents {GRID_EXTENT_M} x 3 configs x "
          f"{len(seeds)} seeds, {args.frames} frames")

    # (i) Calibrate matched R per extent from the held-out glint characterisation.
    sigma_glint = {}
    matched_R = {}
    for extent_m in GRID_EXTENT_M:
        extent = _extent_dict(extent_m)
        t = dataclasses.replace(base, extent=extent)
        sg = measure_sigma_glint(radar_cfg, t, args.frames, CALIB_SEEDS)
        sigma_glint[extent_m] = sg
        matched_R[extent_m] = RANGE_RESOLUTION_VAR + sg * sg
        print(f"  extent={extent_m}: sigma_glint={sg:.2f} bins -> "
              f"matched_R={matched_R[extent_m]:.2f}")

    # (ii) The matrix: each extent x {cc, dbscan, dbscan + matched R}, evaluated
    # on the (out-of-sample) seeds; printed, then written to the CSVs below.
    configs = [("cc", None), ("dbscan", None), ("dbscan", "matched")]
    raw_rows, summary_rows = [], []
    header = (f"{'extent_m':>8s} {'cluster':>8s} {'R':>8s} {'Pd':>11s} "
              f"{'distinct':>8s} {'RMSE_m':>7s} {'switch':>6s}  classes")
    print("\n" + header); print("-" * len(header))
    for extent_m in GRID_EXTENT_M:
        extent = _extent_dict(extent_m)
        for cluster_method, r_mode in configs:
            rmv = matched_R[extent_m] if r_mode == "matched" else None
            rows = run_cell(radar_cfg, base, extent, cluster_method, rmv,
                            seeds, args.frames)
            raw_rows += rows
            pds = np.array([r["pd"] for r in rows])
            rmse = np.array([r["rmse_m"] for r in rows])
            classes = Counter(r["classification"] for r in rows)
            class_str = ", ".join(f"{n} {c}" for c, n in sorted(classes.items()))
            r_label = "matched" if r_mode == "matched" else "default"
            summary_rows.append({
                "extent_m": extent_m if extent_m is not None else 0.0,
                "cluster": cluster_method, "R": r_label,
                "sigma_glint": round(sigma_glint[extent_m], 3),
                "pd_mean": round(float(pds.mean()), 4),
                "pd_std": round(float(pds.std()), 4),
                "distinct_mean": round(float(np.mean(
                    [r["distinct"] for r in rows])), 2),
                "rmse_m_mean": round(float(np.nanmean(rmse)), 4),
                "switches_mean": round(float(np.mean(
                    [r["n_switches"] for r in rows])), 2),
                "classes": class_str,
            })
            el = "point" if extent_m is None else f"{extent_m:.1f}"
            print(f"{el:>8s} {cluster_method:>8s} {r_label:>8s} "
                  f"{pds.mean():>6.2f}±{pds.std():<4.2f} "
                  f"{np.mean([r['distinct'] for r in rows]):>8.1f} "
                  f"{np.nanmean(rmse):>7.3f} "
                  f"{np.mean([r['n_switches'] for r in rows]):>6.2f}  "
                  f"{class_str}")

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)
    print(f"\nwrote {RAW_CSV}")
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
