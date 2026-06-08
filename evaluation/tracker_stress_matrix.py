"""KF multi-seed robustness matrix for Family A (A.4).

Runs the **KF** tracker on the four Family-A stress scenarios over a fixed seed
list, then aggregates per scenario: mean ± std of the per-GT numeric metrics
(Pd, distinct IDs, coast frames) and the run-level metrics (ID switches, false
confirmed tracks), plus the classification-rate breakdown over the seeds.

This is a *robustness* check, not a filter comparison. Family-A motion is
~constant velocity, the one regime KF/EKF/IMM do not differ on (the drag-EKF and
IMM's drag mode collapse onto the CV filter, differing only at the sub-cm,
within-noise level), so Family A cannot meaningfully compare the filters — that
is the job of the Family-B and Family-C case studies. Here the matrix instead
quantifies how reliable the (KF) tracker is across noise realizations and shows
the §3.1-3.3 single-run illustrations are representative.

Reuses the pilot eval logic from tracker_stress.py unchanged; the only new hook
is run_pipeline(..., seed=) for independent noise realizations.

Outputs (all under docs/):
  - tracker_stress_matrix_raw.csv     — one row per (scenario, seed, gt_id);
                                        the auditable source of truth.
  - tracker_stress_matrix_summary.csv — one row per scenario.

Usage:
    python -m evaluation.tracker_stress_matrix [--frames N] [--seeds K]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_false_confirmed_tracks,
    count_id_switches, evaluate_run, run_pipeline,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


SCENARIOS = ["missed_detections", "crossing_targets", "overtaking_targets",
             "intermittent_clutter"]
# KF only: Family A is a robustness study, not a filter comparison (the filters
# are indistinguishable on constant-velocity motion; see the module docstring
# and report §3.4). The filter comparison lives in the Family-B/C case studies.
ALGORITHMS = ["kf"]
# Seeds start at 1, NOT 0: cfar_2d calibrates its detection threshold with its
# own default seed=0, and RadarConfig.seed==0 would make the generated frame's
# noise in-sample to that calibration (see report §2.5). The generator default
# is 99999 for the same reason. Any nonzero seed is an independent PCG64 stream,
# so 1..30 are cleanly decoupled from the CFAR calibration noise.
DEFAULT_SEEDS = list(range(1, 31))
DEFAULT_FRAMES = 80
CLASSES = ["CLEAN", "ID SWITCH", "TRACK BREAK", "TOTAL CONFUSION"]

DOCS = Path("docs")
RAW_CSV = DOCS / "tracker_stress_matrix_raw.csv"
SUMMARY_CSV = DOCS / "tracker_stress_matrix_summary.csv"


def scenario_path(name: str) -> str:
    return f"scenarios/scenario_{name}.json"


def run_seed(scenario_name: str, algorithm: str, seed: int,
             n_frames: int) -> list[dict]:
    """Run one (scenario, algorithm, seed) and return one row per GT.

    Run-level metrics (n_switches, false_confirmed_tracks, classification) are
    duplicated across the run's GT rows so the raw CSV is tidy/long; aggregate()
    de-duplicates them per seed.
    """
    path = scenario_path(scenario_name)
    radar_cfg, targets = load_scenario(path)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(targets, gen, n_frames)
    tracks_history = run_pipeline(
        radar_cfg, targets, n_frames, algorithm=algorithm, seed=seed,
    )

    association_history = []
    for (f_gt, gt_at_frame), (f_tr, tracks) in zip(gt_trajectory,
                                                    tracks_history):
        assert f_gt == f_tr
        association_history.append(
            (f_gt, associate_per_frame(gt_at_frame, tracks))
        )

    switch_events = count_id_switches(association_history,
                                      min_persistence_frames=2)
    summary = evaluate_run(association_history, switch_events, tracks_history)
    false = count_false_confirmed_tracks(association_history, tracks_history)

    rows = []
    for gt_id, m in summary["per_gt"].items():
        rows.append({
            "scenario": scenario_name,
            "algorithm": algorithm,
            "seed": seed,
            "gt_id": gt_id,
            "pd": m["pd"],
            "n_assoc_frames": m["n_frames_associated"],
            "distinct_track_ids": m["distinct_track_ids"],
            "coast_frames": m["coast_frames"],
            "run_n_switches": summary["n_switches"],
            "run_false_confirmed_tracks": false["n_false_confirmed"],
            "run_classification": summary["classification"],
        })
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse the long per-GT rows into one summary row per
    (scenario, algorithm).

    Per-GT numeric metrics (pd, distinct, coast) are pooled over all GT-rows ×
    seeds. Run-level metrics (switches, false tracks, classification) are
    de-duplicated to one value per seed first, then aggregated over the runs.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["scenario"], r["algorithm"])].append(r)

    out = []
    for scen in SCENARIOS:
        for algo in ALGORITHMS:
            rs = cells.get((scen, algo))
            if not rs:
                continue
            pd_mean, pd_std = _mean_std([r["pd"] for r in rs])
            dist_mean, dist_std = _mean_std(
                [r["distinct_track_ids"] for r in rs])
            coast_mean, coast_std = _mean_std([r["coast_frames"] for r in rs])

            # Run-level: one record per seed (any GT row carries the dup).
            by_seed = {}
            for r in rs:
                by_seed[r["seed"]] = r
            runs = list(by_seed.values())
            n = len(runs)
            sw_mean, sw_std = _mean_std([r["run_n_switches"] for r in runs])
            fc_mean, fc_std = _mean_std(
                [r["run_false_confirmed_tracks"] for r in runs])
            classes = [r["run_classification"] for r in runs]

            out.append({
                "scenario": scen,
                "algorithm": algo,
                "n_seeds": n,
                "pd_mean": pd_mean, "pd_std": pd_std,
                "distinct_mean": dist_mean, "distinct_std": dist_std,
                "coast_mean": coast_mean, "coast_std": coast_std,
                "switches_mean": sw_mean, "switches_std": sw_std,
                "false_confirmed_mean": fc_mean, "false_confirmed_std": fc_std,
                "pct_clean": classes.count("CLEAN") / n,
                "pct_id_switch": classes.count("ID SWITCH") / n,
                "pct_track_break": classes.count("TRACK BREAK") / n,
                "pct_total_confusion": classes.count("TOTAL CONFUSION") / n,
            })
    return out


def print_summary(summary: list[dict]) -> None:
    by_scen = defaultdict(list)
    for s in summary:
        by_scen[s["scenario"]].append(s)
    for scen in SCENARIOS:
        rows = by_scen.get(scen, [])
        if not rows:
            continue
        print()
        print("─" * 78)
        print(f"{scen}  (n_seeds={rows[0]['n_seeds']})")
        print("─" * 78)
        print(f"  {'algo':>4s}  {'Pd':>13s}  {'distinct':>10s}  "
              f"{'coast':>10s}  {'switches':>10s}  {'false':>10s}")
        for s in rows:
            print(f"  {s['algorithm']:>4s}  "
                  f"{s['pd_mean']:.2f}±{s['pd_std']:.2f}    "
                  f"{s['distinct_mean']:.2f}±{s['distinct_std']:.2f}  "
                  f"{s['coast_mean']:.1f}±{s['coast_std']:.1f}  "
                  f"{s['switches_mean']:.2f}±{s['switches_std']:.2f}  "
                  f"{s['false_confirmed_mean']:.1f}±{s['false_confirmed_std']:.1f}")
        for s in rows:
            br = (f"{int(round(s['pct_clean'] * s['n_seeds']))}/{s['n_seeds']} CLEAN, "
                  f"{int(round(s['pct_id_switch'] * s['n_seeds']))} ID SWITCH, "
                  f"{int(round(s['pct_track_break'] * s['n_seeds']))} TRACK BREAK, "
                  f"{int(round(s['pct_total_confusion'] * s['n_seeds']))} TOTAL CONFUSION")
            print(f"    {s['algorithm']:>4s} class: {br}")


def write_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "algorithm", "seed", "gt_id", "pd", "n_assoc_frames",
              "distinct_track_ids", "coast_frames", "run_n_switches",
              "run_false_confirmed_tracks", "run_classification"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def write_summary_csv(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "algorithm", "n_seeds", "pd_mean", "pd_std",
              "distinct_mean", "distinct_std", "coast_mean", "coast_std",
              "switches_mean", "switches_std", "false_confirmed_mean",
              "false_confirmed_std", "pct_clean", "pct_id_switch",
              "pct_track_break", "pct_total_confusion"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {path} ({len(summary)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                   help="number of seeds (uses 1..K, excluding 0; see §2.5)")
    args = p.parse_args()
    seeds = list(range(1, args.seeds + 1))

    total = len(SCENARIOS) * len(ALGORITHMS) * len(seeds)
    print(f"matrix (KF): {len(SCENARIOS)} scenarios × {len(seeds)} seeds "
          f"= {total} runs ({args.frames} frames)")

    rows = []
    done = 0
    for scen in SCENARIOS:
        for algo in ALGORITHMS:
            for seed in seeds:
                rows.extend(run_seed(scen, algo, seed, args.frames))
                done += 1
                print(f"  [{done}/{total}] {scen} {algo} seed={seed}",
                      flush=True)

    summary = aggregate(rows)
    print_summary(summary)
    write_raw_csv(rows, RAW_CSV)
    write_summary_csv(summary, SUMMARY_CSV)


if __name__ == "__main__":
    main()
