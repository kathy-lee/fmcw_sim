"""Tracker hyperparameter sweep on the missed-detections scenario (C.2).

Runs the same pipeline as evaluation.tracker_stress, but across a grid of
TrackerConfig.max_coast and TrackerConfig.process_noise_q values. Produces a
single comparison table (printed to stdout and written to CSV) so the C.2
hypothesis can be confirmed numerically:

  - tighter max_coast  -> more track breaks (distinct_tracks grows, Pd drops)
  - higher process_noise_q -> more aggressive coasting (wider gates absorb
    redetections before max_coast fires, defending against track break even
    at tight max_coast settings)

Usage:
    python -m evaluation.tracker_stress_sweep [--scenario PATH] [--frames N]
        [--out PATH]

The plotting step from tracker_stress is skipped here; the sweep produces a
table only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_id_switches,
    run_pipeline, summarize_run,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


DEFAULT_SCENARIO_PATH = "scenarios/scenario_missed_detections.json"
DEFAULT_FRAMES = 80
DEFAULT_MAX_COAST_GRID = (1, 2, 3, 5)
DEFAULT_Q_GRID = (0.1, 0.3, 1.0)
DEFAULT_CSV_PATH = Path("docs/tracker_stress_sweep_missed_detections.csv")


def run_one(scenario_path: str, n_frames: int, max_coast: int,
            process_noise_q: float) -> dict:
    """Run the pipeline once with the given tracker overrides; return a row of
    metrics for the single GT (the missed-detections scenario has 1 target).

    Returns: {"max_coast": int, "process_noise_q": float, "pd": float,
              "distinct_tracks": int, "coast_frames": int,
              "classification": str}.
    """
    radar_cfg, initial_targets = load_scenario(scenario_path)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(initial_targets, gen, n_frames)
    tracks_history = run_pipeline(
        radar_cfg, initial_targets, n_frames,
        max_coast=max_coast, process_noise_q=process_noise_q,
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
    summary = summarize_run(association_history, switch_events,
                            tracks_history)

    # Single-GT scenario: report gt_0 metrics.
    gt0 = summary["per_gt"][0]
    return {
        "max_coast": max_coast,
        "process_noise_q": process_noise_q,
        "pd": gt0["pd"],
        "distinct_tracks": gt0["distinct_track_ids"],
        "coast_frames": gt0["coast_frames"],
        "classification": summary["classification"],
    }


def print_table(rows: list[dict]) -> None:
    """Tabular console summary, grouped by max_coast for readability."""
    print()
    print("─" * 78)
    print("Tracker stress sweep — missed detections scenario")
    print("─" * 78)
    print(f"  {'max_coast':>9s}  {'q':>5s}  {'Pd':>6s}  "
          f"{'distinct':>8s}  {'coast':>5s}  {'classification':<16s}")
    print(f"  {'-' * 9}  {'-' * 5}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 5}  {'-' * 16}")
    for r in rows:
        print(f"  {r['max_coast']:>9d}  {r['process_noise_q']:>5.2f}  "
              f"{r['pd']:>6.2f}  {r['distinct_tracks']:>8d}  "
              f"{r['coast_frames']:>5d}  {r['classification']:<16s}")
    print("─" * 78)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["max_coast", "process_noise_q", "pd",
                  "distinct_tracks", "coast_frames", "classification"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO_PATH,
                   help="path to scenario JSON")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="number of frames per run")
    p.add_argument("--out", type=Path, default=DEFAULT_CSV_PATH,
                   help="output CSV path for the comparison table")
    args = p.parse_args()

    print(f"sweep over max_coast={DEFAULT_MAX_COAST_GRID} × "
          f"q={DEFAULT_Q_GRID} on {args.scenario} ({args.frames} frames, "
          f"{len(DEFAULT_MAX_COAST_GRID) * len(DEFAULT_Q_GRID)} runs)")

    rows = []
    for mc in DEFAULT_MAX_COAST_GRID:
        for q in DEFAULT_Q_GRID:
            print(f"  running max_coast={mc}, q={q}...", end="", flush=True)
            row = run_one(args.scenario, args.frames, mc, q)
            rows.append(row)
            print(f" -> {row['classification']}")

    print_table(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
