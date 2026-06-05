"""Tracker max_coast sweep on the missed-detections scenario (A.1.2).

A 1-D sweep over TrackerConfig.max_coast on the marginal-SNR missed-detections
scenario, reporting the *discrete* survival metrics (Pd, distinct tracks, coast
frames, classification). This is the half of the old max_coast x q sweep that
actually carries signal: coast tolerance is the survival lever for missed
detections, walking the run CLEAN -> ID SWITCH -> TOTAL CONFUSION as it tightens.

The orthogonal knob, process_noise_q, is swept separately in
tracker_stress_q_sweep.py against a continuous RMSE metric -- against these
discrete metrics q is provably inert (it never changes which frames associate),
so it is not a second axis here.

The table is printed to the console (the report embeds it in §3.1.2); it is not
written to disk.

Usage:
    python -m evaluation.tracker_stress_maxcoast_sweep [--scenario PATH]
        [--frames N]
"""

from __future__ import annotations

import argparse

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_id_switches,
    run_pipeline, evaluate_run,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


DEFAULT_SCENARIO_PATH = "scenarios/scenario_missed_detections.json"
DEFAULT_FRAMES = 80
DEFAULT_MAX_COAST_GRID = (1, 2, 3, 5)


def run_one(scenario_path: str, n_frames: int, max_coast: int) -> dict:
    """Run the pipeline once at the given max_coast (q left at default); return
    a row of discrete metrics for the single GT (this scenario has 1 target).

    Returns: {"max_coast": int, "pd": float, "distinct_tracks": int,
              "coast_frames": int, "classification": str}.
    """
    radar_cfg, initial_targets = load_scenario(scenario_path)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(initial_targets, gen, n_frames)
    tracks_history = run_pipeline(
        radar_cfg, initial_targets, n_frames, max_coast=max_coast,
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
    summary = evaluate_run(association_history, switch_events,
                            tracks_history)

    gt0 = summary["per_gt"][0]
    return {
        "max_coast": max_coast,
        "pd": gt0["pd"],
        "distinct_tracks": gt0["distinct_track_ids"],
        "coast_frames": gt0["coast_frames"],
        "classification": summary["classification"],
    }


def print_table(rows: list[dict]) -> None:
    """Tabular console summary."""
    print()
    print("─" * 64)
    print("Tracker max_coast sweep — missed detections scenario")
    print("─" * 64)
    print(f"  {'max_coast':>9s}  {'Pd':>6s}  {'distinct':>8s}  "
          f"{'coast':>5s}  {'classification':<16s}")
    print(f"  {'-' * 9}  {'-' * 6}  {'-' * 8}  {'-' * 5}  {'-' * 16}")
    for r in rows:
        print(f"  {r['max_coast']:>9d}  {r['pd']:>6.2f}  "
              f"{r['distinct_tracks']:>8d}  {r['coast_frames']:>5d}  "
              f"{r['classification']:<16s}")
    print("─" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO_PATH,
                   help="path to scenario JSON")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="number of frames per run")
    args = p.parse_args()

    print(f"sweep over max_coast={DEFAULT_MAX_COAST_GRID} on {args.scenario} "
          f"({args.frames} frames, {len(DEFAULT_MAX_COAST_GRID)} runs)")

    rows = []
    for mc in DEFAULT_MAX_COAST_GRID:
        print(f"  running max_coast={mc}...", end="", flush=True)
        row = run_one(args.scenario, args.frames, mc)
        rows.append(row)
        print(f" -> {row['classification']}")

    print_table(rows)


if __name__ == "__main__":
    main()
