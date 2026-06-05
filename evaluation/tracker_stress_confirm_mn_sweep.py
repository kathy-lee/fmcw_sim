"""Tracker confirm_M-of-confirm_N sweep on the intermittent-clutter scenario
(A.3.2).

The clutter analogue of the max_coast sweep. Coherent clutter bursts spawn
spurious detections; the M-of-N confirmation rule is the lever that decides how
many of them get *promoted* to confirmed tracks. This 1-D sweep tightens the
rule along the "M of N=5" axis (ending at the strict 5-of-5 / M=N corner) and
reports the suppression-vs-cost tradeoff **jointly**:

  - false_confirmed_tracks / max_simultaneous_false — the suppression benefit
    (distinct clutter-born confirmed tracks, and the peak alive at once), and
  - pd / distinct_tracks / classification for the *true* target — the cost.

Single seed (the scenario's canonical seed), KF, sharing the
run_one(scenario, n_frames, confirm_M, confirm_N) shape of the other sweeps.
The table is printed to the console (the report embeds it in §3.3.2); it is not
written to disk.

Usage:
    python -m evaluation.tracker_stress_confirm_mn_sweep [--scenario PATH]
        [--frames N]
"""

from __future__ import annotations

import argparse

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, count_false_confirmed_tracks,
    count_id_switches, evaluate_run, run_pipeline,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


DEFAULT_SCENARIO_PATH = "scenarios/scenario_intermittent_clutter.json"
DEFAULT_FRAMES = 80
# Strictness axis: vary M at fixed N = 5, ending at the 5-of-5 (M = N) corner.
DEFAULT_CONFIRM_GRID = ((2, 5), (3, 5), (4, 5), (5, 5))


def run_one(scenario_path: str, n_frames: int,
            confirm_M: int, confirm_N: int) -> dict:
    """Run the pipeline once at the given M-of-N rule (single canonical seed);
    return a row pairing clutter-suppression columns with the true target's
    survival columns.

    Returns: {"confirm_M": int, "confirm_N": int,
              "false_confirmed_tracks": int, "max_simultaneous_false": int,
              "pd": float, "distinct_tracks": int, "classification": str}.
    pd / distinct_tracks describe the single true GT (this scenario has one).
    """
    radar_cfg, initial_targets = load_scenario(scenario_path)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    gt_trajectory = build_gt_trajectory(initial_targets, gen, n_frames)
    tracks_history = run_pipeline(
        radar_cfg, initial_targets, n_frames,
        confirm_M=confirm_M, confirm_N=confirm_N,
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

    gt0 = summary["per_gt"][0]
    return {
        "confirm_M": confirm_M,
        "confirm_N": confirm_N,
        "false_confirmed_tracks": false["n_false_confirmed"],
        "max_simultaneous_false": false["max_simultaneous_false"],
        "pd": gt0["pd"],
        "distinct_tracks": gt0["distinct_track_ids"],
        "classification": summary["classification"],
    }


def print_table(rows: list[dict]) -> None:
    print()
    print("─" * 72)
    print("Tracker confirm_M-of-N sweep — intermittent clutter scenario")
    print("─" * 72)
    print(f"  {'rule':>6s}  {'false':>6s}  {'max_sim':>7s}  {'Pd':>5s}  "
          f"{'distinct':>8s}  {'classification':<16s}")
    print(f"  {'-' * 6}  {'-' * 6}  {'-' * 7}  {'-' * 5}  {'-' * 8}  "
          f"{'-' * 16}")
    for r in rows:
        rule = f"{r['confirm_M']}of{r['confirm_N']}"
        print(f"  {rule:>6s}  {r['false_confirmed_tracks']:>6d}  "
              f"{r['max_simultaneous_false']:>7d}  {r['pd']:>5.2f}  "
              f"{r['distinct_tracks']:>8d}  {r['classification']:<16s}")
    print("─" * 72)
    print("  false = distinct clutter-born confirmed tracks; "
          "max_sim = peak alive at once")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO_PATH,
                   help="path to scenario JSON")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="number of frames per run")
    args = p.parse_args()

    print(f"sweep over confirm rules {DEFAULT_CONFIRM_GRID} on {args.scenario} "
          f"({args.frames} frames, {len(DEFAULT_CONFIRM_GRID)} runs)")

    rows = []
    for m, n in DEFAULT_CONFIRM_GRID:
        print(f"  running {m}-of-{n}...", end="", flush=True)
        row = run_one(args.scenario, args.frames, m, n)
        rows.append(row)
        print(f" -> false={row['false_confirmed_tracks']}, "
              f"Pd={row['pd']:.2f}, {row['classification']}")

    print_table(rows)


if __name__ == "__main__":
    main()
