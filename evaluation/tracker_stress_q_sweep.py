"""Tracker process_noise_q sweep on the missed-detections scenario (A.1.2).

A 1-D sweep over TrackerConfig.process_noise_q, reporting the *continuous*
track-position RMSE -- the metric q actually moves. Paired with the orthogonal
max_coast sweep (tracker_stress_maxcoast_sweep.py): max_coast governs survival
(measured by the discrete metrics), q governs estimate quality (measured here).
Both sweeps are single-seed (the scenario's canonical seed), 1-D, and share the
same run_one(scenario, n_frames, value) shape.

Two things the table shows:
  - RMSE varies monotonically with q (the smoothing-vs-responsiveness tradeoff;
    on this CV-matched slow target, lower q -> lower RMSE).
  - n_assoc is constant across q -- the explicit demonstration that q is inert
    on association, so it never belonged as an axis of the discrete sweep.

The table is printed to the console (the report embeds it in §3.1.2); it is not
written to disk.

Usage:
    python -m evaluation.tracker_stress_q_sweep [--scenario PATH] [--frames N]
"""

from __future__ import annotations

import argparse

from evaluation.tracker_stress import (
    associate_per_frame, build_gt_trajectory, run_pipeline,
    track_position_rmse,
)
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


DEFAULT_SCENARIO_PATH = "scenarios/scenario_missed_detections.json"
DEFAULT_FRAMES = 80
DEFAULT_Q_GRID = (0.05, 0.1, 0.3, 1.0, 3.0, 10.0)


def run_one(scenario_path: str, n_frames: int,
            process_noise_q: float) -> dict:
    """Run the pipeline once at the given q (single canonical seed); return a
    row with the continuous RMSE metric and the n_assoc control column.

    Returns: {"process_noise_q": float, "rmse_m": float, "n_assoc": int}.
    rmse_m is track-position RMSE in metres; n_assoc is the number of frames
    with an associated track (expected constant across q -- q's inertness).
    """
    radar_cfg, initial_targets = load_scenario(scenario_path)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    range_per_bin = float(gen.range_axis_m()[1] - gen.range_axis_m()[0])
    gt_trajectory = build_gt_trajectory(initial_targets, gen, n_frames)
    tracks_history = run_pipeline(
        radar_cfg, initial_targets, n_frames,
        process_noise_q=process_noise_q,
    )

    association_history = []
    for (f_gt, gt_at_frame), (f_tr, tracks) in zip(gt_trajectory,
                                                    tracks_history):
        assert f_gt == f_tr
        association_history.append(
            (f_gt, associate_per_frame(gt_at_frame, tracks))
        )

    rmse = track_position_rmse(gt_trajectory, tracks_history,
                               association_history, range_per_bin)
    n_assoc = sum(
        1 for _, m in association_history
        if any(v is not None for v in m.values())
    )
    return {
        "process_noise_q": process_noise_q,
        "rmse_m": rmse,
        "n_assoc": n_assoc,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print("─" * 56)
    print("Tracker process_noise_q sweep — missed detections scenario")
    print("─" * 56)
    print(f"  {'q':>6s}  {'RMSE cm':>8s}  {'n_assoc':>8s}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 8}")
    for r in rows:
        print(f"  {r['process_noise_q']:>6.2f}  "
              f"{r['rmse_m'] * 100:>8.2f}  {r['n_assoc']:>8d}")
    print("─" * 56)
    print("  (n_assoc constant across q  ->  q is inert on association)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO_PATH,
                   help="path to scenario JSON")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                   help="number of frames per run")
    args = p.parse_args()

    print(f"sweep over q={DEFAULT_Q_GRID} on {args.scenario} "
          f"({args.frames} frames, {len(DEFAULT_Q_GRID)} runs)")

    rows = []
    for q in DEFAULT_Q_GRID:
        print(f"  running q={q}...", end="", flush=True)
        row = run_one(args.scenario, args.frames, q)
        rows.append(row)
        print(f" -> RMSE {row['rmse_m'] * 100:.2f} cm")

    print_table(rows)


if __name__ == "__main__":
    main()
