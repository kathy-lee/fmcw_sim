"""Per-stage timing of the Python MIMO pipeline, matching the C report.

Generates frames with the same generator/seed the C exporter uses (so both
implementations are timed on identical data), runs process_mimo_frame per
frame, and prints mean/max microseconds per stage in the same table format
as the C binary's timing report.

A warm-up frame runs first: the initial cfar_2d call triggers the one-time
empirical alpha calibration (lru_cached per process), which would otherwise
bury the real cfar timing. The C side has no such cost — its alpha arrives
precalibrated in params.cfg.

Stage-name mapping to the C report: integrate = nci; doa_pc matches;
cluster has no C counterpart, and total here includes cluster.

Usage:
    (uv run) python -m evaluation.time_mimo_pipeline --scenario scenarios/scenario_mimo.json --frames 3
(Note: First run takes tens of seconds — that's the α calibration warm-up; the timed frames come after.)

For comparison, run the following command to get the C timing report:
./c_frontend/fmcw_frontend --config data/c_export_smoke/params.cfg --frames data/c_export_smoke/frames --out data/c_export_smoke/detections.csv --dump data/c_export_smoke/dump
"""


from __future__ import annotations

import argparse

import numpy as np

from processing.mimo_frame import process_mimo_frame
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-stage timing of the Python MIMO pipeline.")
    ap.add_argument("--scenario", default="scenarios/scenario_mimo.json")
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    cfg, targets = load_scenario(args.scenario)
    gen = FMCWBeatSignalGenerator(cfg)
    frames = [gen.generate_frame_mimo(targets, frame_index=i)
              for i in range(args.frames)]
    header = build_header(frame=frames[0], frame_id=0, topic="cube",
                          metadata=gen.metadata(), ground_truth=[],
                          timestamp_unix_s=0.0)

    process_mimo_frame(frames[0], header, doa_method="bartlett", n_sources=1)

    acc: dict[str, list[float]] = {}
    for f in frames:
        t = process_mimo_frame(f, header, doa_method="bartlett",
                               n_sources=1)["timings"]
        for k, v in t.items():
            acc.setdefault(k, []).append(v)

    print(f"timing report ({len(frames)} frames):")
    print(f"  {'stage':<10}{'mean us':>12}{'max us':>12}")
    for k, vs in acc.items():
        print(f"  {k:<10}{np.mean(vs) * 1e6:>12.1f}{np.max(vs) * 1e6:>12.1f}")


if __name__ == "__main__":
    main()
