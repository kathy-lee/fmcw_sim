"""CLI driver: simulate radar cubes from a scenario, export .npz, preview/view.

Usage:
  python -m simulator.simulate_cubes --scenario <path> --out <dir> --frames N \
      [--levels adc,rd_cube,rad] [--fp] [--preview] [--doppler D]
  python -m simulator.simulate_cubes --view <path.npz> [--doppler D]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from processing.config import FixedPointConfig
from processing.cube_export import LEVELS, extract_cubes, save_cubes
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header


def _header(gen, frame, frame_index):
    return build_header(frame=frame, frame_id=frame_index, topic="cube",
                        metadata=gen.metadata(), ground_truth=[],
                        timestamp_unix_s=0.0)


def run_export(scenario, out_dir, frames, levels=LEVELS, fp=None):
    """Export `frames` frames of `scenario` to out_dir/frame_%04d.npz.

    fp: None (float), True (default FixedPointConfig with adc_full_scale sized to
    frame 0), or a FixedPointConfig instance. Returns the written paths.
    """
    cfg, targets = load_scenario(scenario)
    gen = FMCWBeatSignalGenerator(cfg)
    out_dir = Path(out_dir)

    fp_cfg = fp if isinstance(fp, FixedPointConfig) else None
    if fp is True:
        frame0 = gen.generate_frame_mimo(targets, frame_index=0)
        fp_cfg = dataclasses.replace(
            FixedPointConfig(), adc_full_scale=float(np.max(np.abs(frame0))))

    paths = []
    for i in range(frames):
        frame = gen.generate_frame_mimo(targets, frame_index=i)
        cubes = extract_cubes(frame, _header(gen, frame, i),
                              fp=fp_cfg, levels=tuple(levels))
        path = out_dir / f"frame_{i:04d}.npz"
        save_cubes(cubes, path)
        paths.append(str(path))
    return paths


def _load_npz_as_cubes(path):
    npz = np.load(path)
    cubes = {k: npz[k] for k in npz.files
             if not k.startswith("axis_") and k != "meta"}
    cubes["axes"] = {k[len("axis_"):]: npz[k] for k in npz.files
                     if k.startswith("axis_")}
    return cubes


def _preview(cubes, doppler_bin):
    import matplotlib.pyplot as plt

    from processing import cube_viz
    if "adc" in cubes:
        cube_viz.show_adc(cubes)
    if "power" in cubes:
        cube_viz.show_rd_map(cubes)
    if "rad" in cubes:
        cube_viz.show_rad_slice(cubes, doppler_bin)
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export/preview front-end cubes.")
    ap.add_argument("--scenario",
                    default="scenarios/scenario_mimo_fixedpoint_dynrange.json")
    ap.add_argument("--out", default="data/cubes")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--levels", default=",".join(LEVELS),
                    help="comma-separated subset of " + ",".join(LEVELS))
    ap.add_argument("--fp", action="store_true",
                    help="apply fixed-point (adc = Q_adc)")
    ap.add_argument("--preview", action="store_true",
                    help="plot the first exported frame")
    ap.add_argument("--view", default=None,
                    help="plot a previously saved .npz and exit")
    ap.add_argument("--doppler", type=int, default=0,
                    help="Doppler bin for the RAD slice")
    args = ap.parse_args(argv)

    if args.view is not None:
        _preview(_load_npz_as_cubes(args.view), args.doppler)
        return

    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    paths = run_export(args.scenario, args.out, args.frames, levels=levels,
                       fp=(True if args.fp else None))
    print(f"wrote {len(paths)} file(s) to {args.out}")
    if args.preview and paths:
        _preview(_load_npz_as_cubes(paths[0]), args.doppler)


if __name__ == "__main__":
    main()
