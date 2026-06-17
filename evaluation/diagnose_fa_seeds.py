"""Noise-floor false-alarm rate of the detection lens, across noise seeds.

Reproduces the FA caveat in the target-realism report (Part I, §3.2). The
centroid-level false-alarm rate is measured through the SAME OS-CFAR + CC
lens as the amplitude sweep (via evaluation.sweep_amplitude.run_one_point),
but with an empty target list, so every surviving centroid is a false alarm
and the returned FP/frame is the pure noise-floor FA rate.

This shows two things the report relies on:
  - report 1's 100-frame estimate (~0.05/frame) was undersampled: the same
    canonical seed 99999 measures ~0.123/frame over 300 frames;
  - even at 300 frames the estimate has substantial seed-to-seed spread
    (FA is a rare event), so the rate is reported as a range over seeds.

Usage:
    python -m evaluation.diagnose_fa_seeds                  # 10 seeds, 300 frames
    python -m evaluation.diagnose_fa_seeds --frames 100 --seeds 99999
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from evaluation.sweep_amplitude import run_one_point
from simulator.beat_signal_generator import load_scenario

# Fixed, documented seed set so the reported range is reproducible. The first
# entry is report 1's canonical noise seed (RadarConfig.seed default).
DEFAULT_SEEDS = [99999, 1, 2, 3, 4, 5, 12345, 54321, 2024, 7]


def measure(scenario: str, seeds: list[int], n_frames: int) -> np.ndarray:
    """FA/frame at the noise floor for each noise seed (targets removed)."""
    cfg, _ = load_scenario(scenario)
    fas = []
    for s in seeds:
        c = dataclasses.replace(cfg, seed=s)
        # empty target list -> every centroid is a false alarm
        _, fa = run_one_point(c, [], "os", n_frames, advance_frame_index=True)
        fas.append(fa)
        print(f"  seed={s:>6}: FA/frame={fa:.3f}")
    return np.array(fas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="scenarios/scenario_two_targets.json")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = p.parse_args()

    fas = measure(args.scenario, args.seeds, args.frames)
    if fas.size > 1:
        print(f"\n  n={fas.size} seeds @ {args.frames}f:  "
              f"mean={fas.mean():.3f}  min={fas.min():.3f}  "
              f"max={fas.max():.3f}  std={fas.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
