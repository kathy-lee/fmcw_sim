"""Diagnose CFAR Pfa calibration on the actual pipeline.

Runs N noise-only frames (no targets, leakage stripped) through the full
range-Doppler + CFAR chain and measures the empirical cell-level false-alarm
rate per CFAR method. Compares to design Pfa.

If GO is biased low and SO biased high in mirror image, the most likely root
cause is that the MC alpha-calibration in `_goso_alpha` assumes i.i.d.
exponential noise per cell, while the real RD map has bin-to-bin correlation
introduced by Hann windowing.

Usage:
    python -m scripts.diagnose_cfar_pfa
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from processing.cfar import cfar_2d
from processing.range_doppler import range_doppler
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


N_FRAMES = 200
PFA_DESIGN = 1e-4
METHODS = ["ca", "os", "go", "so"]


def main() -> None:
    radar_cfg, _ = load_scenario("scenarios/scenario_two_targets.json")
    # RadarConfig.seed defaults to 99999 to stay decoupled from cfar's
    # calibration noise (which uses seed=0 internally). This diagnostic
    # therefore measures generalization, not training-set self-consistency.
    radar_cfg = replace(radar_cfg, leakage_amplitude=0.0)
    gen = FMCWBeatSignalGenerator(radar_cfg)

    fa_counts = {m: 0 for m in METHODS}
    total_cells = 0
    rd_shape = None

    for i in range(N_FRAMES):
        frame = gen.generate_frame(targets=[], frame_index=0)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        if rd_shape is None:
            rd_shape = rd_db.shape
            print(f"RD map shape: {rd_shape}")
        for method in METHODS:
            mask = cfar_2d(rd_db, method=method, pfa=PFA_DESIGN)
            fa_counts[method] += int(mask.sum())
        total_cells += rd_db.size

    print(f"\nFrames: {N_FRAMES}")
    print(f"Total cells tested: {total_cells:,}")
    print(f"Design Pfa: {PFA_DESIGN}")
    print(f"Designed FA/frame: {PFA_DESIGN * rd_shape[0] * rd_shape[1]:.2f}")
    print()
    print(f"{'method':>6s}  {'empirical Pfa':>14s}  "
          f"{'ratio_to_design':>17s}  {'cell-FA/frame':>14s}")
    print("-" * 60)
    for method in METHODS:
        emp_pfa = fa_counts[method] / total_cells
        ratio = emp_pfa / PFA_DESIGN
        fa_per_frame = fa_counts[method] / N_FRAMES
        print(f"{method.upper():>6s}   {emp_pfa:>12.2e}       "
              f"{ratio:>10.2f}x      {fa_per_frame:>10.2f}")


if __name__ == "__main__":
    main()
