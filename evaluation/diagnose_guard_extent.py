"""Self-masking vs OS-CFAR guard size (target realism report, Part II §9.2).

Reproduces the report's claim that enlarging the OS-CFAR **range guard** to cover
an extended object's range extent recovers the per-cell detection fraction: an
object wider than the guard leaks its own scatterers into the CFAR training
cells, raising the threshold (self-masking), and a wider guard keeps them out.

The main detection sweep (evaluation.sweep_extent) runs the shared `detect_frame`
lens, which fixes `guard = (2, 4)`; this diagnostic instead calls range_doppler +
cfar_2d directly so the range guard can be swept. The per-cell fraction is the
same metric as sweep_extent.object_metrics: of the CFAR mask cells in the
object's range span at its Doppler row, the fraction that crossed threshold.

Default operating point: 1.2 m extent (K = 10) at the −13.6 dB sweep point (the
second-highest amplitude of sweep_extent's grid).

Usage:
    python -m evaluation.diagnose_guard_extent
    python -m evaluation.diagnose_guard_extent --range-extent-m 1.2 --frames 120
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from processing.cfar import cfar_2d
from processing.range_doppler import range_doppler
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario

# Second-highest amplitude of evaluation.sweep_extent's grid (~ −13.6 dB).
_SWEEP_AMPLITUDES = np.logspace(np.log10(3.5e-5), np.log10(3.5e-3), 10)
DEFAULT_SCALE = float(_SWEEP_AMPLITUDES[-2])
DEFAULT_GUARDS = [4, 8, 12, 16, 20]


def per_cell_fraction(radar_cfg, target, guard_range: int, n_frames: int) -> float:
    """Mean per-cell detection fraction in the object span, for one range guard.

    Mirrors sweep_extent.object_metrics' per-cell fraction, but with cfar_2d
    called directly so `guard_range` is tunable (detect_frame fixes guard=(2,4)).
    """
    gen = FMCWBeatSignalGenerator(radar_cfg)
    fracs = []
    for i in range(n_frames):
        frame = gen.generate_frame([target], frame_index=i)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", guard=(2, guard_range),
                       training=(4, 8), pfa=1e-4, window="hann")
        mask[:, 0] = False   # range-0 leakage notch
        g = gen.ground_truth([target], frame_index=i)[0]
        r_c, d_c, w = (g["range_bin"], g["doppler_bin_shifted"],
                       g["range_half_width_bins"])
        d_idx = int(round(d_c))
        lo = max(0, int(np.floor(r_c - w)))
        hi = min(rd.shape[1] - 1, int(np.ceil(r_c + w)))
        seg = mask[d_idx, lo:hi + 1]
        fracs.append(float(seg.mean()) if seg.size else 0.0)
    return float(np.mean(fracs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario",
                   default="scenarios/scenario_extended_detection.json")
    p.add_argument("--range-extent-m", type=float, default=1.2)
    p.add_argument("--n-scatterers", type=int, default=10)
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--guards", type=int, nargs="+", default=DEFAULT_GUARDS)
    args = p.parse_args()

    radar_cfg, base = load_scenario(args.scenario)
    snr_db = (20.0 * np.log10(base[0].amplitude * args.scale)
              - 10.0 * np.log10(radar_cfg.noise_power))
    target = dataclasses.replace(
        base[0], amplitude=base[0].amplitude * args.scale,
        extent={"n_scatterers": args.n_scatterers,
                "range_extent_m": args.range_extent_m})
    print(f"extent {args.range_extent_m} m (K={args.n_scatterers}), "
          f"scale={args.scale:.2e} (~{snr_db:.1f} dB), {args.frames} frames")
    for gr in args.guards:
        frac = per_cell_fraction(radar_cfg, target, gr, args.frames)
        print(f"  range guard={gr:2d}: per-cell fraction={frac:.3f}")


if __name__ == "__main__":
    main()
