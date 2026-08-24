"""Detection performance of the C fixed-point front end vs the float ones.

Three columns on the dynamic-range scenario, using report 5's metrics
unchanged — weak-target Pd, Pfa on target-free frames, output SNR, SFDR —
so the numbers here can be read directly against that study's bit-width
sweep:

    Python float   the reference NumPy chain
    C float        the same maths in C
    C int16        the fixed-point chain

The middle column is a control, not a result. Per-stage equivalence between
the Python and C float paths is already proven by tests/test_c_frontend.py,
so it must land within Monte-Carlo noise of the first column. If it does
not, the harness is wrong and the third column means nothing.

Why this scenario and not the one the timing benchmark uses: scenario_mimo
has targets strong enough that quantization cannot plausibly lose them (the
signal-quality bench measures 99.999% mask agreement on it). Quantization
should bite on a weak target sitting near the threshold beside a strong
one, which is exactly what scenario_mimo_fixedpoint_dynrange was built for.

Usage:
    (uv run) python -m evaluation.fixedpoint_c_eval [--frames N] [--seeds N]

Build the library first:  make -C c_frontend lib
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.fixedpoint_sweep import frame_metrics
from processing.c_frontend import CFrontend
from scripts.export_for_c import export
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario

SCENARIO = "scenarios/scenario_mimo_fixedpoint_dynrange.json"


def make_c_pipeline(fe: CFrontend, fixed: bool):
    """A pipeline callable for frame_metrics, backed by the C library.

    Generates the frame exactly as the NumPy reference does — same
    dataclasses.replace on the noise seed, same generator, same frame index
    — so the only difference from the reference column is which
    implementation consumes it. `fp` is accepted and ignored: the C chain
    has no FixedPointConfig, its arithmetic being fixed at compile time.
    """
    def c_power_db(cfg, targets, fp, frame_index, seed):
        cfg2 = dataclasses.replace(cfg, element_noise_seed=seed)
        gen = FMCWBeatSignalGenerator(cfg2)
        frame = np.ascontiguousarray(
            gen.generate_frame_mimo(targets, frame_index=frame_index),
            dtype=np.complex64)
        if fixed:
            rd = fe.q15_rd(frame)
            power_q = fe.q15_nci(rd)
            mask = fe.q15_cfar(power_q)
            power = power_q * fe.power_scale
        else:
            rd = fe.rd(frame)
            power = fe.nci(rd)
            mask = fe.cfar(power)
        return 10.0 * np.log10(power + 1e-12), mask
    return c_power_db


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detection performance: Python float vs C float vs C int16.")
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--frames", type=int, default=10,
                    help="frames per operating point")
    ap.add_argument("--seeds", type=int, default=5, help="Monte-Carlo seeds")
    args = ap.parse_args()

    cfg, targets = load_scenario(str(ROOT / args.scenario))

    with tempfile.TemporaryDirectory() as tmp:
        # params.cfg for this scenario: dimensions, CFAR geometry, the
        # calibrated alpha, and the ADC full scale. Exported rather than
        # hand-written so alpha matches what the Python chain calibrates
        # for itself — otherwise the C columns would differ by threshold
        # rather than by arithmetic.
        export(str(ROOT / args.scenario), tmp, frames=5)
        fe = CFrontend(Path(tmp) / "params.cfg")

        kw = dict(n_frames=args.frames, n_seeds=args.seeds)
        fe.reset_saturations()
        columns = [
            ("Python float", frame_metrics(cfg, targets, fp=None, **kw)),
            ("C float", frame_metrics(cfg, targets, fp=None,
                                      pipeline=make_c_pipeline(fe, False), **kw)),
            ("C int16", frame_metrics(cfg, targets, fp=None,
                                      pipeline=make_c_pipeline(fe, True), **kw)),
        ]
        sat = fe.saturations()
        fe.close()

    trials = args.frames * args.seeds
    print(f"detection performance ({args.scenario}, "
          f"{args.frames} frames x {args.seeds} seeds = {trials} trials)\n")
    print(f"  {'metric':<16}" + "".join(f"{name:>15}" for name, _ in columns))
    for key, label, fmt in (("pd", "weak-target Pd", "{:.3f}"),
                            ("pfa", "Pfa per cell", "{:.2e}"),
                            ("snr_db", "weak SNR [dB]", "{:.2f}"),
                            ("sfdr_db", "SFDR [dB]", "{:.2f}")):
        cells = "".join(f"{fmt.format(m[key]):>15}" for _, m in columns)
        print(f"  {label:<16}{cells}")

    print(f"\n  saturation during the C runs: {sat[0]} quantization, "
          f"{sat[1]} range-Doppler")
    print(f"  Pd resolution is 1/{trials} — raise --frames/--seeds before "
          f"reading small differences as real.")

    ref, c_float = columns[0][1], columns[1][1]
    drift = abs(c_float["pd"] - ref["pd"])
    if drift > 2.0 / trials:
        print(f"  WARNING: the C float control differs from Python float by "
              f"{drift:.3f} in Pd, more than Monte-Carlo noise explains — "
              f"treat the int16 column as unverified until that is resolved.")


if __name__ == "__main__":
    main()
