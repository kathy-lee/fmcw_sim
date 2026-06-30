"""Fixed-point bit-width sweep + metrics for the MIMO front end.

Golden = the float pipeline (fp=None). For a FixedPointConfig we measure, vs
the float golden on the dynamic-range scenario: weak-target Pd, Pfa, noise-floor
elevation, and SFDR. `sweep_stage` walks one stage's bit width.
See docs/superpowers/specs/2026-06-26-fixedpoint-mimo-frontend-design.md.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from processing.config import FixedPointConfig
from processing.mimo_frame import frame_params, process_mimo_frame
from simulator.beat_signal_generator import FMCWBeatSignalGenerator
from simulator.wire import build_header

_GATE = 1   # +/- cells around the weak-target cell counted as "the target"


def _header(gen, frame, fi=0):
    return build_header(frame=frame, frame_id=fi, topic="fp",
                        metadata=gen.metadata(), ground_truth=[],
                        timestamp_unix_s=0.0)


def _target_cell(cfg, targets, gid, frame_index=0):
    """(doppler_bin, range_bin) of target `gid`, via ground truth."""
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame_mimo(targets, frame_index=frame_index)
    p = frame_params(_header(gen, frame, frame_index))
    g = gen.ground_truth(targets, frame_index=frame_index)[gid]
    r_bin = int(np.argmin(np.abs(p["range_axis"] - g["range_m"])))
    d_bin = int(np.argmin(np.abs(p["vel_axis"] - g["velocity_mps"])))
    return d_bin, r_bin


def _weak_target_cell(cfg, targets, frame_index=0):
    """(doppler_bin, range_bin) of the weakest-amplitude target."""
    gid = int(np.argmin([t.amplitude for t in targets]))
    return _target_cell(cfg, targets, gid, frame_index)


def _power_db(cfg, targets, fp, frame_index, seed):
    cfg2 = dataclasses.replace(cfg, element_noise_seed=seed)
    gen = FMCWBeatSignalGenerator(cfg2)
    frame = gen.generate_frame_mimo(targets, frame_index=frame_index)
    out = process_mimo_frame(frame, _header(gen, frame, frame_index), fp=fp)
    return out["power_db"], out["mask"]


def _gate(d, r):
    return (slice(max(0, d - _GATE), d + _GATE + 1),
            slice(max(0, r - _GATE), r + _GATE + 1))


def frame_metrics(cfg, targets, fp, *, n_frames=10, n_seeds=5):
    """Pd/Pfa/noise-floor/SFDR for one config, Monte-Carlo over seeds.

    Pd is for the weakest target; SFDR references the strongest (dominant) target
    and excludes BOTH target gates so the spur is a genuine quantization artifact,
    not the other real target. The targets move, so the gates are computed
    per frame (the ground-truth cell, axes-independent of seed/fp)."""
    gid_weak = int(np.argmin([t.amplitude for t in targets]))
    gid_strong = int(np.argmax([t.amplitude for t in targets]))
    gen = FMCWBeatSignalGenerator(cfg)
    p = frame_params(_header(gen, gen.generate_frame_mimo(targets, frame_index=0)))
    ra, va = p["range_axis"], p["vel_axis"]

    def gate_at(gid, fi):
        g = gen.ground_truth(targets, frame_index=fi)[gid]
        r_bin = int(np.argmin(np.abs(ra - g["range_m"])))
        d_bin = int(np.argmin(np.abs(va - g["velocity_mps"])))
        return _gate(d_bin, r_bin)

    weak_gates = [gate_at(gid_weak, fi) for fi in range(n_frames)]
    strong_gates = [gate_at(gid_strong, fi) for fi in range(n_frames)]

    hits = trials = fa = fa_cells = 0
    snr_acc, sfdr_acc = [], []
    # noise_acc = []   # noise-floor-elevation metric (retired -> SNR loss; kept commented)
    for seed in range(n_seeds):
        for fi in range(n_frames):
            power_db, mask = _power_db(cfg, targets, fp, fi, seed)
            weak_sl, strong_sl = weak_gates[fi], strong_gates[fi]
            # Pd: any detection within +/-_GATE of the (moving) weak-target cell
            hits += int(mask[weak_sl].any()); trials += 1
            # Pfa: false-alarm rate on a NOISE-ONLY frame (no targets), same
            # fp/seed/frame -- a clean false-alarm rate, decoupled from the strong
            # return's sidelobes (whose spur story is carried by SFDR instead).
            _, mask_n = _power_db(cfg, [], fp, fi, seed)
            mn = mask_n.copy(); mn[:, :6] = False
            fa += int(mn.sum()); fa_cells += mn.size
            # noise floor = median power_db over off-DC cells (needed for SNR)
            noise_floor = float(np.median(power_db[:, 6:]))
            # noise_acc.append(noise_floor)   # noise-floor-elevation metric (retired)
            # Output SNR of the weak target: its peak above the noise floor.
            # SNR loss (vs float, clamped at 0) is derived in sweep_stage.
            snr_acc.append(float(power_db[weak_sl].max()) - noise_floor)
            # SFDR: dominant (strong) target peak vs the largest non-target spur
            ref = float(power_db[strong_sl].max())
            spur_map = power_db.copy()
            spur_map[weak_sl] = -np.inf       # exclude both real targets
            spur_map[strong_sl] = -np.inf
            spur_map[:, :6] = -np.inf         # and the min-range/DC region
            sfdr_acc.append(ref - float(spur_map.max()))
    return dict(pd=hits / trials, pfa=fa / fa_cells,
                snr_db=float(np.mean(snr_acc)),
                # noise_floor_db=float(np.mean(noise_acc)),   # retired -> SNR loss
                sfdr_db=float(np.median(sfdr_acc)))


def sweep_stage(cfg, targets, *, stage, bits, n_frames=10, n_seeds=5,
                base=None, adc_full_scale=None):
    """Sweep one FixedPointConfig field across `bits`, others held wide.
    Returns {float_*: golden metrics, rows: [per-bit metrics]}."""
    if adc_full_scale is None:
        gen = FMCWBeatSignalGenerator(cfg)
        adc_full_scale = float(np.max(np.abs(
            gen.generate_frame_mimo(targets, frame_index=0))))
    wide = base or dict(adc_bits=20, cube_bits=24, rd_bits=24, mag_bits=24,
                        acc_bits=32)
    gold = frame_metrics(cfg, targets, fp=None, n_frames=n_frames, n_seeds=n_seeds)
    rows = []
    for b in bits:
        kw = dict(wide); kw[stage] = b
        fp = FixedPointConfig(adc_full_scale=adc_full_scale, **kw)
        m = frame_metrics(cfg, targets, fp=fp, n_frames=n_frames, n_seeds=n_seeds)
        m["bits"] = b; rows.append(m)
    return dict(stage=stage,
                float_pd=gold["pd"], float_pfa=gold["pfa"],
                float_snr_db=gold["snr_db"],
                # float_noise_floor_db=gold["noise_floor_db"],   # retired -> SNR loss
                float_sfdr_db=gold["sfdr_db"], rows=rows)


def strong_return_spur_rate(cfg, targets, fp, *, n_frames=10, n_seeds=5,
                            guard=(3, 6)):
    """Spurious-detection rate in target-free cells of the dynamic-range SCENE
    (strong return PRESENT), excluding a guard (doppler, range) around BOTH moving
    targets. This is NOT the classical Pfa (which is target-free/noise-only and
    bit-robust): it captures the strong return's quantization spurs crossing the
    CFAR threshold -- a dynamic-range effect, the real bit-driver for the cube
    stage. Denominator = eligible (non-excluded) cells."""
    gid_w = int(np.argmin([t.amplitude for t in targets]))
    gid_s = int(np.argmax([t.amplitude for t in targets]))
    gen = FMCWBeatSignalGenerator(cfg)
    p = frame_params(_header(gen, gen.generate_frame_mimo(targets, frame_index=0)))
    ra, va = p["range_axis"], p["vel_axis"]
    gd, gr = guard

    def cell(gid, fi):
        g = gen.ground_truth(targets, frame_index=fi)[gid]
        return (int(np.argmin(np.abs(va - g["velocity_mps"]))),
                int(np.argmin(np.abs(ra - g["range_m"]))))

    fa = elig = 0
    for seed in range(n_seeds):
        for fi in range(n_frames):
            _, mask = _power_db(cfg, targets, fp, fi, seed)
            keep = np.ones_like(mask, dtype=bool)
            keep[:, :6] = False
            for gid in (gid_w, gid_s):
                d, r = cell(gid, fi)
                keep[max(0, d - gd):d + gd + 1, max(0, r - gr):r + gr + 1] = False
            fa += int((mask & keep).sum()); elig += int(keep.sum())
    return fa / elig


def _plot_sweep(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # (metric, ylabel, filename, golden-key | "zero" | None)
    specs = [
        ("pd", "Pd (weak target)", "fixedpoint_pd_vs_bits.png", "float_pd"),
        ("pfa", "Pfa (per cell)", "fixedpoint_pfa_vs_bits.png", "float_pfa"),
        ("snr_loss", "SNR loss [dB]", "fixedpoint_snrloss_vs_bits.png", "zero"),
        ("sfdr_db", "SFDR [dB]", "fixedpoint_sfdr_vs_bits.png", "float_sfdr_db"),
    ]
    for metric, ylabel, fname, goldkey in specs:
        plt.figure(figsize=(7, 4.5))
        for res in results:
            bits = [r["bits"] for r in res["rows"]]
            if metric == "snr_loss":
                # SNR loss vs float, clamped at 0 (negative = noise-floor-estimator
                # artifact when the floor collapses; loss can't be physically < 0).
                y = [max(0.0, res["float_snr_db"] - r["snr_db"]) for r in res["rows"]]
            else:
                y = [r[metric] for r in res["rows"]]
            plt.plot(bits, y, "o-", label=res["stage"])
        if goldkey == "zero":
            plt.axhline(0.0, ls="--", color="k", lw=1.2, label="float (no loss)")
        elif goldkey is not None:
            gold = float(np.mean([res[goldkey] for res in results]))
            plt.axhline(gold, ls="--", color="k", lw=1.2, label="float golden")
        plt.xlabel("bits"); plt.ylabel(ylabel); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(out / fname, dpi=110); plt.close()


def main():
    import argparse
    from simulator.beat_signal_generator import load_scenario
    ap = argparse.ArgumentParser(description="Fixed-point bit-width sweep + figures")
    ap.add_argument("--frames", type=int, default=15, help="frames per operating point")
    ap.add_argument("--seeds", type=int, default=10, help="Monte-Carlo seeds")
    ap.add_argument("--scenario",
                    default="scenarios/scenario_mimo_fixedpoint_dynrange.json")
    args = ap.parse_args()

    cfg, targets = load_scenario(args.scenario)
    stages = ["adc_bits", "cube_bits", "rd_bits", "mag_bits"]
    bit_grids = {"adc_bits": [6, 8, 10, 12, 14], "cube_bits": [6, 8, 10, 12, 16],
                 "rd_bits": [6, 8, 10, 12, 16],
                 "mag_bits": [6, 8, 10, 12, 16, 20, 24]}  # power-domain: needs ~20+
    results = [sweep_stage(cfg, targets, stage=s, bits=bit_grids[s],
                           n_frames=args.frames, n_seeds=args.seeds) for s in stages]
    _plot_sweep(results, "report/fixedpoint_evaluation")
    print(f"wrote figures to report/fixedpoint_evaluation/ "
          f"(frames={args.frames}, seeds={args.seeds})")


if __name__ == "__main__":
    main()
