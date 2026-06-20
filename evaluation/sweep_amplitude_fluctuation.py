"""Pd-vs-SNR sweep: steady vs Swerling-fluctuating targets.

Companion to sweep_amplitude.py (detection report §3): SAME scenario
(scenario_two_targets.json), same OS-CFAR lens, same CC clustering
(min_cells=2), same ±1-bin matching, same SNR axis. Differences:
  - frame_index ADVANCES so per-frame Swerling factors evolve (a frozen
    frame_index would reuse factor[0] forever); truth bins are recomputed
    per frame so the matcher follows the slow drift (0.1 m/s x 15 s =
    1.5 m over 300 frames, inside the unambiguous range; the simulator's
    SNR is range-independent, so Pd is stationary across the run).
  - The x-axis is the MEAN input SNR for fluctuating cases (Swerling
    factors are unit-mean-power).

Cases: steady / swerling1 / swerling3, all at frame_corr=0 — the classic
scan-to-scan-independent fluctuation. Frame correlation does not change the
marginal Pd, only its estimator variance; correlation belongs to the
tracker experiment (tracker_stress_swerling_sweep.py).

The steady curve doubles as an anchor check: it must reproduce detection
report §3's OS curve (Pd climbing ~0 -> 1 over roughly -35 -> -18 dB,
FA ~0.05/frame) within Monte Carlo noise.

Usage:
    python -m evaluation.sweep_amplitude_fluctuation --frames 300
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.sweep_amplitude import run_one_point
from simulator.beat_signal_generator import load_scenario

CASES = {
    "steady": None,
    "swerling1": {"model": "swerling1", "frame_corr": 0.0},
    "swerling3": {"model": "swerling3", "frame_corr": 0.0},
}
CFAR_METHOD = "os"
# Each (case, amplitude) point gets its own fade realisation so points are
# independent draws; base offset matches the RadarConfig default stream.
FLUCT_SEED_BASE = 77777


def run_sweep(
    scenario: Path,
    amplitudes: np.ndarray,
    n_frames: int,
) -> tuple[dict, dict, list]:
    radar_cfg, base_targets = load_scenario(str(scenario))
    n_targets = len(base_targets)
    pd_results = {(c, ti): [] for c in CASES for ti in range(n_targets)}
    fa_results = {c: [] for c in CASES}
    for case, fluct in CASES.items():
        for pi, scale in enumerate(amplitudes):
            cfg = dataclasses.replace(
                radar_cfg, fluctuation_seed=FLUCT_SEED_BASE + pi)
            targets = [
                dataclasses.replace(t, amplitude=t.amplitude * scale,
                                    fluctuation=fluct)
                for t in base_targets
            ]
            pd, fa = run_one_point(cfg, targets, CFAR_METHOD, n_frames,
                                   advance_frame_index=True)
            for ti, t in enumerate(targets):
                snr_db = (20.0 * np.log10(t.amplitude)
                          - 10.0 * np.log10(cfg.noise_power))
                pd_results[(case, ti)].append((snr_db, float(pd[ti])))
            avg_snr = float(np.mean([
                20.0 * np.log10(t.amplitude)
                - 10.0 * np.log10(cfg.noise_power)
                for t in targets
            ]))
            fa_results[case].append((avg_snr, fa))
            print(f"  {case:9s} scale={scale:8.2e} "
                  f"Pd={pd.round(2).tolist()} FA/frame={fa:.3f}")
    return pd_results, fa_results, base_targets


def snr_at_pd(points: list, level: float) -> float:
    """Interpolated SNR (dB) at which the Pd curve first reaches `level`.

    Monotonicises the curve with a running max so np.interp is valid;
    returns NaN when the curve never reaches `level`.
    """
    pts = sorted(points)
    snr = np.array([p[0] for p in pts])
    pd = np.maximum.accumulate(np.array([p[1] for p in pts]))
    if pd.max() < level:
        return float("nan")
    return float(np.interp(level, pd, snr))


def plot(pd_results, fa_results, base_targets, out: Path) -> None:
    fig, (ax_pd, ax_fa) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"steady": "tab:blue", "swerling1": "tab:red",
              "swerling3": "tab:green"}
    linestyles = ["-", "--"]
    for case in CASES:
        for ti, t in enumerate(base_targets):
            pts = sorted(pd_results[(case, ti)])
            ax_pd.plot([p[0] for p in pts], [p[1] for p in pts],
                       color=colors[case], linestyle=linestyles[ti % 2],
                       marker="o", markersize=4, linewidth=1.4,
                       label=f"{case}  target@{t.range_m:.0f}m")
        pts = sorted(fa_results[case])
        ax_fa.plot([p[0] for p in pts], [p[1] for p in pts],
                   color=colors[case], marker="s", markersize=4,
                   linewidth=1.4, label=case)
    ax_pd.set_ylabel("Pd")
    ax_pd.set_ylim(-0.05, 1.05)
    ax_pd.set_title("Pd vs mean input SNR — steady vs Swerling "
                    "(OS-CFAR, Pfa=1e-4, min_cells=2)")
    ax_pd.grid(True, alpha=0.3)
    ax_pd.legend(fontsize=7, ncol=2, loc="lower right")
    ax_fa.set_xlabel("mean input SNR per sample [dB]")
    ax_fa.set_ylabel("false alarms / frame")
    ax_fa.grid(True, alpha=0.3)
    ax_fa.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def write_csv(pd_results, fa_results, n_targets: int,
              amplitudes: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "target", "amplitude_scale", "snr_db", "pd",
                    "fa_per_frame"])
        for case in CASES:
            for pi, scale in enumerate(amplitudes):
                fa = fa_results[case][pi][1]
                for ti in range(n_targets):
                    snr_db, pd = pd_results[(case, ti)][pi]
                    w.writerow([case, ti, f"{scale:.6e}", f"{snr_db:.2f}",
                                f"{pd:.4f}", f"{fa:.4f}"])
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=Path,
                   default=Path("scenarios/scenario_two_targets.json"))
    p.add_argument("--frames", type=int, default=300,
                   help="frames per (case, amplitude) point; fluctuation "
                        "adds variance, so more than the baseline's 100")
    p.add_argument("--out", type=Path,
                   default=Path("report/sweep_amplitude_fluctuation.png"))
    p.add_argument("--csv", type=Path,
                   default=Path("report/sweep_amplitude_fluctuation.csv"))
    args = p.parse_args()

    # Same 10-point grid as sweep_amplitude.py (±20 dB around A_50%).
    amplitudes = np.logspace(np.log10(3.5e-5), np.log10(3.5e-3), 10)

    pd_results, fa_results, base_targets = run_sweep(
        args.scenario, amplitudes, args.frames)
    n_targets = len(base_targets)

    print("\nHeadline SNR thresholds (dB):")
    for ti in range(n_targets):
        s50 = snr_at_pd(pd_results[("steady", ti)], 0.5)
        s90 = snr_at_pd(pd_results[("steady", ti)], 0.9)
        print(f"  target {ti} steady   : SNR@Pd0.5 {s50:6.1f}  "
              f"SNR@Pd0.9 {s90:6.1f}")
        for case in ("swerling1", "swerling3"):
            c50 = snr_at_pd(pd_results[(case, ti)], 0.5)
            c90 = snr_at_pd(pd_results[(case, ti)], 0.9)
            print(f"  target {ti} {case:9s}: SNR@Pd0.5 {c50:6.1f} "
                  f"(Δ{c50 - s50:+.1f})  SNR@Pd0.9 {c90:6.1f} "
                  f"(Δ{c90 - s90:+.1f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(pd_results, fa_results, base_targets, args.out)
    write_csv(pd_results, fa_results, n_targets, amplitudes, args.csv)


if __name__ == "__main__":
    main()
