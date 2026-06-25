"""DoA facet: azimuth accuracy vs SNR and angular resolution (4 methods).

Uses controlled synthetic array snapshots (the standard DoA benchmark — isolates
angle estimation from detection), reusing processing/doa.py on a lambda/2 ULA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.doa import estimate_doa

METHODS = ("bartlett", "capon", "music", "esprit")


def _snapshot(pos, az_deg_list, snr_db, rng):
    """One array snapshot: unit sources at the given azimuths + complex noise."""
    a = np.deg2rad(np.atleast_1d(az_deg_list))
    A = np.exp(1j * 2 * np.pi * np.outer(pos, np.sin(a)))      # (n_elem, n_src)
    x = A @ np.ones(len(a))
    sigma = 10 ** (-snr_db / 20.0)
    noise = sigma * (rng.standard_normal(len(pos)) +
                     1j * rng.standard_normal(len(pos))) / np.sqrt(2)
    return x + noise


def azimuth_rmse_vs_snr(az_true_deg, n_elem, snr_db_list, methods=METHODS,
                        n_trials=50, seed=0):
    """Azimuth RMSE (deg) per method over Monte-Carlo trials at each SNR."""
    pos = 0.5 * np.arange(n_elem)
    rng = np.random.default_rng(seed)
    rmse = {m: [] for m in methods}
    for snr in snr_db_list:
        errs = {m: [] for m in methods}
        for _ in range(n_trials):
            x = _snapshot(pos, az_true_deg, snr, rng)
            for m in methods:
                est = estimate_doa(x, pos, method=m, n_sources=1)
                if len(est):
                    errs[m].append((est[0] - az_true_deg) ** 2)
        for m in methods:
            rmse[m].append(float(np.sqrt(np.mean(errs[m]))) if errs[m]
                           else float("nan"))
    return list(snr_db_list), rmse


def min_resolvable_separation(n_elem, seps_deg, snr_db=20.0, methods=METHODS,
                              n_trials=20, seed=0, center_deg=0.0):
    """For each method, the smallest separation (deg) at which two equal sources
    are resolved in >=80% of trials. Returns {method: separation|nan}."""
    pos = 0.5 * np.arange(n_elem)
    rng = np.random.default_rng(seed)
    out = {}
    for m in methods:
        resolved_sep = float("nan")
        for sep in seps_deg:
            azs = [center_deg - sep / 2, center_deg + sep / 2]
            ok = 0
            for _ in range(n_trials):
                x = _snapshot(pos, azs, snr_db, rng)
                est = np.sort(estimate_doa(x, pos, method=m, n_sources=2))
                if len(est) == 2 and abs(est[1] - est[0] - sep) < 0.5 * sep:
                    ok += 1
            if ok / n_trials >= 0.8:
                resolved_sep = sep
                break
        out[m] = resolved_sep
    return out


def resolution_vs_snr(n_elem, seps_deg, snr_list, methods=METHODS, n_trials=40,
                      seed=0):
    """Min resolvable separation (deg) per method at each SNR.
    Returns (snr_list, {method: [sep|nan per snr]})."""
    out = {m: [] for m in methods}
    for snr in snr_list:
        res = min_resolvable_separation(n_elem, seps_deg, snr_db=snr,
                                        methods=methods, n_trials=n_trials,
                                        seed=seed)
        for m in methods:
            out[m].append(res[m])
    return list(snr_list), out


def main():
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    # Off-grid source (12.4°) so the grid methods show their real quantization
    # floor instead of snapping to exactly-zero error (which log-scale can't plot).
    snr, rmse = azimuth_rmse_vs_snr(12.4, 12, snr_list, n_trials=100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        ax.semilogy(snr, rmse[m], "o-", label=m)
    ax.set(xlabel="SNR [dB]", ylabel="azimuth RMSE [deg]",
           title="DoA accuracy vs SNR (source at 12.4°, 12-elem ULA)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    out1 = ROOT / "report/mimo_evaluation/mimo_doa_accuracy.png"
    fig.tight_layout(); fig.savefig(out1, dpi=120); print("wrote", out1)

    seps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
    snr_r, res = resolution_vs_snr(12, seps, [0, 5, 10, 15, 20, 25, 30],
                                   n_trials=40)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        ax2.plot(snr_r, res[m], "o-", label=m)
    ax2.set(xlabel="SNR [dB]", ylabel="min resolvable separation [deg]",
            title="Angular resolution vs SNR (two equal sources, 12-elem ULA)")
    ax2.grid(alpha=0.3); ax2.legend()
    out2 = ROOT / "report/mimo_evaluation/mimo_doa_resolution.png"
    fig2.tight_layout(); fig2.savefig(out2, dpi=120); print("wrote", out2)


if __name__ == "__main__":
    main()
