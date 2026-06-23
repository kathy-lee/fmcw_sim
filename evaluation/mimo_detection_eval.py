"""Detection facet (M5): object Pd vs SNR, single-channel vs V-channel (NCI gain)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.cfar import cfar_2d
from processing.mimo_frame import _CFAR_KW
from processing.point_cloud import integrate
from processing.range_doppler import range_doppler
from simulator.beat_signal_generator import FMCWBeatSignalGenerator


def _detected_near_gt(mask, gt_r_bin, gt_d_bin, tol_r=3, tol_d=3):
    """True if any detection lies within tol of the GT cell in BOTH range and
    Doppler. The 2-D match (not range-only) suppresses scattered false alarms
    that would otherwise inflate the apparent Pd at low SNR."""
    if not mask.any():
        return False
    dd, rr = np.where(mask)
    return bool(np.any((np.abs(rr - gt_r_bin) <= tol_r)
                       & (np.abs(dd - gt_d_bin) <= tol_d)))


def pd_vs_snr(cfg, targets, noise_powers, n_frames=8):
    """Object Pd at each noise level for single-channel (channel 0, n_integrate=1)
    vs V-channel non-coherent integration. Returns (snr_db, pd_single, pd_integ).

    A detection counts only if a CFAR cell falls near the GT in both range and
    Doppler (2-D cell match via the range/velocity axes)."""
    snr_db, pd_single, pd_integ = [], [], []
    for npow in noise_powers:
        c = dataclasses.replace(cfg, noise_power=npow)
        gen = FMCWBeatSignalGenerator(c)
        range_axis = gen.range_axis_m()
        vel_axis = gen.velocity_axis_mps() / c.n_tx
        hit_s = hit_i = 0
        for fi in range(n_frames):
            # GT cell per frame — the target moves, so its (range, Doppler) bin
            # drifts across the run.
            gt = gen.ground_truth(targets, frame_index=fi)[0]
            gt_r_bin = int(np.argmin(np.abs(range_axis - gt["range_m"])))
            gt_d_bin = int(np.argmin(np.abs(vel_axis - gt["velocity_mps"])))
            frame = gen.generate_frame_mimo(targets, frame_index=fi)
            rd = np.stack([range_doppler(frame[v]) for v in range(frame.shape[0])])
            # single channel
            sdb = 10.0 * np.log10(np.abs(rd[0]) ** 2 + 1e-12)
            m_s = cfar_2d(sdb, method="ca", n_integrate=1, **_CFAR_KW)
            m_s[:, :6] = False
            hit_s += _detected_near_gt(m_s, gt_r_bin, gt_d_bin)
            # V-channel integration
            idb = 10.0 * np.log10(integrate(rd) + 1e-12)
            m_i = cfar_2d(idb, method="ca", n_integrate=frame.shape[0], **_CFAR_KW)
            m_i[:, :6] = False
            hit_i += _detected_near_gt(m_i, gt_r_bin, gt_d_bin)
        snr_db.append(10.0 * np.log10(targets[0].amplitude ** 2 / npow))
        pd_single.append(hit_s / n_frames)
        pd_integ.append(hit_i / n_frames)
    return snr_db, pd_single, pd_integ


def main():
    from simulator.beat_signal_generator import load_scenario
    cfg, targets = load_scenario(str(ROOT / "scenarios/scenario_mimo_2d.json"))
    # Noise regime where both curves span 0->1 (target amplitude is 0.01).
    noise = [0.002, 0.006, 0.02, 0.06, 0.2, 0.6, 2.0]
    snr, pds, pdi = pd_vs_snr(cfg, targets, noise, n_frames=20)
    order = np.argsort(snr)
    snr = np.array(snr)[order]; pds = np.array(pds)[order]; pdi = np.array(pdi)[order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr, pds, "o-", label="single channel (V=1)")
    ax.plot(snr, pdi, "s-", label=f"integrated (V={cfg.n_tx * cfg.n_rx})")
    ax.set(xlabel="per-channel SNR [dB]", ylabel="object Pd",
           title="Detection: non-coherent integration gain", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3); ax.legend()
    out = ROOT / "report/mimo_evaluation/mimo_detection.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
