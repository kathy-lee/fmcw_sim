"""Tracking facet (M5): end-to-end CTRV-EKF on multi-target + turn scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.beat_signal_generator import FMCWBeatSignalGenerator
from evaluation.mimo_pipeline import (
    run_pipeline_mimo, build_gt_xy, associate_xy, track_rmse_xy)

OUT = ROOT / "report" / "mimo_evaluation"


def _assoc_history(gt_traj, history, max_dist_m=2.5):
    return [(fi, associate_xy(rows, snaps, max_dist_m))
            for (fi, rows), (_, snaps) in zip(gt_traj, history)]


def run_turn_eval(cfg, targets, n_frames=30):
    """Track the ct_xy turn target; return RMSE + final ω estimate (rad/frame)."""
    gen = FMCWBeatSignalGenerator(cfg)
    gt = build_gt_xy(targets, gen, n_frames)
    history = run_pipeline_mimo(cfg, targets, n_frames)
    assoc = _assoc_history(gt, history)
    rmse = track_rmse_xy(gt, history, assoc)
    # distinct track IDs ever associated with GT 0 — track-ID stability.
    ids = {m.get(0) for _, m in assoc if m.get(0) is not None}
    # final ω of the GT-owning track (reported, but only weakly observable).
    omega = float("nan")
    tid = assoc[-1][1].get(0)
    if tid is not None:
        for s in history[-1][1]:
            if s.id == tid:
                omega = float(s.state[4])
    return {"rmse_m": rmse, "omega_frame_final": omega, "n_track_ids": len(ids),
            "gt": gt, "history": history, "assoc": assoc}


def _bev(gt, history, assoc, out, title):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    gx = [rows[0][1] for _, rows in gt]; gy = [rows[0][2] for _, rows in gt]
    ax.plot(gx, gy, "k--", lw=1.5, label="GT")
    tx, ty = [], []
    for (_, snaps), (_, m) in zip(history, assoc):
        tid = m.get(0)
        for s in snaps:
            if s.id == tid:
                tx.append(s.state[0]); ty.append(s.state[1])
    ax.plot(tx, ty, "C1.-", label="track")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title); ax.axis("equal")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


def main():
    from simulator.beat_signal_generator import load_scenario
    # turn
    cfg, tg = load_scenario(str(ROOT / "scenarios/scenario_mimo_turn.json"))
    res = run_turn_eval(cfg, tg, n_frames=35)
    print(f"turn: rmse={res['rmse_m']:.2f} m  omega_final={res['omega_frame_final']:.4f}")
    _bev(res["gt"], res["history"], res["assoc"],
         OUT / "mimo_tracking_turn.png",
         f"Tracking a turn (RMSE {res['rmse_m']:.2f} m)")
    
    # multi-target crossing
    cfg2, tg2 = load_scenario(str(ROOT / "scenarios/scenario_mimo_crossing.json"))
    gen2 = FMCWBeatSignalGenerator(cfg2)
    gt2 = build_gt_xy(tg2, gen2, 80)
    hist2 = run_pipeline_mimo(cfg2, tg2, 80)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for gid in range(len(tg2)):
        ax.plot([r[gid][1] for _, r in gt2], [r[gid][2] for _, r in gt2],
                "--", lw=1.2, label=f"GT {gid}")
    for fi, snaps in hist2:
        for s in snaps:
            if s.status == "confirmed":
                ax.plot(s.state[0], s.state[1], "C1.", ms=3)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Tracking: crossing targets")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    out = OUT / "mimo_tracking_crossing.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
