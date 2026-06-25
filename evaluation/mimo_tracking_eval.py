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

from processing.config import CartesianTrackerConfig
from simulator.beat_signal_generator import FMCWBeatSignalGenerator
from evaluation.mimo_pipeline import (
    run_pipeline_mimo, build_gt_xy, associate_xy, track_rmse_xy)

OUT = ROOT / "report" / "mimo_evaluation"


def _tracker_cfg(frame_period):
    """Tracker config with predict dt = frame period (so the state is SI: v in
    m/s, ω in rad/s, consistent with the m/s Doppler) and process noise sized for
    the frame rate (the dataclass defaults are the per-step dt=1 convention)."""
    return CartesianTrackerConfig(dt=frame_period, sigma_a=5.0,
                                  sigma_yaw_accel=2.0)


def _assoc_history(gt_traj, history, max_dist_m=2.5):
    return [(fi, associate_xy(rows, snaps, max_dist_m))
            for (fi, rows), (_, snaps) in zip(gt_traj, history)]


def run_turn_eval(cfg, targets, n_frames=30):
    """Track the ct_xy turn target; return RMSE + final ω estimate (rad/frame)."""
    gen = FMCWBeatSignalGenerator(cfg)
    gt = build_gt_xy(targets, gen, n_frames)
    history = run_pipeline_mimo(cfg, targets, n_frames,
                                tracker_config=_tracker_cfg(cfg.frame_period_s))
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


def _state_vs_frame(history, assoc, target, frame_period, out, title):
    """Estimate vs ground truth of the non-position states (v, ψ, ω) over frames,
    for the GT-owning track. GT v/ψ come from the ct_xy motion model; GT ω is the
    scenario turn rate (rad/frame)."""
    from simulator.motion_models import _ct_xy_state
    w_true = target.motion_params.get("turn_rate", 0.0)   # rad/s (state ω is SI)
    frames, v_e, psi_e, w_e, v_g, psi_g = [], [], [], [], [], []
    for (fi, snaps), (_, m) in zip(history, assoc):
        snap = next((s for s in snaps if s.id == m.get(0)), None)
        if snap is None:
            continue
        _, _, vx, vy = _ct_xy_state(target, fi * frame_period)
        frames.append(fi)
        v_e.append(snap.state[2]); psi_e.append(np.degrees(snap.state[3]))
        w_e.append(snap.state[4])
        v_g.append(np.hypot(vx, vy)); psi_g.append(np.degrees(np.arctan2(vy, vx)))
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].plot(frames, v_g, "k--", label="GT"); axs[0].plot(frames, v_e, "C1.-", label="est")
    axs[0].set(ylabel="speed v [m/s]"); axs[0].grid(alpha=0.3); axs[0].legend()
    axs[1].plot(frames, psi_g, "k--"); axs[1].plot(frames, psi_e, "C1.-")
    axs[1].set(ylabel="heading ψ [deg]"); axs[1].grid(alpha=0.3)
    axs[2].axhline(w_true, color="k", ls="--", label="GT")
    axs[2].plot(frames, w_e, "C1.-", label="est")
    axs[2].set(xlabel="frame", ylabel="turn rate ω [rad/s]")
    axs[2].grid(alpha=0.3); axs[2].legend()
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


def main():
    from simulator.beat_signal_generator import load_scenario
    # turn
    cfg, tg = load_scenario(str(ROOT / "scenarios/scenario_mimo_turn.json"))
    res = run_turn_eval(cfg, tg, n_frames=60)
    print(f"turn: rmse={res['rmse_m']:.2f} m  omega_final={res['omega_frame_final']:.4f}")
    _bev(res["gt"], res["history"], res["assoc"],
         OUT / "mimo_tracking_turn.png",
         f"Tracking a turn (RMSE {res['rmse_m']:.2f} m)")
    _state_vs_frame(res["history"], res["assoc"], tg[0], cfg.frame_period_s,
                    OUT / "mimo_tracking_turn_states.png",
                    "Turn: state estimates (v, ψ, ω) vs ground truth")

    # multi-target crossing
    cfg2, tg2 = load_scenario(str(ROOT / "scenarios/scenario_mimo_crossing.json"))
    gen2 = FMCWBeatSignalGenerator(cfg2)
    gt2 = build_gt_xy(tg2, gen2, 80)
    hist2 = run_pipeline_mimo(cfg2, tg2, 80,
                              tracker_config=_tracker_cfg(cfg2.frame_period_s))
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
