"""Tracking facet: end-to-end CTRV-EKF tracking evaluation.

One generic core — `track_scenario` runs the MIMO pipeline + CartesianTracker on
any scenario and returns per-target metrics. Presentation is chosen by target
count: a single target gets a BEV + state-vs-frame plot (state ground truth is
finite-differenced from the GT trajectory, so it works for *any* motion model);
multiple targets get an x/y-vs-frame plot (never degenerate, unlike a BEV when
targets are collinear). So new scenarios — more targets, other motion models —
drop in without code changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.config import CartesianTrackerConfig
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from evaluation.mimo_pipeline import (
    run_pipeline_mimo, build_gt_xy, associate_xy, track_rmse_xy)

OUT = ROOT / "report" / "mimo_evaluation"


def _tracker_cfg(frame_period):
    """Tracker config with predict dt = frame period (so the state is SI: v in
    m/s, ω in rad/s, consistent with the m/s Doppler) and process noise sized for
    the frame rate (the dataclass defaults are the per-step dt=1 convention)."""
    return CartesianTrackerConfig(dt=frame_period, sigma_a=5.0,
                                  sigma_yaw_accel=2.0)


# --------------------------------------------------------------------------- #
# Generic core                                                                #
# --------------------------------------------------------------------------- #

def track_scenario(cfg, targets, n_frames, tracker_cfg=None, max_dist_m=2.5):
    """Run the MIMO pipeline + CartesianTracker on a scenario (any #targets,
    any motion model). Returns dict: gt, history, assoc, rmse_m (aggregate),
    per_gt {gid: {rmse_m, n_track_ids}}."""
    gen = FMCWBeatSignalGenerator(cfg)
    gt = build_gt_xy(targets, gen, n_frames)
    tc = tracker_cfg or _tracker_cfg(cfg.frame_period_s)
    history = run_pipeline_mimo(cfg, targets, n_frames, tracker_config=tc)
    assoc = [(fi, associate_xy(rows, snaps, max_dist_m))
             for (fi, rows), (_, snaps) in zip(gt, history)]
    return dict(gt=gt, history=history, assoc=assoc,
                rmse_m=track_rmse_xy(gt, history, assoc),
                per_gt=_per_target_metrics(gt, history, assoc))


def _per_target_metrics(gt, history, assoc):
    """Per-GT position RMSE and distinct-track-ID count (ID-continuity / swap
    detector). Scales to any number of targets."""
    snaps_by_frame = {fi: {s.id: s for s in snaps} for fi, snaps in history}
    out = {}
    for gid, *_ in gt[0][1]:
        sq, ids = [], set()
        for (fi, rows), (_, mapping) in zip(gt, assoc):
            tid = mapping.get(gid)
            if tid is None:
                continue
            ids.add(tid)
            s = snaps_by_frame[fi].get(tid)
            row = next(r for r in rows if r[0] == gid)
            if s is not None:
                sq.append((s.state[0] - row[1]) ** 2 + (s.state[1] - row[2]) ** 2)
        out[gid] = dict(rmse_m=float(np.sqrt(np.mean(sq))) if sq else float("nan"),
                        n_track_ids=len(ids))
    return out


def run_turn_eval(cfg, targets, n_frames=30):
    """Single-target turn study (thin wrapper over track_scenario; adds the
    turn-specific final-ω readout). Kept for the turn test."""
    res = track_scenario(cfg, targets, n_frames)
    omega = float("nan")
    tid = res["assoc"][-1][1].get(0)
    if tid is not None:
        for s in res["history"][-1][1]:
            if s.id == tid:
                omega = float(s.state[4])
    res["omega_frame_final"] = omega
    res["n_track_ids"] = res["per_gt"][0]["n_track_ids"]
    return res


# --------------------------------------------------------------------------- #
# Presentation (chosen by target count)                                       #
# --------------------------------------------------------------------------- #

def _gt_xy(gt, gid):
    gx = [next(r for r in rows if r[0] == gid)[1] for _, rows in gt]
    gy = [next(r for r in rows if r[0] == gid)[2] for _, rows in gt]
    return np.array(gx), np.array(gy)


def _track_owning(history, assoc, gid):
    """(frames, xs, ys, states) of the GT-gid-owning track over the run."""
    snaps_by_frame = {fi: {s.id: s for s in snaps} for fi, snaps in history}
    f, xs, ys, st = [], [], [], []
    for fi, mapping in assoc:
        tid = mapping.get(gid)
        s = snaps_by_frame[fi].get(tid) if tid is not None else None
        if s is not None:
            f.append(fi); xs.append(s.state[0]); ys.append(s.state[1]); st.append(s.state)
    return f, xs, ys, st


def _bev(gt, history, assoc, out, title):
    f, tx, ty, _ = _track_owning(history, assoc, 0)
    gx, gy = _gt_xy(gt, 0)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(gx, gy, "k--", lw=1.5, label="GT")
    ax.plot(tx, ty, "C1.-", label="track")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


def _true_state_series(target, frames, dt):
    """True (v [m/s], ψ [deg], ω [rad/s]) at each frame from the target's motion
    model. range_at/azimuth_at give exact (x,y) at any *continuous* frame, so this
    is motion-model agnostic and free of the endpoint artifacts a finite-difference
    of the (frame-center/frame-start-mixed) GT trajectory would introduce."""
    from simulator.motion_models import range_at, azimuth_at

    def pos(f):
        r, _ = range_at(target, f, dt)
        th = azimuth_at(target, f, dt)
        return np.array([r * np.cos(th), r * np.sin(th)])

    def heading_speed(f, h=1e-3):
        vel = (pos(f + h) - pos(f - h)) / (2.0 * h * dt)
        return np.hypot(vel[0], vel[1]), np.arctan2(vel[1], vel[0])

    v_s, psi_s, w_s = [], [], []
    for fi in frames:
        s, psi = heading_speed(fi)
        _, pp = heading_speed(fi + 0.5)
        _, pm = heading_speed(fi - 0.5)
        v_s.append(s); psi_s.append(np.degrees(psi))
        w_s.append(((pp - pm + np.pi) % (2 * np.pi) - np.pi) / dt)   # ω over 1 frame
    return np.array(v_s), np.array(psi_s), np.array(w_s)


def _state_vs_frame(target, history, assoc, gid, dt, out, title):
    """Estimate vs GT of (v, ψ, ω) over frames for the gid-owning track. GT comes
    from the target's motion model (any model), so no endpoint artifacts."""
    frames = [fi for fi, _ in assoc]
    v_g, psi_g, w_g = _true_state_series(target, frames, dt)
    fe, v_e, psi_e, w_e = [], [], [], []
    for fi, _, _, st in zip(*_track_owning(history, assoc, gid)):
        fe.append(fi); v_e.append(st[2]); psi_e.append(np.degrees(st[3])); w_e.append(st[4])
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].plot(frames, v_g, "k--", label="GT"); axs[0].plot(fe, v_e, "C1.-", label="est")
    axs[0].set(ylabel="speed v [m/s]"); axs[0].grid(alpha=0.3); axs[0].legend()
    axs[1].plot(frames, psi_g, "k--"); axs[1].plot(fe, psi_e, "C1.-")
    axs[1].set(ylabel="heading ψ [deg]"); axs[1].grid(alpha=0.3)
    axs[2].plot(frames, w_g, "k--", label="GT"); axs[2].plot(fe, w_e, "C1.-", label="est")
    axs[2].set(xlabel="frame", ylabel="turn rate ω [rad/s]")
    axs[2].grid(alpha=0.3); axs[2].legend()
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


def _xy_vs_frame(gt, history, assoc, out, title):
    """x and y vs frame for all GT + every confirmed track (coloured by id).
    Never degenerate (a BEV collapses when targets are collinear), and shows
    crossings + ID continuity for any number of targets."""
    frames = [fi for fi, _ in gt]
    n = len(gt[0][1])
    fig, (axx, axy) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for gid in range(n):
        gx, gy = _gt_xy(gt, gid)
        axx.plot(frames, gx, "--", lw=1.3, label=f"GT {gid}")
        axy.plot(frames, gy, "--", lw=1.3)
    tx, ty = {}, {}
    for fi, snaps in history:
        for s in snaps:
            if s.status == "confirmed":
                tx.setdefault(s.id, []).append((fi, s.state[0]))
                ty.setdefault(s.id, []).append((fi, s.state[1]))
    for tid in sorted(tx):
        axx.plot([p[0] for p in tx[tid]], [p[1] for p in tx[tid]], ".-", ms=3,
                 label=f"track {tid}")
        axy.plot([p[0] for p in ty[tid]], [p[1] for p in ty[tid]], ".-", ms=3)
    axx.set(ylabel="x [m]", title=title); axx.grid(alpha=0.3); axx.legend(fontsize=8)
    axy.set(xlabel="frame", ylabel="y [m]"); axy.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


def present(res, targets, cfg, stem, title):
    """Single target → BEV + state-vs-frame; ≥2 targets → x/y-vs-frame."""
    if len(targets) == 1:
        _bev(res["gt"], res["history"], res["assoc"], OUT / f"{stem}.png",
             f"{title} (RMSE {res['rmse_m']:.2f} m)")
        _state_vs_frame(targets[0], res["history"], res["assoc"], 0,
                        cfg.frame_period_s, OUT / f"{stem}_states.png",
                        f"{title}: state estimates (v, ψ, ω) vs ground truth")
    else:
        _xy_vs_frame(res["gt"], res["history"], res["assoc"], OUT / f"{stem}.png",
                     f"{title}: x / y vs frame")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _run(scenario_path, n_frames, stem, title):
    cfg, targets = load_scenario(str(scenario_path))
    res = track_scenario(cfg, targets, n_frames)
    print(f"{stem}: rmse={res['rmse_m']:.2f} m  per_gt={res['per_gt']}")
    present(res, targets, cfg, stem, title)


# Figure title and default frame count per scenario (keyed by the figure stem,
# `mimo_tracking_<suffix>`, where <suffix> is the scenario file name minus its
# `scenario_mimo_` prefix). The figure name is scenario-derived but keeps the
# `mimo_tracking_` grouping, so `--scenario X --frames N` reproduces the report
# figure and self-documents which scene it came from. Unknown scenarios fall back
# to a derived title and 60 frames.
_TITLES = {"mimo_tracking_turn": "Tracking a turn",
           "mimo_tracking_angle_crossing": "Tracking: azimuth crossing",
           "mimo_tracking_crossing": "Tracking: crossing targets"}
_FRAMES = {"mimo_tracking_turn": 60, "mimo_tracking_angle_crossing": 80,
           "mimo_tracking_crossing": 80}
_REPORT_SCENARIOS = ["scenario_mimo_turn.json", "scenario_mimo_angle_crossing.json"]


def _stem_for(scenario_path) -> str:
    base = Path(scenario_path).stem
    for prefix in ("scenario_", "mimo_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    return "mimo_tracking_" + base


def main():
    p = argparse.ArgumentParser(description="MIMO tracking evaluation figures")
    p.add_argument("--scenario", default=None,
                   help="run one scenario (default: reproduce all report figures)")
    p.add_argument("--frames", type=int, default=None)
    args = p.parse_args()

    paths = ([args.scenario] if args.scenario
             else [ROOT / "scenarios" / s for s in _REPORT_SCENARIOS])
    for sc in paths:
        stem = _stem_for(sc)
        title = _TITLES.get(stem, stem.replace("mimo_tracking_", "").replace("_", " "))
        _run(sc, args.frames or _FRAMES.get(stem, 60), stem, title)


if __name__ == "__main__":
    main()
