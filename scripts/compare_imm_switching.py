"""Overlay KF, EKF, and IMM tracks on a piecewise (CV→drag) switching scenario.

Scenario-driven: pass any scenario whose single target uses the `piecewise`
motion model with a constant-velocity leg followed by a drag leg. The script
reads the initial velocity, the regime-boundary frame, the drag coefficient, and
the frame count straight from the scenario, so the tracker's drag model is always
matched to the target's and the plot title always reflects the actual scenario.

Examples (run from the repo root):
    python -m scripts.compare_imm_switching \
        --scenario scenarios/scenario_imm_switching.json           # gentle base
    python -m scripts.compare_imm_switching                        # amplified v2
    python -m scripts.compare_imm_switching --forced_coast         # forced coast

`range_meas_var` is held at 5.0 to keep the Kalman gain low, so the predict step
(where the motion models differ) carries weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import IMMConfig, MotionModelSpec, TrackerConfig
from processing.imm import IMMTracker
from processing.range_doppler import range_doppler
from processing.tracking import Tracker, drag_k_metric_to_bins
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)
from simulator.motion_models import range_at


DEFAULT_SCENARIO = "scenarios/scenario_imm_switching_v2.json"
MAX_COAST = 15
RANGE_MEAS_VAR = 5.0  # lowers Kalman gain → predict step weighs more
# Forced coast windows: one in the CV phase, one in the drag phase. Each
# window forces predict-only behaviour so KF/EKF/IMM divergence reflects
# motion-model choice, not stochastic CFAR misses. Enabled via --forced_coast.
FORCED_COAST_WINDOW = set(range(20, 28)) | set(range(50, 58))


def read_piecewise_motion(target):
    """Pull (v0, transition_frame, drag_k_metric, n_frames) from a target whose
    motion_model is `piecewise` (a constant-velocity leg then a drag leg)."""
    v0 = float(target.velocity_mps)
    phases = (target.motion_params or {}).get("phases", [])
    drag_phase = next(
        (p for p in phases if p.get("motion_model") == "drag"), None
    )
    if drag_phase is None:
        raise ValueError(
            "scenario target has no 'drag' phase; this script expects a "
            "piecewise CV→drag target"
        )
    transition_frame = int(drag_phase["start_frame"])
    drag_k = float(
        drag_phase.get("motion_params", {}).get("drag_coefficient_k", 0.0)
    )
    n_frames = int(max(p["end_frame"] for p in phases))
    return v0, transition_frame, drag_k, n_frames


def default_plot_path(scenario_path: str) -> Path:
    """Derive the output PNG path from the scenario filename, e.g.
    scenarios/scenario_imm_switching_v2.json
        -> docs/tracker_stress_imm_switching_v2.png"""
    name = Path(scenario_path).stem.replace("scenario_", "", 1)
    return Path("docs") / f"tracker_stress_{name}.png"


def run(tracker, gen, initial_targets, range_per_bin, radar_cfg,
        forced_coast_frames, n_frames):
    frames_data = []
    for frame_idx in range(n_frames):
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", pfa=1e-4, window="hann",
                       dc_removal=True, clutter_removal=False)
        mask[:, 0] = False
        dets = cluster(mask, rd_db, method="peak", min_cells=2,
                       peak_neighborhood=3)
        if frame_idx in forced_coast_frames:
            dets = []
        tracker.step(dets)
        if not tracker.tracks:
            continue
        truth_r, _ = range_at(initial_targets[0], frame_idx,
                              radar_cfg.frame_period_s)
        candidates = [t for t in tracker.tracks
                      if t.status in ("confirmed", "coasting")]
        if not candidates:
            continue
        best = min(candidates,
                   key=lambda t: abs(float(t.state[0]) * range_per_bin
                                       - truth_r))
        best_pos_m = float(best.state[0]) * range_per_bin
        if abs(best_pos_m - truth_r) > 3.0:
            continue
        mode_probs = (
            tuple(best.mode_probs) if hasattr(best, "mode_probs") else None
        )
        frames_data.append({
            "frame": frame_idx,
            "pos_m": best_pos_m,
            "status": best.status,
            "mode_probs": mode_probs,
        })
    return frames_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="piecewise CV→drag scenario JSON "
                             f"(default: {DEFAULT_SCENARIO})")
    parser.add_argument("--out", default=None,
                        help="output PNG path (default: derived from scenario)")
    parser.add_argument("--forced_coast", action="store_true", default=False,
                        help="force predict-only on frames 20-27 and 50-57")
    args = parser.parse_args()

    radar_cfg, initial_targets = load_scenario(args.scenario)
    v0, transition_frame, drag_k, n_frames = read_piecewise_motion(
        initial_targets[0]
    )

    out = Path(args.out) if args.out else default_plot_path(args.scenario)
    if args.forced_coast:
        forced_coast_frames = FORCED_COAST_WINDOW
        out = out.with_stem(out.stem + "_forced_coast")
    else:
        forced_coast_frames = set()

    gen = FMCWBeatSignalGenerator(radar_cfg)
    range_per_bin = float(gen.range_axis_m()[1] - gen.range_axis_m()[0])
    # Match the tracker's drag model to the target's: `drag_k` is read straight
    # from the scenario's drag-phase `drag_coefficient_k`
    # (read_piecewise_motion), so the filter's assumed coefficient always
    # equals the value in the scenario JSON — no hand-set constant to keep in
    # sync. We only convert metric → bin units for the filter here.
    k_bins = drag_k_metric_to_bins(drag_k, range_per_bin)

    gen_kf = FMCWBeatSignalGenerator(radar_cfg)
    kf_tracker = Tracker(TrackerConfig(method="gnn", motion_model="cv",
                                        max_coast=MAX_COAST,
                                        range_meas_var=RANGE_MEAS_VAR))
    kf_data = run(kf_tracker, gen_kf, initial_targets, range_per_bin, radar_cfg,
                  forced_coast_frames, n_frames)

    gen_ekf = FMCWBeatSignalGenerator(radar_cfg)
    ekf_tracker = Tracker(TrackerConfig(
        method="gnn", motion_model="drag", drag_coefficient_k=k_bins,
        max_coast=MAX_COAST, range_meas_var=RANGE_MEAS_VAR,
    ))
    ekf_data = run(ekf_tracker, gen_ekf, initial_targets, range_per_bin,
                   radar_cfg, forced_coast_frames, n_frames)

    gen_imm = FMCWBeatSignalGenerator(radar_cfg)
    imm_cfg = IMMConfig(
        models=[
            MotionModelSpec(motion_model="cv"),
            MotionModelSpec(motion_model="drag",
                            motion_params={"drag_coefficient_k": k_bins}),
        ],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        initial_mode_probs=[0.5, 0.5],
        max_coast=MAX_COAST,
        range_meas_var=RANGE_MEAS_VAR,
    )
    imm_tracker = IMMTracker(imm_cfg)
    imm_data = run(imm_tracker, gen_imm, initial_targets, range_per_bin,
                   radar_cfg, forced_coast_frames, n_frames)

    truth = [
        (fi, range_at(initial_targets[0], fi, radar_cfg.frame_period_s)[0])
        for fi in range(n_frames)
    ]

    fig, (ax_pos, ax_mu) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_pos.plot([f for f, _ in truth], [r for _, r in truth],
                color="lightskyblue", linewidth=3, alpha=0.7, label="Truth")
    for data, color, name in [
        (kf_data, "tab:red", "KF (CV)"),
        (ekf_data, "tab:green", "EKF (drag)"),
        (imm_data, "tab:purple", "IMM (CV+drag)"),
    ]:
        frames = [d["frame"] for d in data]
        pos = [d["pos_m"] for d in data]
        ax_pos.plot(frames, pos, color=color, linewidth=1.5, alpha=0.85,
                    label=name)
        ax_pos.scatter(frames, pos, color=color, s=18)

    ax_pos.axvline(transition_frame, color="orange", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"regime boundary (frame {transition_frame})")
    ax_pos.set_ylabel("range [m]")
    ax_pos.set_title(
        "KF vs EKF vs IMM on switching scenario\n"
        f"CV phase: frames 0-{transition_frame} (v0={v0:g} m/s) · drag phase: "
        f"frames {transition_frame}-{n_frames} (k={drag_k:g})"
    )
    ax_pos.legend(loc="upper left")
    ax_pos.grid(True, alpha=0.3)

    imm_frames = [d["frame"] for d in imm_data if d["mode_probs"] is not None]
    cv_probs = [d["mode_probs"][0] for d in imm_data
                if d["mode_probs"] is not None]
    drag_probs = [d["mode_probs"][1] for d in imm_data
                  if d["mode_probs"] is not None]
    ax_mu.plot(imm_frames, cv_probs, color="tab:cyan", linewidth=1.8,
               label="IMM µ_CV")
    ax_mu.plot(imm_frames, drag_probs, color="tab:olive", linewidth=1.8,
               label="IMM µ_drag")
    ax_mu.axvline(transition_frame, color="orange", linestyle="--",
                  linewidth=1.2, alpha=0.7)
    ax_mu.set_xlabel("frame")
    ax_mu.set_ylabel("IMM mode probability")
    ax_mu.set_ylim(-0.05, 1.05)
    ax_mu.legend(loc="center right")
    ax_mu.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
