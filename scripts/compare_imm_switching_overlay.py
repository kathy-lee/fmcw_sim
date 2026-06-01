"""Overlay KF, EKF, and IMM tracks on scenario_imm_switching for sub-project A.

The switching scenario has a CV phase (frames 0-40) and a drag phase
(frames 40-80). Expected behavior:
  - KF: tracks CV phase well; drifts during drag phase
  - EKF (drag): tracks drag phase well; drifts during CV phase
  - IMM: tracks both phases via mode probability switching

This overlay plot puts all three tracker outputs on the same axes for
direct visual comparison, and marks the regime boundary at frame 40.
"""

from __future__ import annotations

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


SCENARIO = "scenarios/scenario_imm_switching_demo.json"
FRAMES = 80
K_METRIC = 0.5
MAX_COAST = 15
TRANSITION_FRAME = 40
# Experiment: bump assumed measurement variance to lower Kalman gain and
# give the motion model more weight in the post-update state. Default is
# 0.5 (bins²). At 5.0, K shifts from ~0.67 → ~0.17.
RANGE_MEAS_VAR = 5.0
OUT = Path("docs/tracker_stress_imm_switching_overlay_highvar.png")


def run(tracker, gen, initial_targets, range_per_bin, radar_cfg):
    """Run the pipeline and, at each frame, return the track whose predicted
    position is closest to GT (so we follow the actual target track, not a
    false track at DC range). Skips frames where no track is near GT."""
    frames_data = []
    for frame_idx in range(FRAMES):
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", pfa=1e-4, window="hann",
                       dc_removal=True, clutter_removal=False)
        mask[:, 0] = False
        dets = cluster(mask, rd_db, method="peak", min_cells=2,
                       peak_neighborhood=3)
        tracker.step(dets)
        if not tracker.tracks:
            continue
        # Pick the confirmed track whose state[0] is closest to GT range.
        truth_r, _ = range_at(initial_targets[0], frame_idx,
                              radar_cfg.frame_period_s)
        candidates = [t for t in tracker.tracks
                      if t.status in ("confirmed", "coasting")]
        if not candidates:
            continue
        best = min(candidates,
                   key=lambda t: abs(float(t.state[0]) * range_per_bin
                                       - truth_r))
        # Reject if not actually close (likely all false tracks)
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
    radar_cfg, initial_targets = load_scenario(SCENARIO)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    range_per_bin = float(gen.range_axis_m()[1] - gen.range_axis_m()[0])
    k_bins = drag_k_metric_to_bins(K_METRIC, range_per_bin)

    # 1. KF
    gen_kf = FMCWBeatSignalGenerator(radar_cfg)
    kf_tracker = Tracker(TrackerConfig(method="gnn", motion_model="cv",
                                        max_coast=MAX_COAST,
                                        range_meas_var=RANGE_MEAS_VAR))
    kf_data = run(kf_tracker, gen_kf, initial_targets, range_per_bin, radar_cfg)

    # 2. EKF (drag-only)
    gen_ekf = FMCWBeatSignalGenerator(radar_cfg)
    ekf_tracker = Tracker(TrackerConfig(
        method="gnn", motion_model="drag", drag_coefficient_k=k_bins,
        max_coast=MAX_COAST, range_meas_var=RANGE_MEAS_VAR,
    ))
    ekf_data = run(ekf_tracker, gen_ekf, initial_targets, range_per_bin, radar_cfg)

    # 3. IMM (CV + drag-EKF)
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
    imm_data = run(imm_tracker, gen_imm, initial_targets, range_per_bin, radar_cfg)

    # Truth
    truth = [
        (fi, range_at(initial_targets[0], fi, radar_cfg.frame_period_s)[0])
        for fi in range(FRAMES)
    ]

    fig, (ax_pos, ax_mu) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Position panel
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

    # Regime boundary
    ax_pos.axvline(TRANSITION_FRAME, color="orange", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"regime boundary (frame {TRANSITION_FRAME})")
    ax_pos.set_ylabel("range [m]")
    ax_pos.set_title(
        "KF vs EKF vs IMM on switching scenario\n"
        f"CV phase: frames 0-{TRANSITION_FRAME} · drag phase: "
        f"frames {TRANSITION_FRAME}-{FRAMES}"
    )
    ax_pos.legend(loc="upper left")
    ax_pos.grid(True, alpha=0.3)

    # IMM mode-probability panel (only IMM has mode_probs)
    imm_frames = [d["frame"] for d in imm_data if d["mode_probs"] is not None]
    cv_probs = [d["mode_probs"][0] for d in imm_data
                if d["mode_probs"] is not None]
    drag_probs = [d["mode_probs"][1] for d in imm_data
                  if d["mode_probs"] is not None]
    ax_mu.plot(imm_frames, cv_probs, color="tab:cyan", linewidth=1.8,
               label="IMM µ_CV")
    ax_mu.plot(imm_frames, drag_probs, color="tab:olive", linewidth=1.8,
               label="IMM µ_drag")
    ax_mu.axvline(TRANSITION_FRAME, color="orange", linestyle="--",
                  linewidth=1.2, alpha=0.7)
    ax_mu.set_xlabel("frame")
    ax_mu.set_ylabel("IMM mode probability")
    ax_mu.set_ylim(-0.05, 1.05)
    ax_mu.legend(loc="center right")
    ax_mu.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
