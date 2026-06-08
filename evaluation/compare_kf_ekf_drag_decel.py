"""V2 of compare_kf_ekf_overlay.py with amplified KF-vs-EKF divergence.

Points at scenario_drag_decel.json (v0=12 m/s, k=0.5) and bumps the
tracker's range_meas_var to 5.0 (matching the IMM v2 script) so the Kalman
gain drops and the predict step weighs more — that is where KF and EKF
actually differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import TrackerConfig
from processing.range_doppler import range_doppler
from processing.tracking import Tracker, drag_k_metric_to_bins
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)
from simulator.motion_models import range_at


SCENARIO = "scenarios/scenario_drag_decel.json"
FRAMES = 30
MAX_COAST = 15
K_METRIC = 0.5  # matches the scenario's drag_coefficient_k
RANGE_MEAS_VAR = 5.0  # lowers Kalman gain → predict step weighs more
DOPPLER_AWARE_SPAWN = True
# Force coast frames: on these frames, discard detections and let the tracker
# predict-only. Decouples the KF-vs-EKF demonstration from CFAR/amplitude
# tuning — pick a window during the high-v early phase. Enabled via --forced_coast.
FORCED_COAST_WINDOW = range(5, 13)
OUT = Path("docs/tracker_stress_drag_decel.png")


def run(motion_model: str, forced_coast_frames):
    radar_cfg, initial_targets = load_scenario(SCENARIO)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    range_per_bin = float(gen.range_axis_m()[1] - gen.range_axis_m()[0])
    drag_k_bins = drag_k_metric_to_bins(K_METRIC, range_per_bin)
    if DOPPLER_AWARE_SPAWN:
        lam = 3e8 / radar_cfg.fc_hz
        v_per_dop_mps = lam / (2.0 * radar_cfg.n_chirps * radar_cfg.chirp_period_s)
        dop_scale = v_per_dop_mps * radar_cfg.frame_period_s / range_per_bin
        dop_center = radar_cfg.n_chirps / 2.0
    else:
        dop_scale = None
        dop_center = None
    cfg = TrackerConfig(method="gnn", motion_model=motion_model,
                        drag_coefficient_k=drag_k_bins,
                        max_coast=MAX_COAST,
                        range_meas_var=RANGE_MEAS_VAR,
                        doppler_to_vel_bins_scale=dop_scale,
                        doppler_bin_center=dop_center)
    tracker = Tracker(cfg)
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
        if frame_idx in forced_coast_frames:
            dets = []
        tracker.step(dets)
        if tracker.tracks:
            # Pick the single track to plot for this frame and record its position/status/mode-probabilities
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
            # If even the nearest track is more than 3 m off truth, treat it as "no real track this frame" and continue 
            if abs(best_pos_m - truth_r) > 3.0:
                continue
            frames_data.append({
                "frame": frame_idx,
                "pos_m": best_pos_m,
                "status": best.status,
            })
    return frames_data, range_per_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forced_coast", action="store_true", default=False,
                        help="force predict-only on frames range(5, 13)")
    args = parser.parse_args()

    if args.forced_coast:
        forced_coast_frames = set(FORCED_COAST_WINDOW)
        title_first_line = ("KF vs EKF on drag-decel motion"
                            "(forced coast: range(5, 13))\n")
        out = OUT.with_stem(OUT.stem + "_forced_coast")
    else:
        forced_coast_frames = set()
        title_first_line = "KF vs EKF on drag-decel motion\n"
        out = OUT

    kf_data, range_per_bin = run("cv", forced_coast_frames)
    ekf_data, _ = run("drag", forced_coast_frames)

    radar_cfg, initial_targets = load_scenario(SCENARIO)
    truth = []
    for fi in range(FRAMES):
        r, _ = range_at(initial_targets[0], fi, radar_cfg.frame_period_s)
        truth.append((fi, r))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot([f for f, _ in truth], [r for _, r in truth],
            color="lightskyblue", linewidth=3, alpha=0.7, label="Truth")

    for data, color, name in [(kf_data, "tab:red", "KF (CV)"),
                                (ekf_data, "tab:green", "EKF (drag)")]:
        frames = [d["frame"] for d in data]
        pos = [d["pos_m"] for d in data]
        statuses = [d["status"] for d in data]
        ax.plot(frames, pos, color=color, linewidth=1.5,
                alpha=0.8, label=name)
        for f, p, s in zip(frames, pos, statuses):
            if s == "coasting":
                ax.scatter([f], [p], color=color, s=40, marker="o",
                           facecolors="white", edgecolors=color, linewidth=1.5)
            else:
                ax.scatter([f], [p], color=color, s=40, marker="o")

    ax.set_xlabel("frame")
    ax.set_ylabel("range [m]")
    ax.set_title(
        title_first_line
        + f"v0=12 m/s · k={K_METRIC} · meas_var={RANGE_MEAS_VAR} · "
        "filled=confirmed, hollow=coasting"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
