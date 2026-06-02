"""Overlay KF and EKF tracks on the same plot to make the divergence visible.

Generates a single figure showing:
  - Truth trajectory (light blue line)
  - KF track positions (red markers and line)
  - EKF track positions (green markers and line)

Coast frames are drawn with smaller markers so the prediction-only segments
are visually distinct from the measurement-update frames.
"""

from __future__ import annotations

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


SCENARIO = "scenarios/scenario_drag_demo.json"
FRAMES = 30
MAX_COAST = 15
K_METRIC = 0.5  # matches the simulator's drag_coefficient_k in scenario JSON
DOPPLER_AWARE_SPAWN = False
OUT = Path("docs/tracker_stress_drag_demo_overlay.png")


def run(motion_model: str):
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
                        drag_coefficient_k=drag_k_bins, max_coast=MAX_COAST,
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
        tracker.step(dets)
        if tracker.tracks:
            t = tracker.tracks[0]
            frames_data.append({
                "frame": frame_idx,
                "pos_m": float(t.state[0]) * range_per_bin,
                "status": t.status,
            })
    return frames_data, range_per_bin


def main():
    kf_data, range_per_bin = run("cv")
    ekf_data, _ = run("drag")

    # Truth trajectory
    radar_cfg, initial_targets = load_scenario(SCENARIO)
    truth = []
    for fi in range(FRAMES):
        r, _ = range_at(initial_targets[0], fi, radar_cfg.frame_period_s)
        truth.append((fi, r))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Truth as thick light-blue line
    ax.plot([f for f, _ in truth], [r for _, r in truth],
            color="lightskyblue", linewidth=3, alpha=0.7, label="Truth")

    # KF and EKF tracks. Confirmed frames as filled markers, coasting as
    # hollow (so coast streaks are visually distinct from association frames).
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
        f"KF vs EKF on drag-decel scenario\n"
        f"Filled = confirmed; hollow = coasting. EKF (green) should stay "
        f"closer to truth during coast streaks."
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
