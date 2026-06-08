"""Diagnose whether KF and EKF actually differ when invoked through the
sub-project A harness on scenario_drag_decel.json.

Hypothesis to test: both filters' post-update positions are nearly identical
because the Kalman gain snaps to measurements, but their velocity state
estimates and predict-only states should differ if EKF wiring is correct.
"""

from __future__ import annotations

import numpy as np

from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import TrackerConfig
from processing.range_doppler import range_doppler
from processing.tracking import Tracker
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


def run_one(motion_model: str, drag_k_bins: float = 0.0333):
    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_drag_decel.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)
    cfg = TrackerConfig(method="gnn", motion_model=motion_model,
                        drag_coefficient_k=drag_k_bins)
    tracker = Tracker(cfg)
    history = []
    for frame_idx in range(80):
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", pfa=1e-4, window="hann",
                       dc_removal=True, clutter_removal=False)
        mask[:, 0] = False
        dets = cluster(mask, rd_db, method="peak", min_cells=2,
                       peak_neighborhood=3)
        tracks = tracker.step(dets)
        if tracks:
            t = tracks[0]
            history.append({
                "frame": frame_idx,
                "id": t.id,
                "pos": float(t.state[0]),
                "vel": float(t.state[1]),
                "status": t.status,
                "n_dets": len(dets),
            })
        else:
            history.append({
                "frame": frame_idx, "id": None, "pos": None, "vel": None,
                "status": None, "n_dets": len(dets),
            })
    return history


kf = run_one("cv")
ekf = run_one("drag", drag_k_bins=0.0333)

print(f"{'frame':>5s} {'kf_pos':>8s} {'ekf_pos':>8s} {'Δpos':>7s} "
      f"{'kf_vel':>8s} {'ekf_vel':>8s} {'Δvel':>7s} "
      f"{'kf_status':>10s} {'ekf_status':>10s}")
for k, e in zip(kf, ekf):
    if k["pos"] is None or e["pos"] is None:
        print(f"{k['frame']:5d}  (no track yet)")
        continue
    dpos = e["pos"] - k["pos"]
    dvel = e["vel"] - k["vel"]
    print(f"{k['frame']:5d} {k['pos']:8.3f} {e['pos']:8.3f} {dpos:+7.4f} "
          f"{k['vel']:8.4f} {e['vel']:8.4f} {dvel:+7.4f} "
          f"{k['status']:>10s} {e['status']:>10s}")
