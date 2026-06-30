"""Offline end-to-end MIMO pipeline + Cartesian scorers.

run_pipeline_mimo reuses the single-sourced per-frame core process_mimo_frame
(simulator.streaming_subscriber_pointcloud) and adds the CartesianTracker. This
is the offline (no-ZMQ) analogue of tracker_stress.run_pipeline.
"""
from __future__ import annotations

import dataclasses
import math
from collections import namedtuple

import numpy as np
import scipy.optimize

from processing.mimo_frame import process_mimo_frame
from processing.tracking_cartesian import CartesianTracker, CartesianTrackerConfig
from simulator.beat_signal_generator import FMCWBeatSignalGenerator
from simulator.wire import build_header

# Immutable per-frame track snapshot (CartesianTracker mutates tracks in place).
CartesianTrackSnapshot = namedtuple(
    "CartesianTrackSnapshot", ["id", "state", "status"])


def _snapshot(t):
    return CartesianTrackSnapshot(id=t.id, state=np.array(t.state, copy=True),
                                  status=str(t.status))


def run_pipeline_mimo(radar_cfg, targets, n_frames, *, doa_method="music",
                      cfar_method="ca", eps_m=0.8, vel_scale=1.0, min_samples=2,
                      tracker_config=None, seed=None, fp=None):
    """generate_frame_mimo -> process_mimo_frame -> CartesianTracker.step per
    frame. Returns list[(frame_idx, [CartesianTrackSnapshot, ...])].

    seed (when not None) re-seeds the per-element noise via dataclasses.replace,
    leaving the scenario untouched (the Monte-Carlo hook)."""
    if seed is not None:
        radar_cfg = dataclasses.replace(radar_cfg, element_noise_seed=seed)
    gen = FMCWBeatSignalGenerator(radar_cfg)
    metadata = gen.metadata()
    # Predict timestep = frame period, so the tracker's state velocity is m/s,
    # consistent with the m/s Doppler measurement.
    tracker = CartesianTracker(
        tracker_config or CartesianTrackerConfig(dt=radar_cfg.frame_period_s))
    history = []
    for fi in range(n_frames):
        frame = gen.generate_frame_mimo(targets, frame_index=fi)
        header = build_header(frame=frame, frame_id=fi, topic="eval",
                              metadata=metadata, ground_truth=[],
                              timestamp_unix_s=0.0)
        out = process_mimo_frame(frame, header, doa_method=doa_method,
                                 cfar_method=cfar_method, eps_m=eps_m,
                                 vel_scale=vel_scale, min_samples=min_samples,
                                 fp=fp)
        tracks = tracker.step(out["objects"])
        history.append((fi, [_snapshot(t) for t in tracks]))
    return history


def build_gt_xy(targets, gen, n_frames):
    """Per-frame GT as (frame_idx, [(gt_id, x, y, vx, vy, az_deg), ...]).
    Cartesian from ground_truth()'s range_m + azimuth_deg; vx/vy are the
    radial-frame components (v·cos az, v·sin az) — adequate for position
    scoring (velocity RMSE uses radial speed only)."""
    out = []
    for fi in range(n_frames):
        rows = []
        for gid, g in enumerate(gen.ground_truth(targets, frame_index=fi)):
            az = math.radians(g["azimuth_deg"])
            r, v = g["range_m"], g["velocity_mps"]
            rows.append((gid, r * math.cos(az), r * math.sin(az),
                         v * math.cos(az), v * math.sin(az), g["azimuth_deg"]))
        out.append((fi, rows))
    return out


def associate_xy(gt_rows, snapshots, max_dist_m=2.0):
    """Hungarian match GT (x,y) to confirmed track (x,y). {gt_id: track_id|None}."""
    confirmed = [s for s in snapshots if s.status == "confirmed"]
    out = {row[0]: None for row in gt_rows}
    if not confirmed:
        return out
    cost = np.full((len(gt_rows), len(confirmed)), np.inf)
    for i, row in enumerate(gt_rows):
        for j, s in enumerate(confirmed):
            cost[i, j] = np.hypot(s.state[0] - row[1], s.state[1] - row[2])
    ri, ci = scipy.optimize.linear_sum_assignment(
        np.where(np.isfinite(cost), cost, 1e9))
    for r, c in zip(ri, ci):
        if cost[r, c] <= max_dist_m:
            out[gt_rows[r][0]] = confirmed[c].id
    return out


def track_rmse_xy(gt_traj, tracks_history, assoc_history, cutoff_m=None):
    """Position RMSE (m) over associated GT-frames (GOSPA cutoff optional)."""
    sq = []
    for (_, rows), (_, snaps), (_, mapping) in zip(gt_traj, tracks_history,
                                                   assoc_history):
        by_id = {s.id: s for s in snaps}
        for row in rows:
            tid = mapping.get(row[0])
            s = by_id.get(tid) if tid is not None else None
            if s is None:
                if cutoff_m is not None:
                    sq.append(cutoff_m ** 2)
                continue
            e2 = (s.state[0] - row[1]) ** 2 + (s.state[1] - row[2]) ** 2
            sq.append(min(e2, cutoff_m ** 2) if cutoff_m is not None else e2)
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")
