"""Offline end-to-end MIMO pipeline runner.

run_pipeline_mimo reuses the single-sourced per-frame core process_mimo_frame
(simulator.streaming_subscriber_pointcloud) and adds the CartesianTracker. This
is the offline (no-ZMQ) analogue of tracker_stress.run_pipeline. The Cartesian
scorers that consume its history live in evaluation.mimo_tracking_eval.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from processing.mimo_frame import process_mimo_frame
from processing.tracking_cartesian import CartesianTracker, CartesianTrackerConfig
from simulator.wire import build_header

# Immutable per-frame track snapshot (CartesianTracker mutates tracks in place).
CartesianTrackSnapshot = namedtuple(
    "CartesianTrackSnapshot", ["id", "state", "status"])


def _track_snapshot(t):
    return CartesianTrackSnapshot(id=t.id, state=np.array(t.state, copy=True),
                                  status=str(t.status))


def run_pipeline_mimo(gen, targets, n_frames, *, doa_method="music",
                      cfar_method="ca", eps_m=0.8, vel_scale=1.0, min_samples=2,
                      tracker_config=None, fp=None):
    """generate_frame_mimo -> process_mimo_frame -> CartesianTracker.step per
    frame, on a configured generator. Returns list[(frame_idx,
    [CartesianTrackSnapshot, ...])].

    For a Monte-Carlo trial, pass a re-seeded generator:
    FMCWBeatSignalGenerator(dataclasses.replace(cfg, element_noise_seed=seed))."""
    metadata = gen.metadata()
    # Predict timestep = frame period, so the tracker's state velocity is m/s,
    # consistent with the m/s Doppler measurement.
    tracker = CartesianTracker(
        tracker_config or CartesianTrackerConfig(dt=gen.cfg.frame_period_s))
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
        history.append((fi, [_track_snapshot(t) for t in tracks]))
    return history
