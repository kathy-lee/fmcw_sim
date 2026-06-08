"""Temporary diagnostic for the tracker stress test track-break bug.

Prints per-frame: GT positions, CFAR/clustering detections, and tracker
state (id, state[0]=range_bin, state[1]=range_rate, status).
"""
import numpy as np
from processing.cfar import cfar_2d
from processing.clustering import cluster
from processing.config import TrackerConfig
from processing.range_doppler import range_doppler
from processing.tracking import Tracker
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)


def main():
    radar_cfg, initial_targets = load_scenario(
        "scenarios/scenario_crossing_targets.json"
    )
    gen = FMCWBeatSignalGenerator(radar_cfg)
    tracker = Tracker(TrackerConfig(method="gnn"))

    for frame_idx in range(8):
        # Ground truth bin positions for this frame.
        gts = gen.ground_truth(initial_targets, frame_index=frame_idx)
        gt_str = ", ".join(
            f"GT-{i}(r_bin={g['range_bin']:.2f}, d_bin={g['doppler_bin_shifted']:.2f})"
            for i, g in enumerate(gts)
        )

        # Pipeline.
        frame = gen.generate_frame(initial_targets, frame_index=frame_idx)
        rd = range_doppler(frame, window="hann", dc_removal=True,
                           clutter_removal=False)
        rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
        mask = cfar_2d(rd_db, method="os", pfa=1e-4, window="hann",
                       dc_removal=True, clutter_removal=False)
        mask[:, 0] = False
        detections = cluster(mask, rd_db, method="peak", min_cells=2,
                             peak_neighborhood=3)
        det_str = ", ".join(
            f"D{i}(r_bin={d.range_bin:.2f}, d_bin={d.doppler_bin:.2f}, "
            f"peak={d.peak_db:.1f}dB)"
            for i, d in enumerate(detections)
        )

        tracks_before = [
            f"T{t.id}(r={t.state[0]:.2f}, rr={t.state[1]:.3f}, "
            f"d={t.last_doppler_bin:.2f}, {t.status})"
            for t in tracker.tracks
        ]

        tracks = tracker.step(detections)
        tracks_after = [
            f"T{t.id}(r={t.state[0]:.2f}, rr={t.state[1]:.3f}, "
            f"d={t.last_doppler_bin:.2f}, {t.status})"
            for t in tracks
        ]

        print(f"=== frame {frame_idx} ===")
        print(f"  GT:      {gt_str}")
        print(f"  Dets:    {det_str if det_str else '(none)'}")
        print(f"  Before:  {', '.join(tracks_before) if tracks_before else '(none)'}")
        print(f"  After:   {', '.join(tracks_after) if tracks_after else '(none)'}")
        print()


if __name__ == "__main__":
    main()
