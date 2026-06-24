"""Clustering facet (M5): DBSCAN eps / vel_scale object-count study."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.clustering import cluster_point_cloud
from processing.mimo_frame import process_mimo_frame
from simulator.beat_signal_generator import FMCWBeatSignalGenerator
from simulator.wire import build_header


def _cloud_at(cfg, targets, frame_index):
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame_mimo(targets, frame_index=frame_index)
    header = build_header(frame=frame, frame_id=frame_index, topic="eval",
                          metadata=gen.metadata(), ground_truth=[],
                          timestamp_unix_s=0.0)
    return process_mimo_frame(frame, header)["cloud"]


def object_count_grid(cfg, targets, frame_index, eps_list, vel_list):
    """Object count for each (eps_m, vel_scale) on a fixed processed cloud.
    Returns (eps_list, vel_list, counts[i][j]) with i over eps, j over vel."""
    cloud = _cloud_at(cfg, targets, frame_index)
    counts = [[len(cluster_point_cloud(cloud, eps_m=e, vel_scale=v, min_samples=2))
               for v in vel_list] for e in eps_list]
    return eps_list, vel_list, counts


def main():
    from simulator.beat_signal_generator import load_scenario
    cfg, targets = load_scenario(str(ROOT / "scenarios/scenario_mimo_crossing.json"))
    eps_list = [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]
    vel_list = [0.0, 0.5, 1.0, 2.0, 4.0]
    _, _, counts = object_count_grid(cfg, targets, 40, eps_list, vel_list)
    counts = np.array(counts)
    # Categorical heatmap: the (eps, vel_scale) grid is non-uniform, so imshow+extent
    # would misplace the ticks — instead plot on integer indices, label the ticks with
    # the actual values, and annotate each cell with its object count.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(counts, origin="lower", aspect="auto", cmap="viridis",
                   vmin=1, vmax=6)
    ax.set_xticks(range(len(vel_list))); ax.set_xticklabels(vel_list)
    ax.set_yticks(range(len(eps_list))); ax.set_yticklabels(eps_list)
    ax.set(xlabel="vel_scale", ylabel="eps [m]",
           title="Clustering: object count at crossing (GT = 2)")
    for i in range(len(eps_list)):
        for j in range(len(vel_list)):
            c = int(counts[i, j])
            ax.text(j, i, str(c), ha="center", va="center",
                    color="white" if c < 3 else "black", fontsize=11,
                    fontweight="bold")
    fig.colorbar(im, ax=ax, label="object count")
    out = ROOT / "report/mimo_evaluation/mimo_clustering.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
