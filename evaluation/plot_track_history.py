"""Plot tracker history from a tracks.csv produced by the streaming pipeline.

Range vs frame_id, one colored line per track_id. Tentative samples are drawn
as 'x', confirmed as 'o'.

Usage:
    python -m evaluation.plot_track_history --tracks PATH [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Return {track_id: {frame_id, range_m, status}} arrays sorted by frame."""
    rows: dict[int, list[tuple]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[int(r["track_id"])].append((
                int(r["frame_id"]),
                float(r["range_m"]),
                r["status"],
            ))
    tracks = {}
    for tid, entries in rows.items():
        entries.sort(key=lambda e: e[0])
        frame_id = np.array([e[0] for e in entries], dtype=int)
        range_m = np.array([e[1] for e in entries], dtype=float)
        status = np.array([e[2] for e in entries], dtype=str)
        tracks[tid] = {"frame_id": frame_id, "range_m": range_m, "status": status}
    return tracks


def plot(tracks: dict[int, dict[str, np.ndarray]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    sorted_ids = sorted(tracks.keys())
    for i, tid in enumerate(sorted_ids):
        t = tracks[tid]
        color = cmap(i % 10)
        confirmed = t["status"] == "confirmed"
        tentative = ~confirmed

        ax.plot(t["frame_id"], t["range_m"], color=color, linewidth=1.0, alpha=0.7)
        if tentative.any():
            ax.scatter(t["frame_id"][tentative], t["range_m"][tentative],
                       color=color, marker="x", s=35)
        if confirmed.any():
            ax.scatter(t["frame_id"][confirmed], t["range_m"][confirmed],
                       marker="o", s=30, facecolors="none", edgecolors=color)

        ax.annotate(f"id={tid}", (t["frame_id"][0], t["range_m"][0]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color=color)

    ax.set_xlabel("frame_id")
    ax.set_ylabel("range [m]")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], marker="x", color="gray", linestyle="",
                   label="tentative"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="",
                   markerfacecolor="none", label="confirmed"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)

    ax.set_title(f"Track history ({len(sorted_ids)} tracks)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True,
                   help="path to tracks.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="output image path (default: alongside tracks.csv as .png)")
    args = p.parse_args()

    out = args.out or args.tracks.with_suffix(".png")
    tracks = load_tracks(args.tracks)
    if not tracks:
        raise SystemExit(f"no tracks found in {args.tracks}")
    plot(tracks, out)


if __name__ == "__main__":
    main()
