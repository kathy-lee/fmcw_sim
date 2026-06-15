"""Extended-target detection sweep (target realism report, Part II).

Companion to sweep_amplitude_fluctuation.py (Part I) / report 1 §3: SAME
OS-CFAR lens (via evaluation.sweep_amplitude.detect_frame), SAME SNR axis,
frame-advancing loop. Adds an EXTENT axis and three object-level metrics
(design §5):
  - object-level Pd: a frame is a hit if any detection centroid falls within
    the object's range span [centre +/- w] (w from ground_truth);
  - per-cell detection fraction: of the CFAR mask cells inside the span, the
    fraction that crossed threshold (spreading-loss / self-masking diagnostic);
  - cluster count per object (fragmentation bridge to the tracking study).

extent=None reproduces the point-target curve -- the report-1 / Part I anchor.

Usage:
    python -m evaluation.sweep_extent --frames 300
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.sweep_amplitude import detect_frame
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, load_scenario,
)

CFAR_METHOD = "os"
CLUSTER_METHOD = "dbscan"
DBSCAN_EPS = 2.0          # bridges sub-threshold gaps within an extended blob
DBSCAN_MIN_SAMPLES = 1
DOPPLER_TOL_BINS = 1.0    # span is range-only; Doppler stays tight


def object_metrics(gen, target, target_index, frame_index, cfar_method,
                   cluster_method):
    """Per-frame object-level metrics for a single extended/point target."""
    # min_cells=2 matches report 1 / Part I / the baseline detect_frame default
    # (a lone CFAR cell is noise, not a detection), so the point anchor and
    # fragmentation count are measured the project's way.
    centroids, mask, _ = detect_frame(
        gen, [target], frame_index, cfar_method,
        cluster_method=cluster_method, min_cells=2,
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    g = gen.ground_truth([target], frame_index=frame_index)[0]
    r_c = g["range_bin"]
    d_c = g["doppler_bin_shifted"]
    w = g["range_half_width_bins"]

    def in_span(rb, db):
        return (abs(rb - r_c) <= w + 1.0
                and abs(db - d_c) <= DOPPLER_TOL_BINS)

    in_span_clusters = [c for c in centroids if in_span(c.range_bin,
                                                        c.doppler_bin)]
    detected = len(in_span_clusters) >= 1
    cluster_count = len(in_span_clusters)

    # Per-cell fraction: of the range bins in [r_c-w, r_c+w] at the target's
    # Doppler row, the fraction flagged by CFAR.
    n_chirps, n_range = mask.shape
    d_idx = int(round(d_c))
    lo = max(0, int(np.floor(r_c - w)))
    hi = min(n_range - 1, int(np.ceil(r_c + w)))
    if 0 <= d_idx < n_chirps and hi >= lo:
        span_cells = mask[d_idx, lo:hi + 1]
        cell_fraction = float(span_cells.mean()) if span_cells.size else 0.0
    else:
        cell_fraction = 0.0

    return {"detected": detected, "cell_fraction": cell_fraction,
            "cluster_count": cluster_count}


def _case_key(extent):
    if extent is None:
        return "point"
    return f"ext_{extent['range_extent_m']:g}"


def run_sweep(scenario, amplitudes, extents, n_frames):
    radar_cfg, base_targets = load_scenario(str(scenario))
    base = base_targets[0]
    results = {}
    for extent in extents:
        key = _case_key(extent)
        pd_list, frac_list, frag_list = [], [], []
        for scale in amplitudes:
            cfg = radar_cfg
            t = dataclasses.replace(base, amplitude=base.amplitude * scale,
                                    extent=extent)
            gen = FMCWBeatSignalGenerator(cfg)
            hits = 0
            frac_sum = 0.0
            frag_sum = 0
            for i in range(n_frames):
                m = object_metrics(gen, t, 0, i, CFAR_METHOD, CLUSTER_METHOD)
                hits += int(m["detected"])
                frac_sum += m["cell_fraction"]
                frag_sum += m["cluster_count"]
            pd_list.append(hits / n_frames)
            frac_list.append(frac_sum / n_frames)
            frag_list.append(frag_sum / n_frames)
            snr_db = (20.0 * np.log10(t.amplitude)
                      - 10.0 * np.log10(cfg.noise_power))
            print(f"  {key:10s} scale={scale:8.2e} SNR={snr_db:6.1f} "
                  f"Pd={hits / n_frames:.2f} cellfrac={frac_sum / n_frames:.2f} "
                  f"clusters={frag_sum / n_frames:.2f}")
        results[key] = {"pd": pd_list, "cell_fraction": frac_list,
                        "cluster_count": frag_list,
                        "snr_db": [20.0 * np.log10(base.amplitude * s)
                                   - 10.0 * np.log10(radar_cfg.noise_power)
                                   for s in amplitudes]}
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=Path,
                   default=Path("scenarios/scenario_extended_detection.json"))
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--out", type=Path,
                   default=Path("report/sweep_extent.png"))
    p.add_argument("--csv", type=Path,
                   default=Path("report/sweep_extent.csv"))
    args = p.parse_args()

    amplitudes = np.logspace(np.log10(3.5e-5), np.log10(3.5e-3), 10)
    extents = [
        None,
        {"n_scatterers": 10, "range_extent_m": 0.3},
        {"n_scatterers": 10, "range_extent_m": 0.6},
        {"n_scatterers": 10, "range_extent_m": 1.2},
    ]
    results = run_sweep(args.scenario, amplitudes, extents, args.frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_pd, ax_frac, ax_frag) = plt.subplots(3, 1, figsize=(9, 9),
                                                  sharex=True)
    for key, r in results.items():
        ax_pd.plot(r["snr_db"], r["pd"], marker="o", label=key)
        ax_frac.plot(r["snr_db"], r["cell_fraction"], marker="o", label=key)
        ax_frag.plot(r["snr_db"], r["cluster_count"], marker="o", label=key)
    ax_pd.set_ylabel("object-level Pd"); ax_pd.grid(alpha=0.3); ax_pd.legend(fontsize=8)
    ax_pd.set_title("Extended-target detection vs point (equal total power)")
    ax_frac.set_ylabel("per-cell detection fraction"); ax_frac.grid(alpha=0.3)
    ax_frag.set_ylabel("clusters / object"); ax_frag.grid(alpha=0.3)
    ax_frag.set_xlabel("mean input SNR per sample [dB]")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "snr_db", "pd", "cell_fraction", "cluster_count"])
        for key, r in results.items():
            for i in range(len(r["pd"])):
                w.writerow([key, f"{r['snr_db'][i]:.2f}", f"{r['pd'][i]:.4f}",
                            f"{r['cell_fraction'][i]:.4f}",
                            f"{r['cluster_count'][i]:.4f}"])
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
