"""MIMO DoA + point-cloud demonstration figures (Part II / M2b).

Produces two static figures:
  report/mimo_processing_view.png  -- 3-panel: integrated RD (+CFAR) | DoA spectra
                                      (4 methods) | bird's-eye-view point cloud
  report/mimo_superresolution.png  -- two close coherent sources: Bartlett blurs,
                                      Capon/MUSIC/ESPRIT resolve.

Usage:  python3 -m evaluation.mimo_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.doa import estimate_doa, spectrum
from processing.mimo_frame import process_mimo_frame
from processing.point_cloud import tdm_doppler_compensation
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header

SPECTRAL = ["bartlett", "capon", "music"]


def _db(p):
    return 10.0 * np.log10(p / np.max(p) + 1e-12)


def processing_view(scenario: str, out: Path) -> None:
    cfg, targets = load_scenario(scenario)
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo(targets, frame_index=0)
    # Reuse the single-sourced per-frame core (RD -> integrate -> CFAR-NCI ->
    # min-range gate -> DoA cloud -> cluster); this demo only adds the per-method
    # DoA-spectra panel the core doesn't produce.
    header = build_header(frame=stack, frame_id=0, topic="demo",
                          metadata=gen.metadata(), ground_truth=[],
                          timestamp_unix_s=0.0)
    res = process_mimo_frame(stack, header, doa_method="music", n_sources=1,
                             cfar_method="os")
    rd, rd_db, mask = res["rd"], res["power_db"], res["mask"]
    cloud, objects = res["cloud"], res["objects"]
    pos, range_axis, vel_axis = res["pos"], res["range_axis"], res["vel_axis"]

    d, r = np.unravel_index(np.argmax(rd_db), rd_db.shape)
    snap = tdm_doppler_compensation(rd[:, d, r], d, cfg.n_chirps, cfg.n_tx, cfg.n_rx)
    gt = gen.ground_truth(targets, frame_index=0)
    gt_az = [g["azimuth_deg"] for g in gt]
    gt_xy = [(g["range_m"] * np.cos(np.deg2rad(g["azimuth_deg"])),
              g["range_m"] * np.sin(np.deg2rad(g["azimuth_deg"]))) for g in gt]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: integrated RD map + CFAR detections
    ax0.imshow(rd_db, aspect="auto", origin="lower", cmap="viridis",
               extent=[range_axis[0], range_axis[-1], vel_axis[0], vel_axis[-1]])
    dd, rr = np.where(mask)
    ax0.scatter(range_axis[rr], vel_axis[dd], s=12, facecolors="none",
                edgecolors="red", label="CFAR")
    ax0.set(xlabel="range [m]", ylabel="radial velocity [m/s]",
            title="integrated RD + CFAR")
    ax0.legend(loc="upper right", fontsize=8)

    # Panel 2: DoA spectra at the brightest cell
    for m in SPECTRAL:
        g, p = spectrum(snap, pos, m, n_sources=1)
        ax1.plot(g, _db(p), label=m)
    for az in estimate_doa(snap, pos, method="esprit", n_sources=1):
        ax1.axvline(az, color="purple", ls=":", label="esprit")
    for az in gt_az:
        ax1.axvline(az, color="k", ls="--", lw=1, label="GT")
    ax1.set(xlabel="azimuth [deg]", ylabel="norm. power [dB]",
            title="DoA spectra (brightest cell)", xlim=(-90, 90))
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)

    # Panel 3: BEV point cloud, clustered objects, GT
    if cloud.shape[0]:
        ax2.scatter(cloud[:, 0], cloud[:, 1], c="0.7", s=8, label="points")
    for obj in objects:
        ax2.scatter([obj.x], [obj.y], marker="o", facecolors="none",
                    edgecolors="C1", s=110, linewidths=2.0, label="object")
        ax2.add_patch(plt.Circle((obj.x, obj.y), max(obj.extent_m, 0.2),
                                 fill=False, color="C1", ls=":", lw=0.8))
    for (gx, gy) in gt_xy:
        ax2.scatter([gx], [gy], marker="x", c="k", s=80, label="GT")
    ax2.set(xlabel="x [m]", ylabel="y [m]",
            title=f"BEV: {len(objects)} clustered objects")
    ax2.axis("equal")
    h, l = ax2.get_legend_handles_labels()
    if l:
        ax2.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def superresolution(out: Path, az_true=(-3.0, 3.0)) -> None:
    pos = 0.5 * np.arange(12)
    az = np.deg2rad(np.array(az_true))
    A = np.exp(1j * 2 * np.pi * np.outer(pos, np.sin(az)))
    x = A @ np.ones(len(az_true))                      # two coherent sources
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in SPECTRAL:
        g, p = spectrum(x, pos, m, n_sources=2)
        ax.plot(g, _db(p), label=m)
    for a in estimate_doa(x, pos, method="esprit", n_sources=2):
        ax.axvline(a, color="purple", ls=":", label="esprit")
    for a in az_true:
        ax.axvline(a, color="k", ls="--", lw=1, label="GT")
    ax.set(xlabel="azimuth [deg]", ylabel="norm. power [dB]", xlim=(-30, 30),
           title=f"Two coherent sources {az_true[0]}/{az_true[1]} deg: "
                 f"Bartlett blurs, super-res resolves")
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    rep = ROOT / "report"
    # Two-target scene: shows the cloud collapsing to two separated objects.
    processing_view(str(ROOT / "scenarios/scenario_mimo_2targets.json"),
                    rep / "mimo_processing_view.png")
    superresolution(rep / "mimo_superresolution.png")


if __name__ == "__main__":
    main()
