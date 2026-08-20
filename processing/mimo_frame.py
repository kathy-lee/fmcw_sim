"""Per-frame MIMO processing core (single-sourced).

The detect-then-angle chain for one MIMO frame: per-channel range-Doppler ->
non-coherent integration -> CFAR (NCI-recalibrated) -> min-range gate -> DoA
point cloud -> Cartesian clustering. Lives here (neutral) rather than in a
subscriber/eval module so all three consumers — the live subscriber
(streaming_subscriber_pointcloud), the offline runner (evaluation.mimo_pipeline_runner),
and the single-frame view (evaluation.mimo_single_frame_view) — share one
implementation.
"""
from __future__ import annotations

import time

import numpy as np

from processing.cfar import cfar_2d
from processing.clustering import cluster_point_cloud
from processing.fixed_point import quantize
from processing.point_cloud import integrate, mimo_point_cloud
from processing.range_doppler import range_doppler

C = 299_792_458.0
# CFAR window params (method chosen per call). Default method is "ca"
# (cell-averaging) — fast and vectorized
_CFAR_KW = dict(guard=(2, 4), training=(4, 8), pfa=1e-4)


def frame_params(header: dict) -> dict:
    """Derive the array/axis parameters needed by the pipeline from a header."""
    cfg = header["metadata"]["config"]
    der = header["metadata"]["derived"]
    n_tx, n_rx = int(cfg["n_tx"]), int(cfg["n_rx"])
    n_virtual = n_tx * n_rx
    n_range, n_chirps = int(cfg["n_range"]), int(cfg["n_chirps"])

    pos = float(cfg["rx_spacing_wl"]) * np.arange(n_virtual)
    fs, slope = float(der["fs_hz"]), float(der["slope_hz_per_s"])
    range_axis = (C * np.fft.fftfreq(n_range, d=1.0 / fs)
                  / (2.0 * slope))[: n_range // 2]
    lam = float(der["lambda_m"])
    dop = np.fft.fftshift(np.fft.fftfreq(n_chirps, d=float(cfg["chirp_period_s"])))
    vel_axis = (lam * dop / 2.0) / n_tx          # TDM velocity scaling
    return dict(n_tx=n_tx, n_rx=n_rx, pos=pos,
                range_axis=range_axis, vel_axis=vel_axis)


def process_mimo_frame(frame: np.ndarray, header: dict,
                       doa_method: str = "music", n_sources: int = 1,
                       cfar_method: str = "ca", eps_m: float = 0.8,
                       vel_scale: float = 1.0, min_samples: int = 2,
                       min_range_bins: int = 6, fp=None) -> dict:
    """Run the pipeline on one MIMO frame (testable core).

    Returns the stage outputs (incl. the raw `cloud` and clustered `objects`)
    plus a `timings` dict (seconds) for rd / integrate / cfar / doa_pc / cluster,
    and `total` (range-Doppler -> clustered objects).
    """
    p = frame_params(header)
    t0 = time.perf_counter()
    if fp is not None:                       # Q_adc (digitizer; always fixed scale)
        frame = quantize(frame, fp.adc_bits, fp.adc_full_scale, is_complex=True,
                         rounding=fp.rounding, overflow=fp.overflow)
    rd = np.stack([range_doppler(frame[v], fp=fp) for v in range(frame.shape[0])])
    t1 = time.perf_counter()
    power = integrate(rd, fp=fp)
    power_db = 10.0 * np.log10(power + 1e-12)
    t2 = time.perf_counter()
    # n_integrate = n_virtual: α is recalibrated for the non-coherently
    # integrated (Gamma-V) noise so the realized Pfa stays nominal.
    mask = cfar_2d(power_db, method=cfar_method, n_integrate=frame.shape[0],
                   **_CFAR_KW)
    # Min-range gate: drop the near-DC range bins where CFAR is unreliable (its
    # training window straddles the DC dead-zone -> deflated noise estimate ->
    # a near-DC false-alarm cluster, amplified by non-coherent integration).
    # MIMO-path only — the single-channel reports keep their bin-0 notch for
    # byte-for-byte.
    mask[:, :min_range_bins] = False
    t3 = time.perf_counter()
    cloud = mimo_point_cloud(
        rd, mask, p["pos"], float(p["range_axis"][1]), p["vel_axis"],
        p["n_tx"], p["n_rx"], doa_method=doa_method, n_sources=n_sources)
    t4 = time.perf_counter()
    objects = cluster_point_cloud(cloud, eps_m=eps_m, min_samples=min_samples,
                                  vel_scale=vel_scale)
    t5 = time.perf_counter()
    timings = {"rd": t1 - t0, "integrate": t2 - t1, "cfar": t3 - t2,
               "doa_pc": t4 - t3, "cluster": t5 - t4, "total": t5 - t0}
    return dict(rd=rd, power_db=power_db, mask=mask, cloud=cloud,
                objects=objects, timings=timings, **p)
