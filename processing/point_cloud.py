"""MIMO point-cloud stage.

Non-coherent integration of the per-channel range-Doppler stack (here), and
Cartesian conversion of detected (range, Doppler, azimuth) cells into a point
cloud. Pure NumPy; no generator dependency.
"""
from __future__ import annotations

import numpy as np

from processing.doa import estimate_doa
from processing.fixed_point import quantize_stage


def integrate(rd_stack: np.ndarray, fp=None) -> np.ndarray:
    """Non-coherent integration across virtual channels.

    rd_stack: (n_virtual, D, Rn) complex range-Doppler stack.
    Returns: (D, Rn) real power map = sum_v |rd_stack[v]|^2 — the detection map
    CFAR runs on (channel SNR boost, channel axis collapsed).

    With fp set, each |rd_v|^2 is quantized (Q_mag) and the running sum (Q_acc).
    """
    if fp is None:
        return np.sum(np.abs(rd_stack) ** 2, axis=0)
    acc = np.zeros(rd_stack.shape[1:], dtype=np.float64)
    for v in range(rd_stack.shape[0]):
        mag = np.abs(rd_stack[v]) ** 2
        mag = quantize_stage(mag, fp.mag_bits, fp, fp.mag_full_scale)
        acc = quantize_stage(acc + mag, fp.acc_bits, fp, fp.acc_full_scale)
    return acc


def tdm_doppler_compensation(snapshot: np.ndarray, doppler_bin: float,
                             n_chirps: int, n_tx: int, n_rx: int) -> np.ndarray:
    """Remove the TDM inter-TX Doppler phase from a virtual-array snapshot.

    In TDM, virtual element e = (tx i, rx j) is sampled `i` chirp-periods later
    than tx 0, so at (fftshifted) Doppler bin `doppler_bin` it carries an extra
    phase exp(j 2pi (d - n_chirps/2) i / (n_chirps n_tx)). Removing it restores
    the clean spatial steering vector before DoA. Identity when n_tx == 1.
    """
    i_tx = np.arange(snapshot.shape[0]) // n_rx
    phase = (-2.0 * np.pi * (doppler_bin - n_chirps / 2.0) * i_tx
             / (n_chirps * n_tx))
    return snapshot * np.exp(1j * phase)


def to_point_cloud(detections, range_per_bin: float,
                   velocity_axis: np.ndarray) -> np.ndarray:
    """Convert (range_bin, doppler_bin, azimuth_deg, snr) detections to Cartesian.

    Returns (M, 4) array of [x, y, v_radial, snr]: x = range·cos(az),
    y = range·sin(az) (boresight = +x), v_radial from the Doppler axis.
    """
    pts = []
    for range_bin, doppler_bin, azimuth_deg, snr in detections:
        rng = range_bin * range_per_bin
        az = np.deg2rad(azimuth_deg)
        vr = velocity_axis[int(round(doppler_bin))]
        pts.append((rng * np.cos(az), rng * np.sin(az), vr, snr))
    return np.array(pts, dtype=float).reshape(-1, 4)


def mimo_point_cloud(rd_stack: np.ndarray, mask: np.ndarray, pos_wl: np.ndarray,
                     range_per_bin: float, velocity_axis: np.ndarray,
                     n_tx: int, n_rx: int, doa_method: str = "bartlett",
                     n_sources: int = 1) -> np.ndarray:
    """Detect-then-angle point cloud: for each True cell in `mask`, TDM-compensate
    the per-channel snapshot, estimate azimuth(s), and convert to Cartesian.

    No clustering — one point per (cell, azimuth peak).
    Returns (M, 4) [x, y, v_radial, snr_db].
    """
    n_chirps = rd_stack.shape[1]
    detections = []
    for d, r in np.argwhere(mask):
        snap = tdm_doppler_compensation(rd_stack[:, d, r], d, n_chirps, n_tx, n_rx)
        snr_db = 10.0 * np.log10(np.sum(np.abs(snap) ** 2) + 1e-12)
        for az in estimate_doa(snap, pos_wl, method=doa_method,
                               n_sources=n_sources):
            detections.append((r, d, az, snr_db))
    return to_point_cloud(detections, range_per_bin, velocity_axis)
