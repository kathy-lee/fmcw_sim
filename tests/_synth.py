"""Synthetic range-Doppler frame builders for unit tests.

Frames are returned in dB units, matching the output of `fmcw_rd_fft.py`.
A frame is shape (n_doppler, n_range), float32.
"""

from __future__ import annotations

import numpy as np


def make_noise_frame(
    n_doppler: int = 32,
    n_range: int = 640,
    noise_db: float = -60.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Frame with i.i.d. exponential noise (in linear power) at level `noise_db`.

    The frame is `10*log10(power)` where power ~ Exponential(mean=10**(noise_db/10)).
    This matches the |FFT|^2 distribution of complex Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    mean_lin = 10.0 ** (noise_db / 10.0)
    power = rng.exponential(scale=mean_lin, size=(n_doppler, n_range))
    return (10.0 * np.log10(power + 1e-12)).astype(np.float32)


def add_target(
    frame_db: np.ndarray,
    doppler_bin: int,
    range_bin: int,
    snr_db: float,
    noise_db: float = -60.0,
) -> np.ndarray:
    """Return a copy of `frame_db` with a single bright cell added.

    The cell's linear power is set to `noise_lin * 10**(snr_db/10)` (added to
    whatever was there). Useful for "single bright cell" CFAR tests.
    """
    out = frame_db.copy()
    target_lin = (10.0 ** (noise_db / 10.0)) * (10.0 ** (snr_db / 10.0))
    cur_lin = 10.0 ** (out[doppler_bin, range_bin] / 10.0)
    out[doppler_bin, range_bin] = 10.0 * np.log10(cur_lin + target_lin + 1e-12)
    return out


def add_target_block(
    frame_db: np.ndarray,
    doppler_bin: int,
    range_bin: int,
    snr_db: float,
    extent: tuple = (1, 1),  # (doppler_extent, range_extent), each is half-width
    noise_db: float = -60.0,
) -> np.ndarray:
    """Add a (2*ed+1) x (2*er+1) bright block centered at (doppler_bin, range_bin)."""
    out = frame_db.copy()
    ed, er = extent
    for dd in range(-ed, ed + 1):
        for dr in range(-er, er + 1):
            d = doppler_bin + dd
            r = range_bin + dr
            if 0 <= d < out.shape[0] and 0 <= r < out.shape[1]:
                out = add_target(out, d, r, snr_db, noise_db=noise_db)
    return out


def make_two_targets(
    n_doppler: int = 32,
    n_range: int = 640,
    targets: list = None,
    noise_db: float = -60.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Convenience: noise + a list of bright cells.

    `targets` is a list of (doppler_bin, range_bin, snr_db). Default = two well-separated.
    """
    if targets is None:
        targets = [(16, 100, 30.0), (12, 300, 30.0)]
    frame = make_noise_frame(n_doppler, n_range, noise_db=noise_db, seed=seed)
    for d, r, snr in targets:
        frame = add_target(frame, d, r, snr, noise_db=noise_db)
    return frame
