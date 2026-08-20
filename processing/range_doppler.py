"""Range-Doppler 2D FFT with optional DC removal and slow-time mean subtraction."""

from __future__ import annotations

import numpy as np

from processing.fixed_point import quantize_stage


def range_doppler(
    frame: np.ndarray,
    window: str = "hann",
    dc_removal: bool = True,
    clutter_removal: bool = True,
    fp=None,
    return_range_cube: bool = False,
) -> np.ndarray:
    """Compute the 2D range-Doppler map from a complex beat-signal frame.

    Pipeline:
        1. Per-chirp DC removal (optional, default on).
        2. Range windowing (hann or rect).
        3. Range FFT, keep positive beat frequencies.
        4. Slow-time mean subtraction (optional, default on) —
           suppresses zero-Doppler / stationary clutter.
        5. Doppler windowing.
        6. Doppler FFT, fftshift to centre zero-Doppler.

    Args:
        frame: complex (n_chirps, n_range) input.
        window: "hann" or "rect".
        dc_removal: subtract per-chirp mean before range FFT.
        clutter_removal: subtract slow-time mean before Doppler FFT.

    Returns:
        Complex RD map of shape (n_chirps, n_range // 2).
        If return_range_cube, returns (out, rfft) where rfft is the post-Q_cube
        range-FFT tensor of the same shape.
    """
    n_chirps, n_range = frame.shape
    x = frame.astype(np.complex64, copy=True)

    if dc_removal:
        x = x - x.mean(axis=1, keepdims=True)

    if window == "hann":
        x = x * np.hanning(n_range)[None, :]

    rfft = np.fft.fft(x, axis=1)[:, : n_range // 2]
    if fp is not None:                                    # Q_cube (radar-cube word)
        rfft = quantize_stage(rfft, fp.cube_bits, fp, fp.cube_full_scale)

    if clutter_removal:
        rfft = rfft - rfft.mean(axis=0, keepdims=True)

    if window == "hann":
        rfft = rfft * np.hanning(n_chirps)[:, None]

    out = np.fft.fftshift(np.fft.fft(rfft, axis=0), axes=0)
    if fp is not None:                                    # Q_rd
        out = quantize_stage(out, fp.rd_bits, fp, fp.rd_full_scale)
    if return_range_cube:
        return out, rfft
    return out
