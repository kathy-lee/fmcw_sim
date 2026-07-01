"""Dense range-azimuth-Doppler (RAD) tensor via the Bartlett beamformer.

Applies the conventional beamformer |a^H x|^2 densely over every range-Doppler
cell — the same operation the sparse DoA path (processing.doa.bartlett) runs at
detected cells only, reusing the same steering grid and TDM Doppler compensation.
"""
from __future__ import annotations

import numpy as np

from processing.doa import _GRID_DEG, steering_matrix
from processing.point_cloud import tdm_doppler_compensation


def rad_bartlett(rd_stack, pos_wl, n_tx, n_rx, grid_deg=None):
    """Dense Bartlett RAD tensor.

    rd_stack: (n_virtual, D, Rn) complex per-antenna range-Doppler stack.
    pos_wl:   virtual-element positions in wavelengths.
    Returns:  (n_azimuth, D, Rn) real power, azimuth over `grid_deg`
              (default: the DoA module grid, -90..90 deg, 1 deg, 181 bins).
    """
    grid_deg = _GRID_DEG if grid_deg is None else np.asarray(grid_deg, dtype=float)
    pos_wl = np.asarray(pos_wl, dtype=float).ravel()
    A = steering_matrix(pos_wl, np.deg2rad(grid_deg))     # (n_virtual, n_az)
    n_virtual, D, Rn = rd_stack.shape
    rad = np.empty((grid_deg.size, D, Rn), dtype=np.float64)
    for d in range(D):
        snap = tdm_doppler_compensation(rd_stack[:, d, :], d, D, n_tx, n_rx)
        rad[:, d, :] = np.abs(A.conj().T @ snap) ** 2
    return rad
