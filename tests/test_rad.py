import numpy as np

from processing.doa import _GRID_DEG, estimate_doa, steering_matrix
from processing.point_cloud import tdm_doppler_compensation
from processing.rad import rad_bartlett


def _single_source_stack(true_az_deg=20.0, n_rx=8, D=4, Rn=3):
    """n_tx=1 ULA snapshot for one azimuth, constant across Doppler/range."""
    pos = 0.5 * np.arange(n_rx)                       # half-wavelength ULA
    a = np.exp(1j * 2.0 * np.pi * pos * np.sin(np.deg2rad(true_az_deg)))
    rd = np.empty((n_rx, D, Rn), dtype=np.complex64)
    rd[...] = a[:, None, None]
    return rd, pos


def test_rad_shape():
    rd, pos = _single_source_stack()
    rad = rad_bartlett(rd, pos, n_tx=1, n_rx=8)
    assert rad.shape == (_GRID_DEG.size, 4, 3)
    assert rad.dtype == np.float64


def test_rad_peak_at_true_azimuth():
    rd, pos = _single_source_stack(true_az_deg=20.0)
    rad = rad_bartlett(rd, pos, n_tx=1, n_rx=8)
    peak_deg = _GRID_DEG[np.argmax(rad[:, 0, 0])]
    assert abs(peak_deg - 20.0) <= 1.0


def test_rad_matches_sparse_bartlett():
    """Dense RAD at a cell equals the sparse estimate_doa Bartlett peak."""
    rd, pos = _single_source_stack(true_az_deg=-12.0)
    rad = rad_bartlett(rd, pos, n_tx=1, n_rx=8)
    d, r = 1, 2
    snap = tdm_doppler_compensation(rd[:, d, r], d, rd.shape[1], 1, 8)
    sparse_peak = estimate_doa(snap, pos, method="bartlett", n_sources=1)[0]
    dense_peak = _GRID_DEG[np.argmax(rad[:, d, r])]
    assert dense_peak == sparse_peak


def test_tdm_compensation_2d_matches_columnwise():
    """n-D safety: compensating a (n_virtual, Rn) block equals per-column."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4)))
    block = tdm_doppler_compensation(x, doppler_bin=2, n_chirps=8, n_tx=3, n_rx=2)
    for r in range(x.shape[1]):
        col = tdm_doppler_compensation(x[:, r], 2, 8, 3, 2)
        np.testing.assert_allclose(block[:, r], col)
