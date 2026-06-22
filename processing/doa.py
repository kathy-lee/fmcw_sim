"""Direction-of-arrival (DoA) estimators for a ULA (Part II / M2).

Pure NumPy; no generator dependency — callers pass the virtual-element positions
in wavelengths (e.g. FMCWBeatSignalGenerator.virtual_element_positions()). One
entry point, `estimate_doa`, dispatches on `method`:

  - "bartlett" : conventional beamformer |a^H x|^2 (single snapshot, full aperture)
  - "capon"    : MVDR 1/(a^H R^-1 a)            (covariance via FB smoothing)
  - "music"    : noise-subspace pseudo-spectrum (covariance via FB smoothing)
  - "esprit"   : search-free, rotational invariance (covariance via FB smoothing)

The covariance methods build R from a single snapshot by forward-backward spatial
smoothing (a detected cell yields one snapshot -> rank-1 R otherwise).
"""
from __future__ import annotations

import numpy as np

_GRID_DEG = np.arange(-90.0, 90.0 + 1e-9, 1.0)
_GRID_RAD = np.deg2rad(_GRID_DEG)


def steering_matrix(pos_wl: np.ndarray, grid_rad: np.ndarray) -> np.ndarray:
    """Steering vectors for each angle: (len(pos), len(grid)).

    Column k is a(theta_k) = exp(j 2pi pos sin(theta_k)); pos in wavelengths.
    """
    return np.exp(1j * 2.0 * np.pi * np.outer(pos_wl, np.sin(grid_rad)))


def _peaks(spectrum: np.ndarray, grid_deg: np.ndarray, n: int) -> np.ndarray:
    """Top-`n` local-maximum azimuths (deg), sorted ascending."""
    s = spectrum
    is_peak = (np.r_[True, s[1:] > s[:-1]] & np.r_[s[:-1] > s[1:], True])
    idx = np.where(is_peak)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(s))])
    idx = idx[np.argsort(s[idx])[::-1][:n]]
    return np.sort(grid_deg[idx])


def spectrum(x: np.ndarray, pos_wl: np.ndarray, method: str,
             n_sources: int = 1, subarray_len: int = 8,
             loading: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Spatial spectrum P(theta) over the azimuth grid for a spectral method
    ('bartlett' | 'capon' | 'music'). Returns (grid_deg, P). ESPRIT is gridless
    and has no spectrum. Used for plotting and by the peak-pick wrappers."""
    x = np.asarray(x).ravel()
    pos_wl = np.asarray(pos_wl, dtype=float).ravel()
    if method == "bartlett":
        A = steering_matrix(pos_wl, _GRID_RAD)
        return _GRID_DEG, np.abs(A.conj().T @ x) ** 2
    R = fb_smooth(x, subarray_len)
    L = R.shape[0]
    A = steering_matrix(pos_wl[:L], _GRID_RAD)
    if method == "capon":
        R = R + loading * (np.trace(R).real / L) * np.eye(L)
        denom = np.real(np.sum(A.conj() * (np.linalg.inv(R) @ A), axis=0))
        return _GRID_DEG, 1.0 / denom
    if method == "music":
        _, vecs = np.linalg.eigh(R)
        En = vecs[:, :L - n_sources]
        proj = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
        return _GRID_DEG, 1.0 / proj
    raise ValueError(f"no spectrum for method {method!r}")


def bartlett(x: np.ndarray, pos_wl: np.ndarray, n_sources: int) -> np.ndarray:
    """Conventional beamformer on the full snapshot. P(theta) = |a^H x|^2."""
    g, p = spectrum(x, pos_wl, "bartlett")
    return _peaks(p, g, n_sources)


def fb_smooth(x: np.ndarray, subarray_len: int) -> np.ndarray:
    """Forward-backward spatially-smoothed covariance from one snapshot.

    Averages the J = M - L + 1 length-L forward subarrays, then adds the
    backward (exchange-conjugated) version: R_fb = (R_f + Pi R_f^* Pi)/2. This
    restores rank so Capon/MUSIC/ESPRIT work on a single snapshot.
    """
    x = x.ravel()
    M, L = x.size, subarray_len
    J = M - L + 1
    R = np.zeros((L, L), dtype=complex)
    for j in range(J):
        y = x[j:j + L]
        R += np.outer(y, y.conj())
    R /= J
    Pi = np.fliplr(np.eye(L))
    return (R + Pi @ R.conj() @ Pi) / 2.0


def capon(x: np.ndarray, pos_wl: np.ndarray, n_sources: int,
          subarray_len: int, loading: float) -> np.ndarray:
    """MVDR beamformer: P(theta) = 1 / (a^H R^-1 a), R via FB smoothing."""
    g, p = spectrum(x, pos_wl, "capon", n_sources, subarray_len, loading)
    return _peaks(p, g, n_sources)


def music(x: np.ndarray, pos_wl: np.ndarray, n_sources: int,
          subarray_len: int) -> np.ndarray:
    """MUSIC pseudo-spectrum: P(theta) = 1 / |En^H a|^2, En the noise subspace
    (the L - D smallest eigenvectors of the FB-smoothed R)."""
    g, p = spectrum(x, pos_wl, "music", n_sources, subarray_len)
    return _peaks(p, g, n_sources)


def esprit(x: np.ndarray, pos_wl: np.ndarray, n_sources: int,
           subarray_len: int) -> np.ndarray:
    """TLS-ESPRIT: search-free DoA from the ULA shift invariance.

    Signal subspace E_s (top-D eigenvectors of the FB-smoothed R) obeys
    E_s2 ~ E_s1 Psi, where rows shifted by one element introduce the phase
    exp(j 2pi d sin theta). TLS-solve Psi; its eigenvalue angles give the angles.
    """
    R = fb_smooth(x, subarray_len)
    L = R.shape[0]
    _, vecs = np.linalg.eigh(R)
    Es = vecs[:, L - n_sources:]                        # signal subspace (L, D)
    Es1, Es2 = Es[:-1, :], Es[1:, :]                    # one-element shift
    D = n_sources
    C = np.hstack([Es1, Es2])                           # (L-1, 2D)
    _, _, Vh = np.linalg.svd(C)
    V = Vh.conj().T                                     # (2D, 2D)
    Psi = -V[:D, D:] @ np.linalg.inv(V[D:, D:])         # TLS solution
    phi = np.angle(np.linalg.eigvals(Psi))             # = 2pi d sin theta
    d = pos_wl[1] - pos_wl[0]                           # element spacing (wl)
    sin_theta = np.clip(phi / (2.0 * np.pi * d), -1.0, 1.0)
    return np.sort(np.degrees(np.arcsin(sin_theta)))


def estimate_doa(x: np.ndarray, pos_wl: np.ndarray, method: str = "bartlett",
                 n_sources: int = 1, subarray_len: int = 8,
                 loading: float = 1e-3) -> np.ndarray:
    """Estimate azimuth(s) in degrees from snapshot `x`. See module docstring."""
    x = np.asarray(x).ravel()
    pos_wl = np.asarray(pos_wl, dtype=float).ravel()
    if method == "bartlett":
        return bartlett(x, pos_wl, n_sources)
    if method == "capon":
        return capon(x, pos_wl, n_sources, subarray_len, loading)
    if method == "music":
        return music(x, pos_wl, n_sources, subarray_len)
    if method == "esprit":
        return esprit(x, pos_wl, n_sources, subarray_len)
    raise ValueError(f"unknown doa method: {method!r}")
