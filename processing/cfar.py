"""2D CFAR detection on range-Doppler maps.

Public entry point: `cfar_2d(rd_db, method=...)`. Returns a boolean mask the
same shape as the input. All algorithms operate on linear power.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.ndimage
from scipy.optimize import brentq
from scipy.special import gammaln

_VALID_METHODS = {"ca", "os", "go", "so"}


def cfar_2d(
    rd_db: np.ndarray,
    method: str = "ca",
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
    rank_frac: float = 0.75,
) -> np.ndarray:
    """Dispatch to the configured CFAR algorithm. See module docstring."""
    if method not in _VALID_METHODS:
        raise ValueError(
            f"cfar_2d.method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )
    rd_db = np.nan_to_num(rd_db, nan=-200.0, posinf=20.0, neginf=-200.0)
    if method == "ca":
        return cfar_2d_ca(rd_db, guard=guard, training=training, pfa=pfa)
    if method == "os":
        return cfar_2d_os(rd_db, guard=guard, training=training, pfa=pfa,
                          rank_frac=rank_frac)
    if method in ("go", "so"):
        return cfar_2d_goso(rd_db, mode=method, guard=guard,
                            training=training, pfa=pfa)
    raise NotImplementedError(f"cfar method {method!r} not yet implemented")


def cfar_2d_ca(
    rd_db: np.ndarray,
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
) -> np.ndarray:
    """2D Cell-Averaging CFAR. Returns a boolean detection mask.

    Boundary behavior: uses scipy `mode="nearest"` (edge replication), which
    inflates the apparent noise estimate and suppresses false alarms in the
    outermost ~T_d rows / T_r columns. Tests trim the edge before measuring
    empirical FA rate.
    """
    G_d, G_r = guard
    T_d, T_r = training
    rd_lin = np.power(10.0, rd_db / 10.0).astype(np.float64)

    full_d = 2 * (G_d + T_d) + 1
    full_r = 2 * (G_r + T_r) + 1
    inner_d = 2 * G_d + 1
    inner_r = 2 * G_r + 1

    n_full = full_d * full_r
    n_inner = inner_d * inner_r
    n_train = n_full - n_inner
    if n_train <= 0:
        raise ValueError(
            f"training region is empty for guard={guard}, training={training}"
        )

    # Sum over full window and over inner (guard) block via uniform_filter * size.
    sum_full = scipy.ndimage.uniform_filter(rd_lin, size=(full_d, full_r),
                                            mode="nearest") * n_full
    sum_inner = scipy.ndimage.uniform_filter(rd_lin, size=(inner_d, inner_r),
                                             mode="nearest") * n_inner

    noise = (sum_full - sum_inner) / n_train

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    threshold = alpha * noise

    return rd_lin > threshold


@functools.lru_cache(maxsize=128)
def _os_alpha(N: int, k: int, pfa: float) -> float:
    """Solve Pfa(α) = pfa for OS-CFAR.

    Under i.i.d. exponential noise, the OS-CFAR Pfa as a function of the
    threshold scale α (real-valued, > 0) is

        Pfa(α) = exp[ gammaln(N+1) - gammaln(N-k+1)
                      + gammaln(α + N - k + 1) - gammaln(α + N + 1) ].

    Pfa is strictly decreasing in α, so we invert with brentq.
    """
    if not (1 <= k <= N):
        raise ValueError(f"need 1 <= k <= N, got N={N}, k={k}")

    def log_pfa(alpha: float) -> float:
        return (
            gammaln(N + 1) - gammaln(N - k + 1)
            + gammaln(alpha + N - k + 1) - gammaln(alpha + N + 1)
        )

    target = np.log(pfa)
    # log_pfa(0) > target (Pfa=1), log_pfa(1e7) << target. brentq is safe.
    return float(brentq(lambda a: log_pfa(a) - target, 1e-3, 1e7, xtol=1e-4))


def cfar_2d_os(
    rd_db: np.ndarray,
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
    rank_frac: float = 0.75,
) -> np.ndarray:
    """2D Ordered-Statistics CFAR."""
    G_d, G_r = guard
    T_d, T_r = training
    rd_lin = np.power(10.0, rd_db / 10.0).astype(np.float64)

    full_d = 2 * (G_d + T_d) + 1
    full_r = 2 * (G_r + T_r) + 1
    inner_d = 2 * G_d + 1
    inner_r = 2 * G_r + 1
    pad_d = G_d + T_d
    pad_r = G_r + T_r

    padded = np.pad(rd_lin, ((pad_d, pad_d), (pad_r, pad_r)), mode="edge")
    # shape (n_d, n_r, full_d, full_r)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (full_d, full_r))

    # Mask out the guard block (center inner_d x inner_r).
    keep = np.ones((full_d, full_r), dtype=bool)
    cd0 = (full_d - inner_d) // 2
    cr0 = (full_r - inner_r) // 2
    keep[cd0:cd0 + inner_d, cr0:cr0 + inner_r] = False

    # Extract training cells: (n_d, n_r, N).
    training_cells = windows[..., keep]
    N = training_cells.shape[-1]
    # k is capped at N-1, not N: the maximum order statistic is a poor noise
    # estimator (picks up any nearby strong target), so we exclude it.
    k = max(1, min(N - 1, int(round(rank_frac * N))))

    # k-th smallest training cell per CUT (1-indexed → np.partition uses 0-indexed).
    noise = np.partition(training_cells, k - 1, axis=-1)[..., k - 1]

    alpha = _os_alpha(N, k, pfa)
    threshold = alpha * noise

    return rd_lin > threshold


def _quadrant_kernels(G_d: int, G_r: int, T_d: int, T_r: int):
    """Return (kernels, sizes) for the 4 training-ring quadrants.

    Quadrants are non-overlapping rectangular strips around the guard block.
    Using NumPy index convention (index 0 = first row, increasing downward):
      lead_d:  rows BELOW the guard (higher row indices), full window width
      lag_d:   rows ABOVE the guard (lower row indices), full window width
      lead_r:  cols to the RIGHT of the guard, only within the guard's d-band
      lag_r:   cols to the LEFT of the guard, only within the guard's d-band

    The lead/lag labels are arbitrary — GO/SO take max/min across all four,
    so symmetry makes the assignment cosmetic.
    """
    full_d = 2 * (G_d + T_d) + 1
    full_r = 2 * (G_r + T_r) + 1
    cd = G_d + T_d  # kernel center row
    cr = G_r + T_r  # kernel center col

    def kern(d_slice, r_slice):
        k = np.zeros((full_d, full_r), dtype=np.float64)
        k[d_slice, r_slice] = 1.0
        return k

    k_lead_d = kern(slice(cd + G_d + 1, cd + G_d + 1 + T_d), slice(0, full_r))
    k_lag_d  = kern(slice(cd - G_d - T_d, cd - G_d), slice(0, full_r))
    k_lead_r = kern(slice(cd - G_d, cd + G_d + 1),
                    slice(cr + G_r + 1, cr + G_r + 1 + T_r))
    k_lag_r  = kern(slice(cd - G_d, cd + G_d + 1),
                    slice(cr - G_r - T_r, cr - G_r))

    n_lead_d = T_d * full_r
    n_lag_d = T_d * full_r
    n_lead_r = (2 * G_d + 1) * T_r
    n_lag_r = (2 * G_d + 1) * T_r

    return (k_lead_d, k_lag_d, k_lead_r, k_lag_r), (n_lead_d, n_lag_d, n_lead_r, n_lag_r)


@functools.lru_cache(maxsize=64)
def _goso_alpha(quadrant_sizes: tuple, pfa: float, mode: str,
                n_trials: int = 30000, seed: int = 0) -> float:
    """Calibrate α via Monte Carlo for GO/SO with given quadrant sizes.

    Each quadrant mean of N_q i.i.d. exp(1) cells is Gamma(N_q, 1/N_q).
    Cell-under-test under H0 is exp(1). We pick the empirical (1-pfa)-quantile
    of CUT / max(or min) of quadrant means.

    To keep tail-quantile variance bounded the effective trial count scales
    with 1/pfa: at pfa=1e-4, ~3 events at 30k trials is too few — we bump to
    ~3M trials, taking on the order of 100ms on first call (cached afterwards).
    """
    if mode not in ("go", "so"):
        raise ValueError(f"mode must be 'go' or 'so', got {mode!r}")
    effective_trials = max(n_trials, int(300 / pfa))
    rng = np.random.default_rng(seed)
    means = np.column_stack([
        rng.gamma(shape=N_q, scale=1.0, size=effective_trials) / N_q
        for N_q in quadrant_sizes
    ])
    cut = rng.exponential(scale=1.0, size=effective_trials)
    noise = means.max(axis=1) if mode == "go" else means.min(axis=1)
    ratios = cut / noise
    return float(np.quantile(ratios, 1.0 - pfa))


def cfar_2d_goso(
    rd_db: np.ndarray,
    mode: str,                          # "go" or "so"
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
) -> np.ndarray:
    """2D Greatest-Of / Smallest-Of CFAR with 4 training-ring quadrants."""
    G_d, G_r = guard
    T_d, T_r = training
    rd_lin = np.power(10.0, rd_db / 10.0).astype(np.float64)

    kernels, sizes = _quadrant_kernels(G_d, G_r, T_d, T_r)
    means = []
    for k, n_q in zip(kernels, sizes):
        sums = scipy.ndimage.convolve(rd_lin, k, mode="nearest")
        means.append(sums / n_q)
    stack = np.stack(means, axis=0)             # (4, n_d, n_r)
    noise = stack.max(axis=0) if mode == "go" else stack.min(axis=0)

    alpha = _goso_alpha(tuple(sizes), pfa, mode)
    threshold = alpha * noise
    return rd_lin > threshold
