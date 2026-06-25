"""2D CFAR detection on range-Doppler maps.

Public entry point: `cfar_2d(rd_db, method=...)`. Returns a boolean mask the
same shape as the input. All algorithms operate on linear power.

Threshold calibration:
    The threshold for every method is α × noise_estimate, where α is calibrated
    to hit the design Pfa. Calibration is empirical: we draw pure-noise RD power
    maps from the same windowed-FFT chain the pipeline uses (so adjacent cells
    have the correct windowing-induced correlation), run the method-specific
    noise-estimate operator on them, and take the (1−Pfa) quantile of the
    cut/noise ratios as α.

    The earlier analytical / i.i.d.-Monte-Carlo calibrations were biased
    because the real RD map has correlated cells (Hann-window power
    autocorrelation extends a few bins), violating the i.i.d. exponential
    assumption those formulas rely on.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.ndimage
from scipy.optimize import brentq
from scipy.special import gammaln

_VALID_METHODS = {"ca", "os", "go", "so"}
_VALID_WINDOWS = {"hann", "rect"}

# Frames used for empirical α calibration. 300 frames × 64×128 cells ≈ 2.5M
# samples per noise pool. Enough for stable (1-Pfa)=1−1e-4 quantile estimates.
_CALIBRATION_FRAMES = 300


def cfar_2d(
    rd_db: np.ndarray,
    method: str = "ca",
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
    rank_frac: float = 0.75,
    window: str = "hann",
    dc_removal: bool = True,
    clutter_removal: bool = False,
    n_integrate: int = 1,
) -> np.ndarray:
    """Dispatch to the configured CFAR algorithm. See module docstring.

    `window`, `dc_removal`, `clutter_removal` must match the values passed to
    `range_doppler`. They govern only the α calibration (which simulates the
    same processing chain to generate noise samples); the input `rd_db` itself
    is not modified.

    `n_integrate` is the number of channels non-coherently integrated into
    `rd_db` (MIMO `Σ_v|RD_v|²`). It re-calibrates α for the post-integration
    Gamma(n_integrate) noise so the realized Pfa stays nominal; default 1
    (single channel).
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"cfar_2d.method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )
    if window not in _VALID_WINDOWS:
        raise ValueError(
            f"cfar_2d.window must be one of {sorted(_VALID_WINDOWS)}, got {window!r}"
        )
    rd_db = np.nan_to_num(rd_db, nan=-200.0, posinf=20.0, neginf=-200.0)
    kw = dict(guard=guard, training=training, pfa=pfa, window=window,
              dc_removal=dc_removal, clutter_removal=clutter_removal,
              n_integrate=n_integrate)
    if method == "ca":
        return cfar_2d_ca(rd_db, **kw)
    if method == "os":
        return cfar_2d_os(rd_db, rank_frac=rank_frac, **kw)
    if method in ("go", "so"):
        return cfar_2d_goso(rd_db, mode=method, **kw)
    raise NotImplementedError(f"cfar method {method!r} not yet implemented")


# --------------------------------------------------------------------------- #
# Empirical α calibration                                                      #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=16)
def _windowed_noise_rd_maps(
    rd_shape: tuple,
    window: str,
    dc_removal: bool,
    clutter_removal: bool,
    n_frames: int,
    seed: int,
    n_integrate: int = 1,
) -> np.ndarray:
    """Pure-noise RD power maps with the pipeline's processing chain.

    Mirrors `range_doppler` exactly: complex-Gaussian noise → optional
    per-chirp mean removal → range window → range FFT → optional slow-time
    mean removal → Doppler window → Doppler FFT → fftshift → magnitude squared.

    Each pipeline flag affects which cells become "dead" (range bin 0 for
    dc_removal, doppler bin n_d/2 for clutter_removal). The α calibration
    must see the same dead cells the pipeline produces, otherwise SO (and to
    a lesser extent CA/OS) miscalibrate at the edges of the dead regions.

    `n_integrate` > 1 models MIMO non-coherent integration: each returned map is
    the sum of `n_integrate` INDEPENDENT single-channel power maps (per-cell
    noise becomes Gamma(n_integrate) instead of exponential), so α is calibrated
    against the post-integration statistics. n_integrate=1 is the single-channel
    case (byte-for-byte unchanged).

    Shape returned: (n_frames, n_doppler, n_range_out), linear power.
    """
    if window not in _VALID_WINDOWS:
        raise ValueError(
            f"window must be one of {sorted(_VALID_WINDOWS)}, got {window!r}"
        )
    n_d, n_r_out = rd_shape
    n_r_in = 2 * n_r_out   # range FFT keeps positive half

    if window == "hann":
        wr = np.hanning(n_r_in)
        wd = np.hanning(n_d)
    else:  # "rect"
        wr = np.ones(n_r_in)
        wd = np.ones(n_d)

    rng = np.random.default_rng(seed)
    total = n_frames * n_integrate
    samples = np.empty((total, n_d, n_r_out), dtype=np.float64)
    for i in range(total):
        x = (rng.standard_normal((n_d, n_r_in)) +
             1j * rng.standard_normal((n_d, n_r_in))) / np.sqrt(2.0)
        if dc_removal:
            x = x - x.mean(axis=1, keepdims=True)
        x = x * wr[None, :]
        rfft = np.fft.fft(x, axis=1)[:, :n_r_out]
        if clutter_removal:
            rfft = rfft - rfft.mean(axis=0, keepdims=True)
        rfft = rfft * wd[:, None]
        out = np.fft.fftshift(np.fft.fft(rfft, axis=0), axes=0)
        samples[i] = np.abs(out) ** 2
    if n_integrate > 1:
        # Non-coherent integration: sum each group of n_integrate maps.
        samples = samples.reshape(n_frames, n_integrate, n_d, n_r_out).sum(axis=1)
    return samples


def _ratios_ca(samples: np.ndarray, guard: tuple, training: tuple) -> np.ndarray:
    G_d, G_r = guard
    T_d, T_r = training
    full_d = 2 * (G_d + T_d) + 1
    full_r = 2 * (G_r + T_r) + 1
    inner_d = 2 * G_d + 1
    inner_r = 2 * G_r + 1
    n_full = full_d * full_r
    n_inner = inner_d * inner_r
    n_train = n_full - n_inner

    out = []
    for rd_lin in samples:
        sum_full = scipy.ndimage.uniform_filter(
            rd_lin, size=(full_d, full_r), mode="nearest") * n_full
        sum_inner = scipy.ndimage.uniform_filter(
            rd_lin, size=(inner_d, inner_r), mode="nearest") * n_inner
        noise = (sum_full - sum_inner) / n_train
        out.append((rd_lin / noise).ravel())
    return np.concatenate(out)


def _ratios_os(
    samples: np.ndarray, guard: tuple, training: tuple, rank_frac: float
) -> np.ndarray:
    G_d, G_r = guard
    T_d, T_r = training
    full_d = 2 * (G_d + T_d) + 1
    full_r = 2 * (G_r + T_r) + 1
    inner_d = 2 * G_d + 1
    inner_r = 2 * G_r + 1
    pad_d = G_d + T_d
    pad_r = G_r + T_r

    keep = np.ones((full_d, full_r), dtype=bool)
    cd0 = (full_d - inner_d) // 2
    cr0 = (full_r - inner_r) // 2
    keep[cd0:cd0 + inner_d, cr0:cr0 + inner_r] = False

    out = []
    for rd_lin in samples:
        padded = np.pad(rd_lin, ((pad_d, pad_d), (pad_r, pad_r)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (full_d, full_r))
        training_cells = windows[..., keep]
        N = training_cells.shape[-1]
        k = max(1, min(N - 1, int(round(rank_frac * N))))
        noise = np.partition(training_cells, k - 1, axis=-1)[..., k - 1]
        out.append((rd_lin / noise).ravel())
    return np.concatenate(out)


def _ratios_goso(
    samples: np.ndarray, guard: tuple, training: tuple, mode: str
) -> np.ndarray:
    G_d, G_r = guard
    T_d, T_r = training
    kernels, sizes = _quadrant_kernels(G_d, G_r, T_d, T_r)

    out = []
    for rd_lin in samples:
        means = []
        for k, n_q in zip(kernels, sizes):
            sums = scipy.ndimage.convolve(rd_lin, k, mode="nearest")
            means.append(sums / n_q)
        stack = np.stack(means, axis=0)
        noise = stack.max(axis=0) if mode == "go" else stack.min(axis=0)
        out.append((rd_lin / noise).ravel())
    return np.concatenate(out)


@functools.lru_cache(maxsize=64)
def _alpha_empirical(
    method: str,
    rd_shape: tuple,
    guard: tuple,
    training: tuple,
    pfa: float,
    window: str,
    dc_removal: bool,
    clutter_removal: bool,
    rank_frac: float = 0.75,
    n_frames: int = _CALIBRATION_FRAMES,
    seed: int = 0,
    n_integrate: int = 1,
) -> float:
    """Calibrate α empirically against pure-noise samples generated through the
    same processing chain (window / dc_removal / clutter_removal) the pipeline
    uses, so that "dead" cells produced by the chain are present in the
    calibration data the same way they are in the live RD map.

    Convention: the default `seed=0` is intentional and reproducible. To
    keep calibration noise decoupled from pipeline noise, `RadarConfig.seed`
    defaults to a non-zero value (99999), so the FMCWBeatSignalGenerator's
    RNG stream never accidentally collides with this calibration RNG stream.
    Anyone wiring up a new noise path that interacts with cfar should
    similarly pick a seed != 0; otherwise the two RNG streams are identical
    and any "empirical Pfa" measured downstream becomes self-consistent with
    the calibration data rather than a true out-of-sample measurement. Unit
    tests use `seed=12345` to make this explicit.
    """
    samples = _windowed_noise_rd_maps(
        rd_shape, window, dc_removal, clutter_removal, n_frames, seed,
        n_integrate,
    )
    if method == "ca":
        ratios = _ratios_ca(samples, guard, training)
    elif method == "os":
        ratios = _ratios_os(samples, guard, training, rank_frac)
    elif method in ("go", "so"):
        ratios = _ratios_goso(samples, guard, training, method)
    else:
        raise ValueError(f"unknown method {method!r}")
    return float(np.quantile(ratios, 1.0 - pfa))


# --------------------------------------------------------------------------- #
# Public CFAR variants                                                         #
# --------------------------------------------------------------------------- #

def cfar_2d_ca(
    rd_db: np.ndarray,
    guard: tuple = (2, 4),
    training: tuple = (4, 8),
    pfa: float = 1e-4,
    window: str = "hann",
    dc_removal: bool = True,
    clutter_removal: bool = False,
    n_integrate: int = 1,
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

    sum_full = scipy.ndimage.uniform_filter(rd_lin, size=(full_d, full_r),
                                            mode="nearest") * n_full
    sum_inner = scipy.ndimage.uniform_filter(rd_lin, size=(inner_d, inner_r),
                                             mode="nearest") * n_inner
    noise = (sum_full - sum_inner) / n_train

    # # Previous version under i.i.d. exponential assumption
    # alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    # threshold = alpha * noise
    # return rd_lin > threshold

    alpha = _alpha_empirical("ca", tuple(rd_db.shape), tuple(guard),
                             tuple(training), pfa, window,
                             dc_removal, clutter_removal,
                             n_integrate=n_integrate)
    return rd_lin > alpha * noise


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
    window: str = "hann",
    dc_removal: bool = True,
    clutter_removal: bool = False,
    n_integrate: int = 1,
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

    # # Previous version under i.i.d. exponential assumption
    # alpha = _os_alpha(N, k, pfa)
    # threshold = alpha * noise
    # return rd_lin > threshold

    alpha = _alpha_empirical("os", tuple(rd_db.shape), tuple(guard),
                             tuple(training), pfa, window,
                             dc_removal, clutter_removal, rank_frac=rank_frac,
                             n_integrate=n_integrate)
    return rd_lin > alpha * noise


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
    window: str = "hann",
    dc_removal: bool = True,
    clutter_removal: bool = False,
    n_integrate: int = 1,
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
    stack = np.stack(means, axis=0)
    noise = stack.max(axis=0) if mode == "go" else stack.min(axis=0)

    # # Previous version under i.i.d. exponential assumption
    # alpha = _goso_alpha(tuple(sizes), pfa, mode)
    # threshold = alpha * noise
    # return rd_lin > threshold

    alpha = _alpha_empirical(mode, tuple(rd_db.shape), tuple(guard),
                             tuple(training), pfa, window,
                             dc_removal, clutter_removal,
                             n_integrate=n_integrate)
    return rd_lin > alpha * noise
