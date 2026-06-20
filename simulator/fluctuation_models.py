"""Per-target RCS fluctuation models (Swerling cases).

A fluctuating target's per-frame amplitude is `target.amplitude` multiplied
by a unit-mean-power factor sequence drawn from a dedicated RNG stream
(`RadarConfig.fluctuation_seed`), independent of the thermal-noise stream
(`RadarConfig.seed`) and the CFAR calibration stream (seed 0).

Physical construction: each Swerling case is realised as the RMS magnitude
of k complex-Gaussian scatterer voltages evolving as a first-order AR
process across frames:

    s_f = rho * s_{f-1} + sqrt(1 - rho^2) * w_f,   w_f ~ CN(0, 1)

    power_f  = mean_i |s_f[i]|^2    (k=1 -> Swerling 1, exponential power;
                                     k=2 -> Swerling 3, chi-squared-4 power)
    factor_f = sqrt(power_f)        (E[factor^2] = 1: mean POWER preserved)

`frame_corr` (rho) is the frame-to-frame VOLTAGE correlation; the resulting
power correlation is rho^2. rho = 0 reproduces the classic
scan-to-scan-independent Swerling 1/3 cases; rho > 0 produces fade bursts
whose decorrelation time is -1/ln(rho^2) frames.
"""

from __future__ import annotations

import numpy as np

VALID_MODELS = ("swerling1", "swerling3")
_N_SCATTERERS = {"swerling1": 1, "swerling3": 2}


def validate_fluctuation(fluctuation) -> tuple[str, float]:
    """Validate a Target.fluctuation dict; return (model, frame_corr)."""
    if not isinstance(fluctuation, dict):
        raise ValueError(
            f"fluctuation must be a dict, got {type(fluctuation).__name__}"
        )
    unknown = set(fluctuation) - {"model", "frame_corr"}
    if unknown:
        raise ValueError(f"unknown fluctuation keys: {sorted(unknown)}")
    model = fluctuation.get("model")
    if model not in VALID_MODELS:
        raise ValueError(
            f"fluctuation.model must be one of {list(VALID_MODELS)}, "
            f"got {model!r}"
        )
    frame_corr = float(fluctuation.get("frame_corr", 0.0))
    if not 0.0 <= frame_corr < 1.0:
        raise ValueError(
            f"fluctuation.frame_corr must be in [0, 1), got {frame_corr}"
        )
    return model, frame_corr


def amplitude_factors(
    model: str,
    n_frames: int,
    rng: np.random.Generator,
    frame_corr: float = 0.0,
) -> np.ndarray:
    """Unit-mean-power amplitude factor sequence of length n_frames.

    Prefix property: with a fresh identically-seeded rng,
    `amplitude_factors(model, m, rng)[:n] == amplitude_factors(model, n, rng)`
    for m >= n — draws are consumed strictly frame-by-frame, which is what
    lets the generator regrow a cached sequence without changing history.
    """
    k = _N_SCATTERERS[model]
    rho = frame_corr
    out = np.empty(n_frames, dtype=np.float64)
    s = None
    for f in range(n_frames):
        w = (
            rng.standard_normal(k) + 1j * rng.standard_normal(k)
        ) / np.sqrt(2.0)
        if s is None:
            s = w
        else:
            s = rho * s + np.sqrt(1.0 - rho * rho) * w
        out[f] = np.sqrt(np.mean(np.abs(s) ** 2))
    return out
