"""Extended-target model: K coherent point scatterers over a range extent.

An extended target is realised as K point scatterers spread across
`range_extent_m`, each with a complex-Gaussian (Rayleigh-magnitude) voltage
drawn once from a dedicated RNG stream (`RadarConfig.extent_seed`). The
magnitudes are normalised so the total power equals the equivalent point
target's: sum_i |a_i|^2 = base_amplitude^2. K=1 collapses to a single
scatterer at offset 0 with |a| = base_amplitude — i.e. the point target.

Placement is random-uniform over [-range_extent_m/2, +range_extent_m/2];
magnitudes are Rayleigh (each a_i is complex-Gaussian). Both are hardcoded
(not exposed) — the orthogonal temporal amplitude knob is the Swerling
`fluctuation` field, applied to the whole object per frame elsewhere.
"""

from __future__ import annotations

import numpy as np


def validate_extent(extent) -> tuple[int, float]:
    """Validate a Target.extent dict; return (n_scatterers, range_extent_m)."""
    if not isinstance(extent, dict):
        raise ValueError(
            f"extent must be a dict, got {type(extent).__name__}"
        )
    unknown = set(extent) - {"n_scatterers", "range_extent_m"}
    if unknown:
        raise ValueError(f"unknown extent keys: {sorted(unknown)}")
    n = extent.get("n_scatterers")
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"extent.n_scatterers must be int >= 1, got {n!r}")
    range_extent_m = float(extent.get("range_extent_m", 0.0))
    if range_extent_m < 0.0:
        raise ValueError(
            f"extent.range_extent_m must be >= 0, got {range_extent_m}"
        )
    return n, range_extent_m


def make_scatterers(
    extent,
    base_amplitude: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_r_m, amplitudes) for one extended target.

    delta_r_m: (K,) real range offsets (m) about the target centre.
    amplitudes: (K,) complex scatterer voltages, normalised so
        sum |a_i|^2 == base_amplitude^2.

    K=1 returns offset [0.0] and amplitude [base_amplitude] exactly — the
    point target. Drawn strictly from `rng`, so identical seeds reproduce.
    """
    k, range_extent_m = validate_extent(extent)
    if k == 1:
        return (np.zeros(1, dtype=np.float64),
                np.array([base_amplitude], dtype=np.complex128))
    # Random-uniform placement over the centred extent.
    delta_r = rng.uniform(-0.5 * range_extent_m, 0.5 * range_extent_m, size=k)
    # Complex-Gaussian voltages (Rayleigh magnitudes).
    raw = (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / np.sqrt(2.0)
    power = float(np.sum(np.abs(raw) ** 2))
    # Normalise to equal total power: sum|a|^2 = base_amplitude^2.
    a = raw * (base_amplitude / np.sqrt(power))
    return delta_r.astype(np.float64), a.astype(np.complex128)
