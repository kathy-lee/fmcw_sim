"""Fixed-point quantization primitives (boundary quantization).

Pure NumPy. `quantize` snaps a float/complex array onto a signed n-bit integer
grid spanning +/- full_scale and returns the reconstructed float (q*step) so
downstream float compute is unaffected ("quantize-in-float"). See
docs/superpowers/specs/2026-06-26-fixedpoint-mimo-frontend-design.md.
"""
from __future__ import annotations

import numpy as np

_ROUNDING = ("nearest", "truncate")
_OVERFLOW = ("saturate", "wrap")


def quantize(x, n_bits, full_scale, *, signed=True, rounding="nearest",
             overflow="saturate", is_complex=False):
    """Quantize x to a signed n_bits grid over +/- full_scale; return q*step."""
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")
    if full_scale <= 0:
        raise ValueError(f"full_scale must be > 0, got {full_scale}")
    if rounding not in _ROUNDING:
        raise ValueError(f"rounding must be one of {_ROUNDING}, got {rounding!r}")
    if overflow not in _OVERFLOW:
        raise ValueError(f"overflow must be one of {_OVERFLOW}, got {overflow!r}")

    if is_complex or np.iscomplexobj(x):
        re = quantize(np.real(x), n_bits, full_scale, signed=signed,
                      rounding=rounding, overflow=overflow)
        im = quantize(np.imag(x), n_bits, full_scale, signed=signed,
                      rounding=rounding, overflow=overflow)
        return re + 1j * im

    x = np.asarray(x, dtype=np.float64)
    step = full_scale / (2 ** (n_bits - 1))
    r = x / step
    if rounding == "nearest":          # half away from zero
        q = np.sign(r) * np.floor(np.abs(r) + 0.5)
    else:                              # truncate toward zero
        q = np.trunc(r)

    qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    if overflow == "saturate":
        q = np.clip(q, qmin, qmax)
    else:                             # wrap (two's complement modulo)
        q = ((q - qmin) % (2 ** n_bits)) + qmin
    return q * step


def block_full_scale(x):
    """Per-block full-scale: max(|Re|,|Im|) for complex, max(|x|) for real.
    Returns 0.0 for an empty or all-zero block."""
    if x.size == 0:
        return 0.0
    if np.iscomplexobj(x):
        return float(max(np.max(np.abs(np.real(x))), np.max(np.abs(np.imag(x)))))
    return float(np.max(np.abs(x)))


def quantize_block(x, n_bits, *, scaling, full_scale=None, **kw):
    """Quantize with per-frame block scaling ("block") or caller-fixed ("fixed").
    Block scaling derives full_scale from the data; zero block -> returned as-is."""
    if scaling == "block":
        fs = block_full_scale(x)
        if fs == 0.0:
            return np.array(x, copy=True)
        return quantize(x, n_bits, fs, **kw)
    if scaling == "fixed":
        if full_scale is None:
            raise ValueError("scaling='fixed' requires full_scale")
        return quantize(x, n_bits, full_scale, **kw)
    raise ValueError(f"scaling must be 'block' or 'fixed', got {scaling!r}")


def quantize_stage(x, n_bits, fp, fixed_full_scale=None):
    """Convenience: quantize an internal-stage array per a FixedPointConfig."""
    return quantize_block(x, n_bits, scaling=fp.scaling, full_scale=fixed_full_scale,
                          rounding=fp.rounding, overflow=fp.overflow)


def quantization_noise_power(step):
    """Uniform-quantization noise power = step^2 / 12."""
    return step ** 2 / 12.0


def cube_bytes(n_virtual, n_chirps, n_range_kept, bits_per_component):
    """Analytic radar-cube size in bytes: 2 components (I/Q) per complex sample."""
    return n_virtual * n_chirps * n_range_kept * 2 * (bits_per_component / 8.0)
