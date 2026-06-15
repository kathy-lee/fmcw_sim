"""Tests for simulator.extended_target — K-scatterer extended targets."""

from __future__ import annotations

import numpy as np
import pytest


def test_validate_extent_accepts_valid_dict():
    from simulator.extended_target import validate_extent
    k, ext = validate_extent({"n_scatterers": 10, "range_extent_m": 0.6})
    assert k == 10
    assert ext == pytest.approx(0.6)


def test_validate_extent_rejects_non_dict():
    from simulator.extended_target import validate_extent
    with pytest.raises(ValueError, match="dict"):
        validate_extent([10, 0.6])


def test_validate_extent_rejects_unknown_keys():
    from simulator.extended_target import validate_extent
    with pytest.raises(ValueError, match="unknown"):
        validate_extent({"n_scatterers": 5, "range_extent_m": 0.3, "k": 1})


def test_validate_extent_rejects_bad_n_scatterers():
    from simulator.extended_target import validate_extent
    with pytest.raises(ValueError, match="n_scatterers"):
        validate_extent({"n_scatterers": 0, "range_extent_m": 0.3})


def test_validate_extent_rejects_negative_extent():
    from simulator.extended_target import validate_extent
    with pytest.raises(ValueError, match="range_extent_m"):
        validate_extent({"n_scatterers": 5, "range_extent_m": -0.1})


def test_make_scatterers_equal_total_power():
    """Sum of |a_i|^2 equals base_amplitude^2 (equal-total-power norm)."""
    from simulator.extended_target import make_scatterers
    base = 0.0009
    rng = np.random.default_rng(1)
    dr, a = make_scatterers({"n_scatterers": 12, "range_extent_m": 0.6},
                            base, rng)
    assert dr.shape == (12,)
    assert a.shape == (12,)
    assert float(np.sum(np.abs(a) ** 2)) == pytest.approx(base ** 2, rel=1e-9)


def test_make_scatterers_offsets_within_extent():
    from simulator.extended_target import make_scatterers
    rng = np.random.default_rng(2)
    dr, _ = make_scatterers({"n_scatterers": 200, "range_extent_m": 0.6},
                            1.0, rng)
    # Offsets are centred and span at most +/- range_extent_m/2.
    assert np.all(np.abs(dr) <= 0.6 / 2 + 1e-9)


def test_make_scatterers_deterministic_for_same_seed():
    from simulator.extended_target import make_scatterers
    dr1, a1 = make_scatterers({"n_scatterers": 8, "range_extent_m": 0.3},
                              1.0, np.random.default_rng(7))
    dr2, a2 = make_scatterers({"n_scatterers": 8, "range_extent_m": 0.3},
                              1.0, np.random.default_rng(7))
    np.testing.assert_array_equal(dr1, dr2)
    np.testing.assert_array_equal(a1, a2)


def test_make_scatterers_k1_is_point_target():
    """K=1 collapses to a single scatterer at offset 0 with |a|=base."""
    from simulator.extended_target import make_scatterers
    dr, a = make_scatterers({"n_scatterers": 1, "range_extent_m": 0.6},
                            0.0009, np.random.default_rng(3))
    assert dr.shape == (1,)
    assert dr[0] == pytest.approx(0.0)
    assert abs(a[0]) == pytest.approx(0.0009)


def test_make_scatterers_magnitudes_are_rayleigh_ish():
    """Many complex-Gaussian scatterers: magnitude distribution is Rayleigh,
    so the brightest is several times the median (a few dominant centres)."""
    from simulator.extended_target import make_scatterers
    rng = np.random.default_rng(11)
    _, a = make_scatterers({"n_scatterers": 2000, "range_extent_m": 1.0},
                           1.0, rng)
    mags = np.abs(a)
    assert mags.max() > 3.0 * np.median(mags)
