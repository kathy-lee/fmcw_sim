"""Tests for simulator.fluctuation_models — Swerling RCS fluctuation."""

from __future__ import annotations

import numpy as np
import pytest


def test_swerling1_factors_have_unit_mean_power():
    from simulator.fluctuation_models import amplitude_factors
    rng = np.random.default_rng(123)
    f = amplitude_factors("swerling1", 20000, rng)
    power = f ** 2
    assert power.mean() == pytest.approx(1.0, abs=0.05)
    # Swerling 1 power is Exp(1): variance 1.
    assert power.var() == pytest.approx(1.0, abs=0.15)


def test_swerling3_factors_have_unit_mean_power_and_milder_fades():
    from simulator.fluctuation_models import amplitude_factors
    rng = np.random.default_rng(123)
    f = amplitude_factors("swerling3", 20000, rng)
    power = f ** 2
    assert power.mean() == pytest.approx(1.0, abs=0.05)
    # Swerling 3 power is chi-squared-4 scaled to mean 1: variance 0.5.
    assert power.var() == pytest.approx(0.5, abs=0.10)


def test_factor_sequence_is_deterministic_for_same_seed():
    from simulator.fluctuation_models import amplitude_factors
    f1 = amplitude_factors("swerling1", 100, np.random.default_rng(7),
                           frame_corr=0.6)
    f2 = amplitude_factors("swerling1", 100, np.random.default_rng(7),
                           frame_corr=0.6)
    np.testing.assert_array_equal(f1, f2)


def test_factor_sequence_prefix_property():
    """A longer sequence from the same seed starts with the shorter one.

    The generator's cache regrows sequences by regenerating from a fresh
    identically-seeded rng; history must not change when it does.
    """
    from simulator.fluctuation_models import amplitude_factors
    f_short = amplitude_factors("swerling1", 50, np.random.default_rng(7),
                                frame_corr=0.6)
    f_long = amplitude_factors("swerling1", 200, np.random.default_rng(7),
                               frame_corr=0.6)
    np.testing.assert_allclose(f_long[:50], f_short)


def test_frame_corr_sets_power_autocorrelation():
    from simulator.fluctuation_models import amplitude_factors
    rng = np.random.default_rng(42)
    f = amplitude_factors("swerling1", 50000, rng, frame_corr=0.8)
    p = f ** 2
    d0 = p[:-1] - p.mean()
    d1 = p[1:] - p.mean()
    rho_power = float((d0 * d1).mean() / p.var())
    # Voltage correlation rho -> power correlation rho^2 = 0.64.
    assert rho_power == pytest.approx(0.64, abs=0.05)


def test_zero_frame_corr_factors_are_uncorrelated():
    from simulator.fluctuation_models import amplitude_factors
    rng = np.random.default_rng(42)
    f = amplitude_factors("swerling1", 50000, rng, frame_corr=0.0)
    p = f ** 2
    d0 = p[:-1] - p.mean()
    d1 = p[1:] - p.mean()
    rho_power = float((d0 * d1).mean() / p.var())
    assert rho_power == pytest.approx(0.0, abs=0.05)


def test_validate_fluctuation_accepts_valid_dict():
    from simulator.fluctuation_models import validate_fluctuation
    model, rho = validate_fluctuation(
        {"model": "swerling3", "frame_corr": 0.9})
    assert model == "swerling3"
    assert rho == pytest.approx(0.9)


def test_validate_fluctuation_defaults_frame_corr_to_zero():
    from simulator.fluctuation_models import validate_fluctuation
    model, rho = validate_fluctuation({"model": "swerling1"})
    assert model == "swerling1"
    assert rho == 0.0


def test_validate_fluctuation_rejects_unknown_model():
    from simulator.fluctuation_models import validate_fluctuation
    with pytest.raises(ValueError, match="model"):
        validate_fluctuation({"model": "swerling2"})


def test_validate_fluctuation_rejects_out_of_range_corr():
    from simulator.fluctuation_models import validate_fluctuation
    with pytest.raises(ValueError, match="frame_corr"):
        validate_fluctuation({"model": "swerling1", "frame_corr": 1.0})
    with pytest.raises(ValueError, match="frame_corr"):
        validate_fluctuation({"model": "swerling1", "frame_corr": -0.1})


def test_validate_fluctuation_rejects_unknown_keys():
    from simulator.fluctuation_models import validate_fluctuation
    with pytest.raises(ValueError, match="unknown"):
        validate_fluctuation({"model": "swerling1", "rho": 0.5})


def test_validate_fluctuation_rejects_non_dict():
    from simulator.fluctuation_models import validate_fluctuation
    with pytest.raises(ValueError, match="dict"):
        validate_fluctuation("swerling1")
