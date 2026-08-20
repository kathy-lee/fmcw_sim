import numpy as np
import pytest

from processing.fixed_point import (
    quantize, block_full_scale, quantize_block,
    quantization_noise_power, cube_bytes)


def test_step_size_and_levels():
    # full_scale=1.0, 4-bit signed -> step = 1/8, levels at k/8
    x = np.array([0.0, 0.124, 0.13, 0.49, 0.51])
    q = quantize(x, n_bits=4, full_scale=1.0)
    np.testing.assert_allclose(q, np.array([0.0, 0.125, 0.125, 0.5, 0.5]), atol=1e-9)


def test_saturation_clips_to_range():
    # 4-bit signed: qmax=7 -> +7/8; qmin=-8 -> -1.0
    q = quantize(np.array([10.0, -10.0]), n_bits=4, full_scale=1.0, overflow="saturate")
    np.testing.assert_allclose(q, np.array([7 / 8, -1.0]))


def test_wrap_overflow_wraps_twos_complement():
    # 4-bit: q for 9 -> wraps into [-8,7]; check it differs from saturate and is in range
    q = quantize(np.array([1.2]), n_bits=4, full_scale=1.0, overflow="wrap")
    assert -1.0 <= q[0] <= 7 / 8


def test_truncate_vs_nearest():
    x = np.array([0.24])      # step 0.125 -> x/step = 1.92
    near = quantize(x, n_bits=4, full_scale=1.0, rounding="nearest")  # round 1.92 -> 2
    trunc = quantize(x, n_bits=4, full_scale=1.0, rounding="truncate")  # trunc -> 1
    np.testing.assert_allclose(near, [0.25])
    np.testing.assert_allclose(trunc, [0.125])


def test_complex_quantized_componentwise():
    z = np.array([0.13 + 0.51j])
    q = quantize(z, n_bits=4, full_scale=1.0, is_complex=True)
    np.testing.assert_allclose(q, [0.125 + 0.5j])


def test_large_n_bits_near_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    q = quantize(x, n_bits=24, full_scale=8.0)
    assert np.max(np.abs(q - x)) < (8.0 / 2 ** 23)


def test_block_full_scale_complex_uses_max_component():
    z = np.array([3 + 1j, -1 - 5j])
    assert block_full_scale(z) == 5.0


def test_quantize_block_zero_block_unchanged():
    x = np.zeros(8)
    np.testing.assert_array_equal(quantize_block(x, 8, scaling="block"), x)


def test_quantize_block_fixed_requires_full_scale():
    with pytest.raises(ValueError):
        quantize_block(np.ones(4), 8, scaling="fixed")


def test_quantization_noise_power():
    assert quantization_noise_power(0.125) == pytest.approx(0.125 ** 2 / 12)


def test_cube_bytes():
    # 12 ch x 64 chirps x 128 range, 16-bit I/Q -> 12*64*128 * 2 * 16/8 bytes
    assert cube_bytes(12, 64, 128, 16) == 12 * 64 * 128 * 2 * 2.0


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        quantize(np.ones(3), n_bits=1, full_scale=1.0)
    with pytest.raises(ValueError):
        quantize(np.ones(3), n_bits=8, full_scale=0.0)


# --- Task 2: FixedPointConfig ---

from processing.config import FixedPointConfig


def test_fixedpoint_config_defaults_valid():
    fp = FixedPointConfig()
    assert fp.adc_bits == 12 and fp.cube_bits == 16 and fp.acc_bits == 32
    assert fp.scaling == "block"


def test_fixedpoint_config_bad_bits_raise():
    with pytest.raises(ValueError):
        FixedPointConfig(cube_bits=1)


def test_fixedpoint_config_bad_mode_raises():
    with pytest.raises(ValueError):
        FixedPointConfig(scaling="auto")
    with pytest.raises(ValueError):
        FixedPointConfig(rounding="bankers")


def test_fixedpoint_config_fixed_requires_full_scales():
    with pytest.raises(ValueError):
        FixedPointConfig(scaling="fixed")          # no full-scales
    ok = FixedPointConfig(scaling="fixed", cube_full_scale=1.0, rd_full_scale=1.0,
                          acc_full_scale=1.0)
    assert ok.scaling == "fixed"
