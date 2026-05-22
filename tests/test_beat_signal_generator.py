import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, RadarConfig, Target,
)


C_EM = 299_792_458.0


def _small_cfg(**overrides):
    defaults = dict(
        fc_hz=77e9,
        bandwidth_hz=2e9,
        ramp_duration_s=40e-6,
        chirp_period_s=50e-6,
        n_range=64,
        n_chirps=16,
        noise_power=1e-6,
        leakage_amplitude=0.0,
        seed=0,
    )
    defaults.update(overrides)
    return RadarConfig(**defaults)


# --- RadarConfig.__post_init__ ---

def test_radar_config_post_init_fills_natural_cadence():
    c = RadarConfig()
    assert c.frame_period_s == c.n_chirps * c.chirp_period_s


def test_radar_config_post_init_preserves_explicit_value():
    c = RadarConfig(frame_period_s=0.05)
    assert c.frame_period_s == 0.05


# --- _validate_config ---

@pytest.mark.parametrize("overrides, match", [
    (dict(ramp_duration_s=50e-6, chirp_period_s=40e-6), "chirp_period_s"),
    (dict(frame_period_s=5e-4), "frame_period_s"),
])
def test_validate_config_rejects_inconsistent_timing(overrides, match):
    # n_chirps * chirp_period_s = 16 * 50e-6 = 8e-4 s.
    cfg = _small_cfg(**overrides)
    with pytest.raises(ValueError, match=match):
        FMCWBeatSignalGenerator(cfg)


# --- generate_frame ---

def test_generate_frame_shape_and_dtype():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    assert frame.shape == (cfg.n_chirps, cfg.n_range)
    assert frame.dtype == np.complex64


def test_generate_frame_zero_input_returns_zeros():
    cfg = _small_cfg(noise_power=0.0, leakage_amplitude=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    assert np.allclose(frame, 0.0)


def test_leakage_amplitude_produces_dc_offset_per_chirp():
    cfg = _small_cfg(noise_power=0.0, leakage_amplitude=0.5)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    # Without targets/noise, the frame is exactly leakage_amplitude (constant complex).
    per_chirp_mean = frame.mean(axis=1)
    assert np.allclose(np.real(per_chirp_mean), 0.5, atol=1e-5)
    assert np.allclose(np.imag(per_chirp_mean), 0.0, atol=1e-5)


def test_noise_power_matches_expected_variance():
    cfg = _small_cfg(noise_power=1e-3, leakage_amplitude=0.0,
                     n_range=512, n_chirps=64)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    # E[|x|^2] = sigma_real^2 + sigma_imag^2 = noise_power
    measured = float(np.mean(np.abs(frame) ** 2))
    assert abs(measured - cfg.noise_power) < 0.1 * cfg.noise_power


# --- derived quantities ---

def test_max_unambiguous_range_m_formula():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    expected = C_EM * (gen.fs_hz / 2.0) / (2.0 * gen.slope_hz_per_s)
    assert abs(gen.max_unambiguous_range_m - expected) < 1e-9


def test_max_unambiguous_velocity_mps_formula():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    expected = gen.lambda_m / (4.0 * cfg.chirp_period_s)
    assert abs(gen.max_unambiguous_velocity_mps - expected) < 1e-9


# --- frame timing ---

def test_frame_start_time_s_uses_frame_period_s():
    cfg = _small_cfg(frame_period_s=0.05)
    gen = FMCWBeatSignalGenerator(cfg)
    assert gen.frame_start_time_s(0) == 0.0
    assert gen.frame_start_time_s(10) == pytest.approx(0.5, abs=1e-12)


# --- metadata ---

def test_metadata_includes_config_and_derived():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    md = gen.metadata()
    assert "config" in md
    assert "derived" in md
    for key in ("fs_hz", "slope_hz_per_s", "lambda_m",
                "max_unambiguous_range_m", "max_unambiguous_velocity_mps"):
        assert key in md["derived"], f"missing derived.{key}"
    assert md["config"]["n_range"] == cfg.n_range
    assert md["config"]["n_chirps"] == cfg.n_chirps


# --- ground_truth ---

def test_ground_truth_doppler_bin_zero_for_static_target():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    target = Target(range_m=1.0, velocity_mps=0.0)
    gt = gen.ground_truth([target], frame_index=0)[0]
    # Static target -> zero Doppler -> doppler_bin_shifted == n_chirps/2
    assert abs(gt["doppler_bin_shifted"] - cfg.n_chirps / 2.0) < 1e-9


def test_ground_truth_range_bin_matches_argmax_after_range_fft():
    cfg = _small_cfg(noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    target = Target(range_m=1.0, velocity_mps=0.0)
    frame = gen.generate_frame([target], frame_index=0)
    # Static target: any chirp's range FFT has the peak at the same range bin.
    rfft = np.fft.fft(frame, axis=1).mean(axis=0)
    peak = int(np.argmax(np.abs(rfft[: cfg.n_range // 2])))
    gt = gen.ground_truth([target], frame_index=0)[0]
    assert abs(peak - gt["range_bin"]) <= 1


# --- ClutterRegion ---

def test_clutter_region_valid_inputs():
    from simulator.beat_signal_generator import ClutterRegion
    r = ClutterRegion(range_m_min=4.0, range_m_max=6.0,
                      velocity_mps_min=-0.5, velocity_mps_max=0.5,
                      cnr_db=10.0)
    assert r.range_m_min == 4.0
    assert r.range_m_max == 6.0
    assert r.velocity_mps_min == -0.5
    assert r.velocity_mps_max == 0.5
    assert r.cnr_db == 10.0


def test_clutter_region_unbounded_velocity_defaults():
    from simulator.beat_signal_generator import ClutterRegion
    r = ClutterRegion(range_m_min=4.0, range_m_max=6.0)
    assert r.velocity_mps_min is None
    assert r.velocity_mps_max is None
    assert r.cnr_db == 10.0


def test_clutter_region_rejects_inverted_range():
    from simulator.beat_signal_generator import ClutterRegion
    with pytest.raises(ValueError, match="range_m"):
        ClutterRegion(range_m_min=6.0, range_m_max=4.0)


def test_clutter_region_rejects_inverted_velocity():
    from simulator.beat_signal_generator import ClutterRegion
    with pytest.raises(ValueError, match="velocity_mps"):
        ClutterRegion(range_m_min=4.0, range_m_max=6.0,
                      velocity_mps_min=0.5, velocity_mps_max=-0.5)


def test_clutter_region_rejects_negative_cnr():
    from simulator.beat_signal_generator import ClutterRegion
    with pytest.raises(ValueError, match="cnr_db"):
        ClutterRegion(range_m_min=4.0, range_m_max=6.0, cnr_db=-1.0)


def test_radar_config_clutter_defaults_to_empty_list():
    cfg = RadarConfig()
    assert cfg.clutter == []


def test_generate_frame_with_empty_clutter_matches_no_clutter():
    """Backward compat: with cfg.clutter=[], output is bit-identical to a
    fresh generator with the same seed."""
    cfg = RadarConfig(seed=42)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame(targets=[], frame_index=0)

    cfg2 = RadarConfig(seed=42)
    gen2 = FMCWBeatSignalGenerator(cfg2)
    frame2 = gen2.generate_frame(targets=[], frame_index=0)

    np.testing.assert_array_equal(frame, frame2)
