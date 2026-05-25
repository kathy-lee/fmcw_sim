import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from processing.range_doppler import range_doppler
from simulator.beat_signal_generator import (
    FMCWBeatSignalGenerator, RadarConfig, Target,
)


def _small_cfg(**overrides):
    cfg = RadarConfig(
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
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_output_shape_is_half_range_x_n_chirps():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    rd = range_doppler(frame)
    assert rd.shape == (cfg.n_chirps, cfg.n_range // 2)


def test_zero_input_returns_zeros():
    frame = np.zeros((8, 32), dtype=np.complex64)
    rd = range_doppler(frame, dc_removal=False, clutter_removal=False)
    assert np.allclose(np.abs(rd), 0.0)


def test_single_target_peak_at_expected_bin():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    # Range 1.5 m sits well inside the 2.4 m unambiguous range for this cfg.
    target = Target(range_m=1.5, velocity_mps=4.0)
    frame = gen.generate_frame([target], frame_index=0)
    rd = range_doppler(frame, dc_removal=False, clutter_removal=False)
    power = np.abs(rd)
    peak = np.unravel_index(np.argmax(power), power.shape)
    gt = gen.ground_truth([target], frame_index=0)[0]
    assert abs(peak[1] - gt["range_bin"]) <= 1
    assert abs(peak[0] - gt["doppler_bin_shifted"]) <= 1


def test_hann_window_reduces_sidelobes_vs_rect():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    target = Target(range_m=1.5, velocity_mps=4.0)
    frame = gen.generate_frame([target], frame_index=0)
    rd_hann = range_doppler(frame, window="hann",
                            dc_removal=False, clutter_removal=False)
    rd_rect = range_doppler(frame, window="rect",
                            dc_removal=False, clutter_removal=False)
    p_hann = np.abs(rd_hann)
    p_rect = np.abs(rd_rect)
    peak_d, peak_r = np.unravel_index(np.argmax(p_hann), p_hann.shape)
    # Mask out the peak ±2 bins to look at sidelobes only.
    p_hann_side = p_hann.copy()
    p_rect_side = p_rect.copy()
    p_hann_side[max(0, peak_d - 2):peak_d + 3,
                max(0, peak_r - 2):peak_r + 3] = 0
    p_rect_side[max(0, peak_d - 2):peak_d + 3,
                max(0, peak_r - 2):peak_r + 3] = 0
    assert p_hann_side.max() < p_rect_side.max()


def test_dc_removal_suppresses_zero_range_spike():
    cfg = _small_cfg(leakage_amplitude=0.5)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame([], frame_index=0)
    rd_off = range_doppler(frame, dc_removal=False, clutter_removal=False)
    rd_on = range_doppler(frame, dc_removal=True, clutter_removal=False)
    # Power at range_bin=0, summed across doppler.
    p_off = (np.abs(rd_off[:, 0]) ** 2).sum()
    p_on = (np.abs(rd_on[:, 0]) ** 2).sum()
    assert p_on < 0.01 * p_off, (
        f"DC removal should suppress zero-range power by >100x; "
        f"got off={p_off:.3e}, on={p_on:.3e}"
    )


def test_clutter_removal_suppresses_zero_doppler_stripe():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    # Ranges chosen to sit comfortably within the 2.4 m unambig limit.
    static = Target(range_m=1.0, velocity_mps=0.0, amplitude=1.0)
    mover = Target(range_m=2.0, velocity_mps=5.0, amplitude=1.0)
    frame = gen.generate_frame([static, mover], frame_index=0)
    rd_off = range_doppler(frame, dc_removal=True, clutter_removal=False)
    rd_on = range_doppler(frame, dc_removal=True, clutter_removal=True)
    n_d = cfg.n_chirps
    # Power on the zero-Doppler row (doppler bin = n_chirps/2 after fftshift).
    zd_off = (np.abs(rd_off[n_d // 2, :]) ** 2).sum()
    zd_on = (np.abs(rd_on[n_d // 2, :]) ** 2).sum()
    assert zd_on < 0.1 * zd_off, (
        f"Clutter removal should suppress zero-Doppler power by >10x; "
        f"got off={zd_off:.3e}, on={zd_on:.3e}"
    )


def test_clutter_removal_preserves_moving_targets():
    cfg = _small_cfg()
    gen = FMCWBeatSignalGenerator(cfg)
    mover = Target(range_m=2.0, velocity_mps=5.0, amplitude=1.0)
    frame = gen.generate_frame([mover], frame_index=0)
    rd_off = range_doppler(frame, dc_removal=True, clutter_removal=False)
    rd_on = range_doppler(frame, dc_removal=True, clutter_removal=True)
    peak_off = np.abs(rd_off).max()
    peak_on = np.abs(rd_on).max()
    # Moving-target peak power should change by at most ~3 dB (factor of ~2).
    ratio = peak_on / peak_off
    assert 0.5 < ratio < 2.0, (
        f"Moving-target peak shifted too much: ratio={ratio:.3f}"
    )
