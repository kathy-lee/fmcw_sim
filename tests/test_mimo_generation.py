"""MIMO generation — RadarConfig fields, steering, TDM.

here covers only the config surface; defaults must leave the single-channel
path (n_tx = n_rx = 1) byte-for-byte unchanged (guarded by
tests/test_generate_frame_byte_identical.py).
"""
import numpy as np

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, RadarConfig


def test_radarconfig_mimo_defaults_single_channel():
    cfg = RadarConfig()
    assert cfg.n_tx == 1
    assert cfg.n_rx == 1
    assert cfg.mimo_scheme == "tdm"


def test_radarconfig_mimo_configurable():
    cfg = RadarConfig(n_tx=3, n_rx=4)
    assert cfg.n_tx == 3
    assert cfg.n_rx == 4
    assert cfg.n_tx * cfg.n_rx == 12      # virtual array size V


# --- virtual array + steering ---

def test_virtual_element_positions_contiguous_ula():
    # 3 TX x 4 RX, rx spacing lambda/2 -> contiguous 12-element lambda/2 ULA at
    # positions 0, 0.5, 1.0, ..., 5.5 wavelengths.
    cfg = RadarConfig(n_tx=3, n_rx=4, rx_spacing_wl=0.5)
    gen = FMCWBeatSignalGenerator(cfg)
    pos = gen.virtual_element_positions()
    np.testing.assert_allclose(pos, 0.5 * np.arange(12))


def test_steering_vector_matches_analytic():
    cfg = RadarConfig(n_tx=3, n_rx=4, rx_spacing_wl=0.5)
    gen = FMCWBeatSignalGenerator(cfg)
    theta = np.deg2rad(20.0)
    sv = gen.steering_vector(theta)
    # phase = 2*pi * pos * sin(theta), pos = 0.5*e  ->  pi*e*sin(theta)
    expected = np.exp(1j * np.pi * np.arange(12) * np.sin(theta))
    assert sv.shape == (12,)
    np.testing.assert_allclose(sv, expected)


def test_steering_vector_boresight_is_all_ones():
    cfg = RadarConfig(n_tx=3, n_rx=4)
    gen = FMCWBeatSignalGenerator(cfg)
    sv = gen.steering_vector(0.0)        # boresight -> no phase progression
    np.testing.assert_allclose(sv, np.ones(12, dtype=complex))


# --- generate_frame_mimo (TDM + steering) ---

from simulator.beat_signal_generator import Target


def test_generate_frame_mimo_single_channel_collapses_to_generate_frame():
    # Noise off: compare the deterministic signal model. (MIMO uses a separate
    # frame-addressed noise path, option C, so the single-channel case only
    # matches with noise off.)
    cfg = RadarConfig(noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    targets = [Target(x=4.0, vx=0.5, amplitude=1.0)]
    stack = gen.generate_frame_mimo(targets, frame_index=3)
    frame = gen.generate_frame(targets, frame_index=3)
    assert stack.shape == (1, cfg.n_chirps, cfg.n_range)
    np.testing.assert_array_equal(stack[0], frame)


def test_generate_frame_mimo_shape_3x4():
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo([Target(x=4.0, vx=0.5, amplitude=1.0)])
    assert stack.shape == (12, cfg.n_chirps, cfg.n_range)


def test_generate_frame_mimo_static_target_encodes_steering():
    # A STATIC target (no Doppler) isolates the steering phase: across virtual
    # elements the only phase difference is the array steering vector.
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    theta = np.deg2rad(15.0)
    r = 5.0
    t = Target(x=r * np.cos(theta), y=r * np.sin(theta),
               vx=0.0, vy=0.0, amplitude=1.0)
    stack = gen.generate_frame_mimo([t])
    sv = gen.steering_vector(t.azimuth_rad)
    e0 = stack[0]
    peak = np.unravel_index(np.argmax(np.abs(e0)), e0.shape)
    ratios = np.array([stack[e][peak] / stack[0][peak] for e in range(12)])
    np.testing.assert_allclose(ratios, sv / sv[0], rtol=1e-4, atol=1e-4)


def test_max_unambiguous_velocity_mimo_divides_by_n_tx():
    cfg = RadarConfig(n_tx=3, n_rx=4)
    gen = FMCWBeatSignalGenerator(cfg)
    assert np.isclose(gen.max_unambiguous_velocity_mps_mimo,
                      gen.max_unambiguous_velocity_mps / 3.0)


# --- frame-addressed per-element noise (option C) ---

def test_mimo_noise_is_frame_addressed_pure():
    # Unlike legacy generate_frame (call-ordered self.rng), the MIMO noise is a
    # pure function of (frame_index, element): same frame -> same noise, no matter
    # the call order or generator instance.
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=1e-4)
    g1 = FMCWBeatSignalGenerator(cfg)
    g2 = FMCWBeatSignalGenerator(cfg)
    a = g1.generate_frame_mimo([], frame_index=5)     # no targets -> noise only
    g1.generate_frame_mimo([], frame_index=2)          # advance/intersperse calls
    a_again = g1.generate_frame_mimo([], frame_index=5)
    b = g2.generate_frame_mimo([], frame_index=5)       # fresh generator
    np.testing.assert_array_equal(a, a_again)
    np.testing.assert_array_equal(a, b)


def test_mimo_noise_independent_across_frames_and_elements():
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=1e-4)
    gen = FMCWBeatSignalGenerator(cfg)
    f5 = gen.generate_frame_mimo([], frame_index=5)
    f6 = gen.generate_frame_mimo([], frame_index=6)
    assert not np.array_equal(f5, f6)                   # different frames differ
    assert not np.array_equal(f5[0], f5[1])             # different elements differ


def test_mimo_noise_off_is_deterministic_zero_without_targets():
    cfg = RadarConfig(n_tx=2, n_rx=2, noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo([], frame_index=0)
    np.testing.assert_array_equal(stack, np.zeros_like(stack))


def test_mimo_noise_variance_matches_noise_power():
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=1e-4)
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo([], frame_index=0)   # noise only
    assert np.isclose(np.mean(np.abs(stack) ** 2), 1e-4, rtol=0.05)


# --- cv_xy 2-D motion + off-boresight scenario ---

import math

from simulator.beat_signal_generator import load_scenario
from simulator.motion_models import azimuth_at, range_at


def test_cv_xy_range_at():
    t = Target(x=3.0, y=4.0, vx=1.0, vy=0.0, motion_model="cv_xy")
    r0, v0 = range_at(t, 0, 0.1)
    assert np.isclose(r0, 5.0)              # |(3, 4)|
    assert np.isclose(v0, 0.6)             # (3*1 + 4*0)/5


def test_cv_xy_azimuth_evolves():
    t = Target(x=10.0, y=0.0, vx=0.0, vy=5.0, motion_model="cv_xy")
    assert np.isclose(azimuth_at(t, 0, 0.1), 0.0)          # starts on boresight
    assert np.isclose(azimuth_at(t, 10, 0.1), math.atan2(5.0, 10.0))


def test_generate_frame_mimo_cv_xy_steering_evolves_between_frames():
    cfg = RadarConfig(n_tx=3, n_rx=4, noise_power=0.0)
    gen = FMCWBeatSignalGenerator(cfg)
    t = Target(x=10.0, y=0.0, vx=0.0, vy=5.0,
               motion_model="cv_xy", amplitude=1.0)
    s0 = gen.generate_frame_mimo([t], frame_index=0)
    s20 = gen.generate_frame_mimo([t], frame_index=20)
    # frame 0: boresight (azimuth 0, radial velocity 0) -> elements identical
    np.testing.assert_allclose(s0[1], s0[0], rtol=1e-6, atol=1e-6)
    # later: target has moved off-boresight -> steering makes elements differ
    assert not np.allclose(s20[1], s20[0])


def test_mimo_2d_scenario_loads_and_generates():
    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
    assert cfg.n_tx == 3 and cfg.n_rx == 4
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo(targets, frame_index=0)
    assert stack.shape == (12, cfg.n_chirps, cfg.n_range)
