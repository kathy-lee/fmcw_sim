"""NCI integration, TDM compensation, end-to-end, point cloud."""
import numpy as np
import pytest

from processing.point_cloud import (
    integrate, mimo_point_cloud, tdm_doppler_compensation, to_point_cloud)


def test_integrate_sums_channel_power():
    rng = np.random.default_rng(0)
    stack = (rng.standard_normal((12, 8, 16))
             + 1j * rng.standard_normal((12, 8, 16))).astype(np.complex64)
    out = integrate(stack)
    assert out.shape == (8, 16)
    np.testing.assert_allclose(
        out, np.sum(np.abs(stack) ** 2, axis=0), rtol=1e-6)


def test_integrate_single_channel_is_abs_squared():
    rng = np.random.default_rng(1)
    stack = (rng.standard_normal((1, 4, 4))
             + 1j * rng.standard_normal((1, 4, 4))).astype(np.complex64)
    np.testing.assert_allclose(integrate(stack), np.abs(stack[0]) ** 2, rtol=1e-6)


def test_tdm_compensation_identity_when_single_tx():
    snap = np.array([1 + 2j, 3 - 1j, 0.5j, -2.0], dtype=complex)
    # single TX (n_tx=1) -> no inter-TX stagger -> identity
    out = tdm_doppler_compensation(snap, doppler_bin=40, n_chirps=64,
                                   n_tx=1, n_rx=4)
    np.testing.assert_array_equal(out, snap)


@pytest.mark.parametrize("method", ["bartlett", "capon", "music", "esprit"])
def test_doa_on_mimo_stack_matches_ground_truth_azimuth(method):
    from simulator.beat_signal_generator import (
        load_scenario, FMCWBeatSignalGenerator)
    from processing.range_doppler import range_doppler
    from processing.doa import estimate_doa

    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
    cfg.noise_power = 0.0
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo(targets, frame_index=0)
    rd = np.stack([range_doppler(stack[v]) for v in range(stack.shape[0])])
    power = integrate(rd)
    d, r = np.unravel_index(np.argmax(power), power.shape)
    snap = tdm_doppler_compensation(rd[:, d, r], d, cfg.n_chirps,
                                    cfg.n_tx, cfg.n_rx)
    est = estimate_doa(snap, gen.virtual_element_positions(),
                       method=method, n_sources=1)[0]
    gt_az = gen.ground_truth(targets, frame_index=0)[0]["azimuth_deg"]
    assert abs(est - gt_az) < 2.0


def test_to_point_cloud_places_point_at_range_azimuth():
    velocity_axis = np.linspace(-5.0, 5.0, 64)
    cloud = to_point_cloud([(50.0, 32, 30.0, 10.0)],
                           range_per_bin=0.1, velocity_axis=velocity_axis)
    assert cloud.shape == (1, 4)
    rng = 50.0 * 0.1
    np.testing.assert_allclose(
        cloud[0, :2],
        [rng * np.cos(np.deg2rad(30)), rng * np.sin(np.deg2rad(30))], atol=1e-9)
    assert cloud[0, 2] == velocity_axis[32]
    assert cloud[0, 3] == 10.0


def test_mimo_point_cloud_places_target_at_xy():
    from simulator.beat_signal_generator import (
        load_scenario, FMCWBeatSignalGenerator)
    from processing.range_doppler import range_doppler

    cfg, targets = load_scenario("scenarios/scenario_mimo.json")
    cfg.noise_power = 0.0
    gen = FMCWBeatSignalGenerator(cfg)
    stack = gen.generate_frame_mimo(targets, frame_index=0)
    rd = np.stack([range_doppler(stack[v]) for v in range(stack.shape[0])])
    power = integrate(rd)
    mask = np.zeros(power.shape, dtype=bool)
    d, r = np.unravel_index(np.argmax(power), power.shape)
    mask[d, r] = True
    velocity_axis = gen.velocity_axis_mps() / cfg.n_tx     # TDM velocity scaling
    range_per_bin = float(gen.range_axis_m()[1])
    cloud = mimo_point_cloud(
        rd, mask, gen.virtual_element_positions(), range_per_bin, velocity_axis,
        cfg.n_tx, cfg.n_rx, doa_method="music", n_sources=1)
    assert cloud.shape[0] >= 1
    # GT target at (x, y) = (8, 2)
    assert abs(cloud[0, 0] - 8.0) < 0.3 and abs(cloud[0, 1] - 2.0) < 0.3
