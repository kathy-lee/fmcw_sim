import numpy as np
import pytest

from processing.cfar import cfar_2d
from _synth import (
    make_noise_frame, add_target, make_two_targets,
)


def test_ca_cfar_pure_noise_false_alarm_rate():
    """Empirical FA rate over many noise frames should be near pfa."""
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(20):
        frame = make_noise_frame(seed=seed)
        mask = cfar_2d(frame, method="ca", pfa=pfa, window="rect")
        # Trim edges where uniform_filter mode='nearest' biases statistics
        m = mask[5:-5, 15:-15]
        total_cells += m.size
        total_alarms += int(m.sum())
    empirical = total_alarms / total_cells
    # Two-sided check: detector must fire near pfa, not vanish (catches a dead detector).
    assert empirical < 5 * pfa, f"FA rate {empirical:.2e} too high (target {pfa:.0e})"
    assert empirical > pfa / 10, f"FA rate {empirical:.2e} suspiciously low — detector may be dead"


def test_ca_cfar_detects_single_bright_cell():
    frame = make_noise_frame(seed=1)
    frame = add_target(frame, doppler_bin=16, range_bin=200, snr_db=30.0)
    mask = cfar_2d(frame, method="ca", pfa=1e-4, window="rect")
    assert mask[16, 200], "bright cell not detected"


def test_ca_cfar_detects_two_well_separated_targets():
    frame = make_two_targets(targets=[(16, 100, 30.0), (12, 400, 30.0)], seed=2)
    mask = cfar_2d(frame, method="ca", pfa=1e-4, window="rect")
    assert mask[16, 100]
    assert mask[12, 400]


def test_cfar_2d_rejects_unknown_method():
    frame = make_noise_frame(seed=3)
    with pytest.raises(ValueError, match="method"):
        cfar_2d(frame, method="bogus")


def test_cfar_2d_handles_nan_inf_input():
    """NaN/Inf input should not crash CFAR; replaced via nan_to_num guard."""
    frame = make_noise_frame(seed=4)
    frame[0, 0] = np.nan
    frame[0, 1] = np.inf
    frame[0, 2] = -np.inf
    mask = cfar_2d(frame, method="ca", pfa=1e-4, window="rect")
    assert mask.shape == frame.shape
    assert mask.dtype == bool


def test_os_cfar_detects_single_bright_cell():
    frame = make_noise_frame(seed=10)
    frame = add_target(frame, doppler_bin=10, range_bin=120, snr_db=30.0)
    mask = cfar_2d(frame, method="os", pfa=1e-4, window="rect")
    assert mask[10, 120]


def test_os_cfar_detects_close_targets():
    """Close-by targets: OS-CFAR should detect both even when they sit in each
    other's training window. (At 35 dB SNR both are also detected by CA-CFAR;
    the OS-vs-CA discrimination is documented in the spec but not asserted as
    a regression because it depends on tunable training/guard sizes.)
    """
    targets = [(16, 200, 35.0), (16, 206, 35.0)]
    frame = make_two_targets(targets=targets, seed=11)
    mask_os = cfar_2d(frame, method="os", pfa=1e-4, window="rect")
    assert mask_os[16, 200] and mask_os[16, 206], "OS-CFAR should detect both close targets"


def test_os_cfar_pure_noise_false_alarm_rate():
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(10):
        frame = make_noise_frame(seed=100 + seed)
        mask = cfar_2d(frame, method="os", pfa=pfa, window="rect")
        m = mask[5:-5, 15:-15]
        total_cells += m.size
        total_alarms += int(m.sum())
    empirical = total_alarms / total_cells
    assert empirical < 10 * pfa, f"FA rate {empirical:.2e} too high"
    assert empirical > pfa / 20, f"FA rate {empirical:.2e} suspiciously low — detector may be dead"


@pytest.mark.parametrize("method", ["go", "so"])
def test_goso_cfar_detects_single_bright_cell(method):
    frame = make_noise_frame(seed=20)
    frame = add_target(frame, doppler_bin=14, range_bin=300, snr_db=30.0)
    # Synthetic iid noise has no dc_removal / clutter_removal dead zones; pass
    # both as False so the α calibration's noise model matches the test data.
    mask = cfar_2d(frame, method=method, pfa=1e-4, window="rect",
                   dc_removal=False, clutter_removal=False)
    assert mask[14, 300]


@pytest.mark.parametrize("method", ["go", "so"])
def test_goso_cfar_pure_noise_false_alarm_rate(method):
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(10):
        frame = make_noise_frame(seed=200 + seed)
        mask = cfar_2d(frame, method=method, pfa=pfa, window="rect",
                       dc_removal=False, clutter_removal=False)
        m = mask[5:-5, 15:-15]
        total_cells += m.size
        total_alarms += int(m.sum())
    empirical = total_alarms / total_cells
    # Empirical calibration on iid noise should land both GO and SO close to
    # design (within factor 3 either way).
    assert empirical < 3 * pfa, f"FA rate {empirical:.2e} too high"
    assert empirical > pfa / 3, f"FA rate {empirical:.2e} suspiciously low — detector may be dead"


def test_so_cfar_detects_close_targets():
    targets = [(16, 200, 35.0), (16, 208, 35.0)]
    frame = make_two_targets(targets=targets, seed=21)
    mask = cfar_2d(frame, method="so", pfa=1e-4, window="rect",
                   dc_removal=False, clutter_removal=False)
    assert mask[16, 200] and mask[16, 208]


@pytest.mark.parametrize("method", ["ca", "os", "go", "so"])
@pytest.mark.parametrize("dc_removal,clutter_removal", [
    (True, False),    # production default — range bin 0 is a dead column
    (False, True),    # clutter-removal only — doppler-center row is dead
    (True, True),     # both dead lines present
])
def test_cfar_empirical_pfa_hits_design_on_windowed_noise(
    method, dc_removal, clutter_removal,
):
    """Each CFAR variant should hit its design Pfa for the actual processing
    chain in use. With dc_removal/clutter_removal enabled the chain creates
    'dead' cells (range bin 0 and/or doppler-center row); the empirical α
    calibration redistributes the FA budget across the whole map. We count
    alarms across the full map (no edge trim) because that's what the
    calibration is designed to hit.

    This is the regression test for the SO mis-calibration bug — SO is the
    operator most sensitive to dead-line artifacts in the training quadrants.
    """
    from processing.cfar import _windowed_noise_rd_maps

    pfa = 1e-3
    rd_shape = (64, 128)
    # Different seed than calibration's seed=0 → measures generalization, not
    # training-set fit.
    samples = _windowed_noise_rd_maps(
        rd_shape, "hann", dc_removal, clutter_removal,
        n_frames=80, seed=12345,
    )

    total_cells = 0
    total_alarms = 0
    for rd_lin in samples:
        rd_db = 10.0 * np.log10(rd_lin + 1e-12)
        mask = cfar_2d(rd_db, method=method, pfa=pfa, window="hann",
                       dc_removal=dc_removal, clutter_removal=clutter_removal)
        total_cells += mask.size
        total_alarms += int(mask.sum())
    empirical = total_alarms / total_cells
    tag = f"{method} dc={dc_removal} clutter={clutter_removal}"
    assert empirical < 3 * pfa, (
        f"{tag}: FA rate {empirical:.2e} too high (design {pfa:.0e})"
    )
    assert empirical > pfa / 3, (
        f"{tag}: FA rate {empirical:.2e} too low — calibration drifted"
    )
