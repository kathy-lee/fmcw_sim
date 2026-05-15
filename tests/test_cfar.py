import numpy as np
import pytest

from radar_postproc.cfar import cfar_2d
from radar_postproc.tests._synth import (
    make_noise_frame, add_target, make_two_targets,
)


def test_ca_cfar_pure_noise_false_alarm_rate():
    """Empirical FA rate over many noise frames should be near pfa."""
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(20):
        frame = make_noise_frame(seed=seed)
        mask = cfar_2d(frame, method="ca", pfa=pfa)
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
    mask = cfar_2d(frame, method="ca", pfa=1e-4)
    assert mask[16, 200], "bright cell not detected"


def test_ca_cfar_detects_two_well_separated_targets():
    frame = make_two_targets(targets=[(16, 100, 30.0), (12, 400, 30.0)], seed=2)
    mask = cfar_2d(frame, method="ca", pfa=1e-4)
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
    mask = cfar_2d(frame, method="ca", pfa=1e-4)
    assert mask.shape == frame.shape
    assert mask.dtype == bool


def test_os_cfar_detects_single_bright_cell():
    frame = make_noise_frame(seed=10)
    frame = add_target(frame, doppler_bin=10, range_bin=120, snr_db=30.0)
    mask = cfar_2d(frame, method="os", pfa=1e-4)
    assert mask[10, 120]


def test_os_cfar_detects_close_targets():
    """Close-by targets: OS-CFAR should detect both even when they sit in each
    other's training window. (At 35 dB SNR both are also detected by CA-CFAR;
    the OS-vs-CA discrimination is documented in the spec but not asserted as
    a regression because it depends on tunable training/guard sizes.)
    """
    targets = [(16, 200, 35.0), (16, 206, 35.0)]
    frame = make_two_targets(targets=targets, seed=11)
    mask_os = cfar_2d(frame, method="os", pfa=1e-4)
    assert mask_os[16, 200] and mask_os[16, 206], "OS-CFAR should detect both close targets"


def test_os_cfar_pure_noise_false_alarm_rate():
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(10):
        frame = make_noise_frame(seed=100 + seed)
        mask = cfar_2d(frame, method="os", pfa=pfa)
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
    mask = cfar_2d(frame, method=method, pfa=1e-4)
    assert mask[14, 300]


@pytest.mark.parametrize("method", ["go", "so"])
def test_goso_cfar_pure_noise_false_alarm_rate(method):
    pfa = 1e-3
    total_cells = 0
    total_alarms = 0
    for seed in range(10):
        frame = make_noise_frame(seed=200 + seed)
        mask = cfar_2d(frame, method=method, pfa=pfa)
        m = mask[5:-5, 15:-15]
        total_cells += m.size
        total_alarms += int(m.sum())
    empirical = total_alarms / total_cells
    # GO is conservative (lower FA); SO is liberal (higher FA but still bounded).
    assert empirical < 20 * pfa, f"FA rate {empirical:.2e} too high"
    # Lower bound catches a dead detector. SO can have an FA rate at or above pfa,
    # GO well below; pfa/50 is permissive enough for both while still rejecting all-False.
    assert empirical > pfa / 50, f"FA rate {empirical:.2e} suspiciously low — detector may be dead"


def test_so_cfar_detects_close_targets():
    targets = [(16, 200, 35.0), (16, 208, 35.0)]
    frame = make_two_targets(targets=targets, seed=21)
    mask = cfar_2d(frame, method="so", pfa=1e-4)
    assert mask[16, 200] and mask[16, 208]
