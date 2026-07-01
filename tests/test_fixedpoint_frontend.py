import numpy as np

from processing.config import FixedPointConfig
from processing.range_doppler import range_doppler


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((64, 256)) + 1j * rng.standard_normal((64, 256))
            ).astype(np.complex64)


def test_range_doppler_fp_none_is_bit_identical():
    f = _frame()
    np.testing.assert_array_equal(range_doppler(f), range_doppler(f, fp=None))


def test_range_doppler_wide_fp_close_to_float():
    f = _frame()
    wide = FixedPointConfig(cube_bits=24, rd_bits=24)
    out_f = range_doppler(f)
    out_q = range_doppler(f, fp=wide)
    rel = np.max(np.abs(out_q - out_f)) / np.max(np.abs(out_f))
    assert rel < 1e-3


def test_range_doppler_narrow_fp_changes_output():
    f = _frame()
    narrow = FixedPointConfig(cube_bits=4, rd_bits=4)
    assert not np.allclose(range_doppler(f, fp=narrow), range_doppler(f))


# --- Task 4: integrate (NCI) Q_mag/Q_acc ---

from processing.point_cloud import integrate


def _stack(seed=1, n_v=12):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_v, 64, 128))
            + 1j * rng.standard_normal((n_v, 64, 128))).astype(np.complex64)


def test_integrate_fp_none_bit_identical():
    s = _stack()
    np.testing.assert_array_equal(integrate(s), integrate(s, fp=None))


def test_integrate_wide_fp_close_to_float():
    s = _stack()
    wide = FixedPointConfig(mag_bits=24, acc_bits=32)
    out_f, out_q = integrate(s), integrate(s, fp=wide)
    assert np.max(np.abs(out_q - out_f)) / np.max(out_f) < 1e-3


# --- Task 5: process_mimo_frame Q_adc + threading ---

from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header
from processing.mimo_frame import process_mimo_frame


def _mimo_frame_and_header(fi=0):
    cfg, targets = load_scenario("scenarios/scenario_mimo_crossing.json")
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame_mimo(targets, frame_index=fi)
    header = build_header(frame=frame, frame_id=fi, topic="t",
                          metadata=gen.metadata(), ground_truth=[],
                          timestamp_unix_s=0.0)
    return frame, header


def test_process_mimo_frame_fp_none_bit_identical():
    frame, header = _mimo_frame_and_header()
    a = process_mimo_frame(frame, header)
    b = process_mimo_frame(frame, header, fp=None)
    np.testing.assert_array_equal(a["power_db"], b["power_db"])
    np.testing.assert_array_equal(a["mask"], b["mask"])


def test_process_mimo_frame_fp_runs_and_differs():
    frame, header = _mimo_frame_and_header()
    fp = FixedPointConfig(adc_bits=6, cube_bits=6, rd_bits=6, mag_bits=8,
                          adc_full_scale=float(np.max(np.abs(frame))))
    out = process_mimo_frame(frame, header, fp=fp)
    assert out["power_db"].shape == process_mimo_frame(frame, header)["power_db"].shape
    assert not np.allclose(out["power_db"],
                           process_mimo_frame(frame, header)["power_db"])
