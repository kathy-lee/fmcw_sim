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


# --- integrate (NCI) Q_acc ---

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
    wide = FixedPointConfig(acc_bits=32)
    out_f, out_q = integrate(s), integrate(s, fp=wide)
    assert np.max(np.abs(out_q - out_f)) / np.max(out_f) < 1e-3


# --- process_mimo_frame Q_adc + threading ---

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
    fp = FixedPointConfig(adc_bits=6, cube_bits=6, rd_bits=6,
                          adc_full_scale=float(np.max(np.abs(frame))))
    out = process_mimo_frame(frame, header, fp=fp)
    assert out["power_db"].shape == process_mimo_frame(frame, header)["power_db"].shape
    assert not np.allclose(out["power_db"],
                           process_mimo_frame(frame, header)["power_db"])


def test_fixedpoint_nci_does_not_flood_cfar():
    """Regression: the fixed-point front end must not flood CFAR with false alarms.

    The NCI stage used to quantize the power |rd|^2 block-scaled to the strong
    target peak; squaring doubles the dynamic range, so a 16-bit power word could
    not represent the noise floor and rounded 20-99% of cells to exactly 0. Those
    zeros became -120 dB holes (10*log10(1e-12)) that deflated CA-CFAR's local
    noise estimate, exploding the detection count from ~12 (float) to 200-900.

    The fix computes |rd|^2 exactly from the Q_rd cube and quantizes only the
    (wide) accumulator, so the noise floor survives and the float-calibrated CFAR
    threshold keeps giving a nominal false-alarm rate. This pins the count near
    the float baseline on a weak+strong dynamic-range scenario.
    """
    cfg, targets = load_scenario("scenarios/scenario_mimo_fixedpoint_dynrange.json")
    gen = FMCWBeatSignalGenerator(cfg)
    for fi in range(4):
        frame = gen.generate_frame_mimo(targets, frame_index=fi)
        header = build_header(frame=frame, frame_id=fi, topic="t",
                              metadata=gen.metadata(), ground_truth=[],
                              timestamp_unix_s=0.0)
        fp = FixedPointConfig(adc_bits=12, cube_bits=14, acc_bits=32,
                              adc_full_scale=float(np.max(np.abs(frame))))
        n_float = int(process_mimo_frame(frame, header)["mask"].sum())
        out_fp = process_mimo_frame(frame, header, fp=fp)
        n_fp = int(out_fp["mask"].sum())
        # No zeroed noise floor -> no -120 dB holes.
        assert not np.any(out_fp["power_db"] <= -119.0), (fi, "floor zeroed")
        # No false-alarm flood: stay within a small factor of the float count,
        # not the 20-90x explosion the bug produced.
        assert n_fp <= 3 * n_float + 10, (fi, n_float, n_fp)
