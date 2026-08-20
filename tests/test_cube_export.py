import dataclasses
import json

import numpy as np

from processing.config import FixedPointConfig
from processing.doa import _GRID_DEG
from processing.fixed_point import quantize
from processing.cube_export import extract_cubes, save_cubes
from simulator.beat_signal_generator import FMCWBeatSignalGenerator, load_scenario
from simulator.wire import build_header

_SCENARIO = "scenarios/scenario_mimo_fixedpoint_dynrange.json"


def _frame_and_header():
    cfg, targets = load_scenario(_SCENARIO)
    gen = FMCWBeatSignalGenerator(cfg)
    frame = gen.generate_frame_mimo(targets, frame_index=0)
    header = build_header(frame=frame, frame_id=0, topic="cube",
                          metadata=gen.metadata(), ground_truth=[],
                          timestamp_unix_s=0.0)
    return frame, header


def test_shapes_and_keys_float():
    frame, header = _frame_and_header()
    nv, nc, nr = frame.shape
    cubes = extract_cubes(frame, header, fp=None)
    assert cubes["adc"].shape == (nv, nc, nr)
    assert cubes["range_cube"].shape == (nv, nc, nr // 2)
    assert cubes["rd_cube"].shape == (nv, nc, nr // 2)
    assert cubes["power"].shape == (nc, nr // 2)
    assert cubes["rad"].shape == (_GRID_DEG.size, nc, nr // 2)
    assert set(cubes["axes"]) == {"range_m", "velocity_mps", "azimuth_deg"}


def test_adc_is_float_when_fp_none():
    frame, header = _frame_and_header()
    cubes = extract_cubes(frame, header, fp=None, levels=("adc",))
    np.testing.assert_array_equal(cubes["adc"], np.asarray(frame))


def test_adc_is_qadc_when_fp_set():
    frame, header = _frame_and_header()
    fp = dataclasses.replace(FixedPointConfig(),
                             adc_full_scale=float(np.max(np.abs(frame))))
    cubes = extract_cubes(frame, header, fp=fp, levels=("adc",))
    expected = quantize(frame, fp.adc_bits, fp.adc_full_scale, is_complex=True,
                        rounding=fp.rounding, overflow=fp.overflow)
    np.testing.assert_array_equal(cubes["adc"], expected)


def test_levels_selection():
    frame, header = _frame_and_header()
    cubes = extract_cubes(frame, header, levels=("adc", "rd_cube"))
    assert set(cubes) == {"adc", "rd_cube", "axes", "meta"}


def test_unknown_level_raises():
    frame, header = _frame_and_header()
    try:
        extract_cubes(frame, header, levels=("adc", "bogus"))
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown level")


def test_save_round_trip(tmp_path):
    frame, header = _frame_and_header()
    cubes = extract_cubes(frame, header, levels=("adc", "rd_cube", "rad"))
    path = tmp_path / "frame_0000.npz"
    save_cubes(cubes, path)
    npz = np.load(path)
    np.testing.assert_array_equal(npz["adc"], cubes["adc"])
    np.testing.assert_array_equal(npz["rd_cube"], cubes["rd_cube"])
    np.testing.assert_array_equal(npz["axis_range_m"], cubes["axes"]["range_m"])
    meta = json.loads(str(npz["meta"]))
    assert meta["n_tx"] == cubes["meta"]["n_tx"]
