import numpy as np

from simulator.simulate_cubes import run_export

_SCENARIO = "scenarios/scenario_mimo_fixedpoint_dynrange.json"


def test_run_export_writes_npz(tmp_path):
    paths = run_export(_SCENARIO, tmp_path, frames=2,
                       levels=("adc", "rd_cube"), fp=None)
    assert len(paths) == 2
    for p in paths:
        assert p.endswith(".npz")
    npz = np.load(paths[0])
    assert "adc" in npz
    assert "rd_cube" in npz
    assert "axis_range_m" in npz


def test_run_export_fp_quantizes_adc(tmp_path):
    paths = run_export(_SCENARIO, tmp_path, frames=1,
                       levels=("adc",), fp=True)
    npz = np.load(paths[0])
    # Q_adc output is a quantized grid: far fewer distinct magnitudes than float.
    assert np.unique(np.abs(npz["adc"])).size < npz["adc"].size
