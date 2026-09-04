"""The one-pass composition against the frozen Q15 chain, via ctypes, on
frames exported from the shrunk streaming scenario. Equality is exact: the
two paths run identical C arithmetic, so any difference is an ordering bug,
not precision loss."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from processing.c_frontend import CFrontend
from processing.c_frontend_onepass import CFrontendOnepass, read_params

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "scenarios/scenario_mimo_stream_128.json"

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None,
                               reason="gcc not available")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("stream128_export")
    subprocess.run(
        [sys.executable, "scripts/export_for_c.py", "--scenario", SCENARIO,
         "--out", str(out), "--frames", "5"],
        cwd=ROOT, check=True)
    subprocess.run(["make", "-C", str(ROOT / "c_frontend"), "lib"], check=True)
    return out


@needs_gcc
def test_dims(exported):
    with CFrontendOnepass(exported / "params.cfg") as op:
        assert op.dims == (12, 32, 128)


@needs_gcc
def test_onepass_matches_frozen_chain(exported):
    frames = sorted((exported / "frames").glob("frame_*.bin"))
    with CFrontendOnepass(exported / "params.cfg") as op, \
         CFrontend(exported / "params.cfg") as fe:
        V, C, S = op.dims
        for f in frames:
            adc = np.fromfile(f, dtype=np.complex64).reshape(V, C, S)
            rows_q = op.quantize(adc)
            assert op.last_sat == 0          # the exporter left headroom
            det_op = op.process(rows_q)
            det_ref = fe.q15_process_frame(adc)
            assert det_op.shape == det_ref.shape
            assert det_op.shape[0] >= 1
            # identical C arithmetic on identical quantized input: exact
            assert np.array_equal(det_op, det_ref)


@needs_gcc
def test_detections_land_near_ground_truth(exported):
    """Loose physical sanity so the exact-equality test cannot pass
    vacuously on a broken scenario: both targets appear, angles are on the
    grid near truth."""
    cfg = read_params(exported / "params.cfg")
    frames = sorted((exported / "frames").glob("frame_*.bin"))
    with CFrontendOnepass(exported / "params.cfg") as op:
        V, C, S = op.dims
        adc = np.fromfile(frames[0], dtype=np.complex64).reshape(V, C, S)
        det = op.process(op.quantize(adc))
        rng = det[:, 0] * cfg["range_per_bin"]
        # near target ~3.0 m at ~3.8 deg; far target ~7.0 m at ~25 deg
        assert np.any(np.abs(rng - 3.0) < 0.5)
        assert np.any(np.abs(rng - 7.0) < 0.5)
        far = det[np.abs(rng - 7.0) < 0.5]
        assert np.any(np.abs(far[:, 2] - 25.0) <= 3.0)
