"""The frozen Q15 chain built on the CMSIS-DSP FFT (libfmcw_cmsis.so),
loaded beside the one built on the project FFT (libfmcw.so). It is the
laptop reference for a board that runs the CMSIS-DSP FFT."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from processing.c_frontend import CFrontend
from processing.c_frontend_onepass import read_params

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "c_frontend"
CMSIS_LIB = CDIR / "libfmcw_cmsis.so"
SCENARIO = "scenarios/scenario_mimo_stream_128.json"

CMSIS_DSP = ROOT / "embedded" / "third_party" / "CMSIS-DSP"

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None,
                               reason="gcc not available")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if not (CMSIS_DSP / "Source").is_dir():
        pytest.skip("CMSIS-DSP submodule not checked out: git submodule "
                    "update --init embedded/third_party/CMSIS-DSP")
    out = tmp_path_factory.mktemp("stream128_cmsis")
    subprocess.run(
        [sys.executable, "scripts/export_for_c.py", "--scenario", SCENARIO,
         "--out", str(out), "--frames", "5"],
        cwd=ROOT, check=True)
    subprocess.run(["make", "-C", str(CDIR), "lib"], check=True)
    subprocess.run(["make", "-C", str(CDIR), "lib-cmsis"], check=True)
    return out


def _frames(exported, dims):
    for f in sorted((exported / "frames").glob("frame_*.bin")):
        yield np.fromfile(f, dtype=np.complex64).reshape(dims)


def test_missing_library_is_named(tmp_path):
    with pytest.raises(OSError, match="no_such_lib.so"):
        CFrontend("unused.cfg", lib_path=tmp_path / "no_such_lib.so")


@needs_gcc
def test_both_libraries_load_side_by_side(exported):
    cfg = exported / "params.cfg"
    with CFrontend(cfg) as own, CFrontend(cfg, lib_path=CMSIS_LIB) as dsp:
        assert own.dims == dsp.dims == (12, 32, 128)


@needs_gcc
def test_each_library_runs_its_own_fft(exported):
    """The two FFTs round differently, so the integer range-Doppler cubes
    cannot be equal; if they were, both handles would be running one FFT."""
    cfg = exported / "params.cfg"
    with CFrontend(cfg) as own, CFrontend(cfg, lib_path=CMSIS_LIB) as dsp:
        adc = next(_frames(exported, own.dims))
        rd_own, rd_dsp = own.q15_rd(adc), dsp.q15_rd(adc)
        assert not np.array_equal(rd_own, rd_dsp)
        # and loading the second library did not change the first
        with CFrontend(cfg) as again:
            assert np.array_equal(again.q15_rd(adc), rd_own)


@needs_gcc
def test_cmsis_reference_is_repeatable(exported):
    cfg = exported / "params.cfg"
    with CFrontend(cfg, lib_path=CMSIS_LIB) as a, \
         CFrontend(cfg, lib_path=CMSIS_LIB) as b:
        for adc in _frames(exported, a.dims):
            assert np.array_equal(a.q15_process_frame(adc),
                                  b.q15_process_frame(adc))


@needs_gcc
def test_cmsis_reference_finds_both_targets(exported):
    """The same loose physical check the project-FFT chain passes: both
    targets appear, the far one near its true angle."""
    params = read_params(exported / "params.cfg")
    with CFrontend(exported / "params.cfg", lib_path=CMSIS_LIB) as dsp:
        det = dsp.q15_process_frame(next(_frames(exported, dsp.dims)))
        rng = det[:, 0] * params["range_per_bin"]
        # near target ~3.0 m at ~3.8 deg; far target ~7.0 m at ~25 deg
        assert np.any(np.abs(rng - 3.0) < 0.5)
        assert np.any(np.abs(rng - 7.0) < 0.5)
        far = det[np.abs(rng - 7.0) < 0.5]
        assert np.any(np.abs(far[:, 2] - 25.0) <= 3.0)
