"""Golden validation of the C front end against the Python reference.

Task 7 adds the exporter test; Task 9 adds the build fixture, the C unit-test
wrapper, and the RD-cube comparison.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from processing.range_doppler import range_doppler

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "c_frontend"

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None,
                               reason="gcc not available")


def _read_cfg(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, val = line.split("=", 1)
        out[key] = val
    return out


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("c_export")
    subprocess.run(
        [sys.executable, "scripts/export_for_c.py",
         "--scenario", "scenarios/scenario_mimo.json",
         "--out", str(out), "--frames", "3"],
        cwd=ROOT, check=True)
    return out


def test_export_frames_and_config(exported):
    cfg = _read_cfg(exported / "params.cfg")
    required = {"n_virtual", "n_chirps", "n_samples", "n_tx", "n_rx",
                "doa_n_sources", "cfar_guard_d", "cfar_guard_r",
                "cfar_training_d", "cfar_training_r", "min_range_bins",
                "range_per_bin", "pfa", "cfar_alpha",
                "velocity_axis", "pos_wl"}
    assert required == set(cfg)

    V, C, S = int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"])
    assert V == int(cfg["n_tx"]) * int(cfg["n_rx"])
    assert len(cfg["velocity_axis"].split(",")) == C
    assert len(cfg["pos_wl"].split(",")) == V
    assert float(cfg["cfar_alpha"]) > 1.0

    frames = sorted((exported / "frames").glob("frame_*.bin"))
    assert len(frames) == 3
    for f in frames:
        data = np.fromfile(f, dtype=np.complex64)
        assert data.size == V * C * S
        assert np.all(np.isfinite(data.view(np.float32)))


@pytest.fixture(scope="module")
def binary():
    subprocess.run(["make", "-C", str(CDIR)], check=True)
    return CDIR / "fmcw_frontend"


@needs_gcc
def test_c_unit_tests():
    subprocess.run(["make", "-C", str(CDIR), "test"], check=True)


@needs_gcc
def test_rd_cube_matches_python(binary, exported, tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    subprocess.run(
        [str(binary), "--config", str(exported / "params.cfg"),
         "--frames", str(exported / "frames"), "--dump", str(dump)],
        check=True)

    cfg = _read_cfg(exported / "params.cfg")
    V, C, S = int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"])

    for i in range(3):
        raw = np.fromfile(exported / "frames" / f"frame_{i:04d}.bin",
                          dtype=np.complex64).reshape(V, C, S)
        ref = np.stack([range_doppler(raw[v]) for v in range(V)])
        got = np.fromfile(dump / f"rd_{i:04d}.bin",
                          dtype=np.complex64).reshape(V, C, S // 2)
        scale = np.max(np.abs(ref))
        err = np.max(np.abs(got.astype(np.complex128) - ref))
        assert err <= 1e-4 * scale, f"frame {i}: max err {err:.3g} vs scale {scale:.3g}"
