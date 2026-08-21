"""Golden validation of the C front end against the Python reference.

Task 7 adds the exporter test; Task 9 adds the build fixture, the C unit-test
wrapper, and the RD-cube comparison.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


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
