"""The speed-up board's parameter generator: the DoA steering table and the
refusal of element positions the table cannot represent."""
import re
import subprocess
import sys
from pathlib import Path

GEN = (Path(__file__).resolve().parents[1] / "embedded"
       / "nucleo_f446re_frontend_speedup" / "gen_frontend_params.py")


def _cfg(tmp_path, pos_wl):
    n = len(pos_wl)
    text = "\n".join([
        f"n_virtual={n}", "n_chirps=2", "n_samples=128", "n_tx=1",
        f"n_rx={n}", "doa_n_sources=1", "cfar_guard_d=2", "cfar_guard_r=4",
        "cfar_training_d=4", "cfar_training_r=8", "min_range_bins=6",
        "range_per_bin=0.15", "pfa=0.0001", "cfar_alpha=3.0",
        "adc_full_scale=0.5", "velocity_axis=-1.0,1.0",
        "pos_wl=" + ",".join(repr(x) for x in pos_wl)]) + "\n"
    path = tmp_path / "params.cfg"
    path.write_text(text)
    return path


def _run(tmp_path, pos_wl):
    out = tmp_path / "frontend_params.h"
    proc = subprocess.run(
        [sys.executable, str(GEN), "--cfg", str(_cfg(tmp_path, pos_wl)),
         "--out", str(out)], capture_output=True, text=True)
    return proc, out


def test_nonuniform_positions_refused(tmp_path):
    proc, out = _run(tmp_path, [0.0, 0.5, 1.1])
    assert proc.returncode != 0
    assert "element 2" in proc.stdout + proc.stderr
    assert not out.exists()


def test_doa_table_emitted(tmp_path):
    proc, out = _run(tmp_path, [0.5 * e for e in range(12)])
    assert proc.returncode == 0, proc.stderr
    text = out.read_text()
    block = text.split("FRONTEND_DOA_W[181] = {", 1)[1].split("};", 1)[0]
    entries = re.findall(r"\{[^{}]*\}", block)
    assert len(entries) == 181
    assert entries[90] == "{1.0f, 0.0f}"
