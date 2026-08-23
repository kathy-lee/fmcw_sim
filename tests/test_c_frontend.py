"""Golden validation of the C front end against the Python reference.

Covers the exporter contract, the C unit-test suite (via make), and
stage-level comparisons of the C binary's output against the Python
pipeline on exported frames.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

from processing.doa import estimate_doa
from processing.point_cloud import integrate, tdm_doppler_compensation
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
                "range_per_bin", "pfa", "cfar_alpha", "adc_full_scale",
                "velocity_axis", "pos_wl"}
    assert required == set(cfg)

    V, C, S = int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"])
    assert V == int(cfg["n_tx"]) * int(cfg["n_rx"])
    assert len(cfg["velocity_axis"].split(",")) == C
    assert len(cfg["pos_wl"].split(",")) == V
    assert float(cfg["cfar_alpha"]) > 1.0
    assert float(cfg["adc_full_scale"]) > 0.0

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


@pytest.fixture(scope="module")
def c_run(binary, exported, tmp_path_factory):
    """One full C run over the exported frames: CSV + per-stage dumps."""
    out = tmp_path_factory.mktemp("c_run")
    dump = out / "dump"
    dump.mkdir()
    subprocess.run(
        [str(binary), "--config", str(exported / "params.cfg"),
         "--frames", str(exported / "frames"),
         "--out", str(out / "detections.csv"), "--dump", str(dump)],
        check=True)
    return out


def _dims(cfg):
    return int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"])


def _python_rd_power(exported, cfg, i):
    V, C, S = _dims(cfg)
    raw = np.fromfile(exported / "frames" / f"frame_{i:04d}.bin",
                      dtype=np.complex64).reshape(V, C, S)
    rd = np.stack([range_doppler(raw[v]) for v in range(V)])
    return rd, integrate(rd)


def _cfar_reference(cfg, power):
    """Python-side mask plus borderline classification.

    Noise via scipy uniform_filter (the authentic Python operator) but with
    the exported alpha — identical to the alpha the C binary used, and
    itself validated by test_export_frames_and_config. Cells whose
    cell/threshold ratio sits within 1e-4 of 1.0 are 'borderline': a float
    rounding difference can legitimately flip them, so mask comparisons
    exempt them.
    """
    Gd, Gr = int(cfg["cfar_guard_d"]), int(cfg["cfar_guard_r"])
    Td, Tr = int(cfg["cfar_training_d"]), int(cfg["cfar_training_r"])
    alpha = float(cfg["cfar_alpha"])
    lin = power + 1e-12
    fd, fr = 2 * (Gd + Td) + 1, 2 * (Gr + Tr) + 1
    gd, gr = 2 * Gd + 1, 2 * Gr + 1
    sum_full = scipy.ndimage.uniform_filter(lin, size=(fd, fr), mode="nearest") * fd * fr
    sum_inner = scipy.ndimage.uniform_filter(lin, size=(gd, gr), mode="nearest") * gd * gr
    noise = (sum_full - sum_inner) / (fd * fr - gd * gr)
    ratio = lin / (alpha * noise)
    borderline = np.abs(ratio - 1.0) <= 1e-4
    mask = ratio > 1.0
    mask[:, :int(cfg["min_range_bins"])] = False
    return mask, borderline


@needs_gcc
def test_power_map_matches_python(c_run, exported):
    cfg = _read_cfg(exported / "params.cfg")
    _, C, S = _dims(cfg)
    for i in range(3):
        _, ref = _python_rd_power(exported, cfg, i)
        got = np.fromfile(c_run / "dump" / f"power_{i:04d}.bin",
                          dtype=np.float64).reshape(C, S // 2)
        err = np.max(np.abs(got - ref))
        assert err <= 1e-6 * ref.max(), f"frame {i}: power err {err:.3g}"


@needs_gcc
def test_cfar_mask_matches_python(c_run, exported):
    cfg = _read_cfg(exported / "params.cfg")
    _, C, S = _dims(cfg)
    n_border_total = 0
    for i in range(3):
        _, power = _python_rd_power(exported, cfg, i)
        mask_py, border = _cfar_reference(cfg, power)
        mask_c = np.fromfile(c_run / "dump" / f"mask_{i:04d}.bin",
                             dtype=np.uint8).reshape(C, S // 2).astype(bool)
        n_border_total += int(border.sum())
        bad = (mask_c != mask_py) & ~border
        assert not bad.any(), \
            f"frame {i}: {bad.sum()} non-borderline cell(s) differ at {np.argwhere(bad)[:5]}"
    print(f"\nborderline cells exempted: {n_border_total}")


@needs_gcc
def test_point_cloud_matches_python(c_run, exported):
    cfg = _read_cfg(exported / "params.cfg")
    V, C, S = _dims(cfg)
    pos = np.array([float(v) for v in cfg["pos_wl"].split(",")])
    vel = np.array([float(v) for v in cfg["velocity_axis"].split(",")])
    rpb = float(cfg["range_per_bin"])
    rows = np.genfromtxt(c_run / "detections.csv", delimiter=",", names=True)
    rows = np.atleast_1d(rows)

    for i in range(3):
        rd, power = _python_rd_power(exported, cfg, i)
        mask_py, border = _cfar_reference(cfg, power)
        fr = rows[rows["frame"] == i]
        c_cells = {(int(r["doppler_bin"]), int(r["range_bin"])): r for r in fr}

        # Every C detection must be a Python detection or borderline.
        for (d, r) in c_cells:
            assert mask_py[d, r] or border[d, r], \
                f"frame {i}: C-only detection at d={d}, r={r}"

        # Every non-borderline Python detection must appear with matching values.
        for d, r in np.argwhere(mask_py & ~border):
            key = (int(d), int(r))
            assert key in c_cells, f"frame {i}: missing detection d={d}, r={r}"
            row = c_cells[key]
            snap = tdm_doppler_compensation(rd[:, d, r], d, C,
                                            int(cfg["n_tx"]), int(cfg["n_rx"]))
            snr = 10.0 * np.log10(np.sum(np.abs(snap) ** 2) + 1e-12)
            az = estimate_doa(snap, pos, method="bartlett",
                              n_sources=int(cfg["doa_n_sources"]))[0]
            assert row["azimuth_deg"] == az, \
                f"frame {i} d={d} r={r}: az {row['azimuth_deg']} != {az}"
            for name, ref in (("x", r * rpb * np.cos(np.deg2rad(az))),
                              ("y", r * rpb * np.sin(np.deg2rad(az))),
                              ("v_radial", vel[d]), ("snr_db", snr)):
                tol = 1e-4 * max(abs(ref), 1.0)
                assert abs(row[name] - ref) <= tol, \
                    f"frame {i} d={d} r={r}: {name} {row[name]} != {ref}"


@pytest.fixture(scope="module")
def c_binding(exported):
    subprocess.run(["make", "-C", str(CDIR), "lib"], check=True)
    from processing.c_frontend import CFrontend
    fe = CFrontend(exported / "params.cfg")
    yield fe
    fe.close()


@needs_gcc
def test_c_binding_matches_python(c_binding, exported):
    """In-process, per-stage comparison of the C library against the Python
    reference — each stage's C output against the Python chain, same
    tolerances and borderline rule as the file-based tests."""
    fe = c_binding
    cfg = _read_cfg(exported / "params.cfg")
    V, C, S = _dims(cfg)
    assert fe.dims == (V, C, S)

    for i in range(3):
        rd_py, power_py = _python_rd_power(exported, cfg, i)
        raw = np.fromfile(exported / "frames" / f"frame_{i:04d}.bin",
                          dtype=np.complex64).reshape(V, C, S)

        rd_c = fe.rd(raw)
        scale = np.max(np.abs(rd_py))
        assert np.max(np.abs(rd_c - rd_py)) <= 1e-4 * scale

        power_c = fe.nci(rd_c)
        assert np.max(np.abs(power_c - power_py)) <= 1e-6 * power_py.max()

        mask_py, border = _cfar_reference(cfg, power_py)
        mask_c = fe.cfar(power_c)
        assert not ((mask_c != mask_py) & ~border).any()

        det = fe.pointcloud(rd_c, mask_c)
        cells_c = {(int(r[1]), int(r[0])): r for r in det}
        for d, r in np.argwhere(mask_py & ~border):
            key = (int(d), int(r))
            assert key in cells_c, f"frame {i}: missing detection d={d}, r={r}"
            snap = tdm_doppler_compensation(rd_py[:, d, r], d, C,
                                            int(cfg["n_tx"]), int(cfg["n_rx"]))
            az = estimate_doa(snap, np.array(
                [float(v) for v in cfg["pos_wl"].split(",")]),
                method="bartlett", n_sources=int(cfg["doa_n_sources"]))[0]
            assert cells_c[key][2] == az

        # The facade must agree with its own per-stage chain exactly.
        det_facade = fe.process_frame(raw)
        assert np.array_equal(det_facade, det)


def _sqnr_db(ref, got):
    """Signal-to-quantization-noise ratio of `got` against `ref`, in dB."""
    err = float(np.sum(np.abs(ref - got) ** 2))
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(np.sum(np.abs(ref) ** 2)) / err))


# Floors, not targets: a broken shift or a lost scale factor collapses
# SQNR by tens of dB, so these catch regressions without pinning numbers
# that legitimately drift with the scenario. Measured on this scenario:
# rd 24.0-24.1 dB, power 59.8-62.8 dB.
#
# The gap between the two is not a defect. RD-cube SQNR sums |rd|^2 over
# every cell while the error is about one LSB in each, so the tens of
# thousands of near-empty noise cells dominate the error and contribute
# almost nothing to the signal. The power map weights by |rd|^4, which
# concentrates the measure on target cells where quantization is
# negligible. Same cube, two questions.
#
# The cube's low figure has a physical cause worth remembering: the
# per-stage shifts divide by S*C while a coherent tone gains it back, so
# targets survive at roughly their input amplitude, but thermal noise
# grows only as sqrt(S*C) and lands below one count. The integer noise
# floor is set by rounding, not by the radar's own noise.
SQNR_FLOOR_RD_DB = 20.0
SQNR_FLOOR_POWER_DB = 50.0


@needs_gcc
def test_q15_matches_float_within_sqnr(c_binding, exported):
    """Fixed-point chain against the float chain, per stage.

    Integer arithmetic cannot reproduce float results exactly, so this is
    a statistical comparison rather than the equality the float-vs-Python
    tests use: SQNR floors per stage, mask agreement, and no saturation.
    """
    fe = c_binding
    cfg = _read_cfg(exported / "params.cfg")
    V, C, S = _dims(cfg)
    cells = C * (S // 2)

    for i in range(3):
        raw = np.fromfile(exported / "frames" / f"frame_{i:04d}.bin",
                          dtype=np.complex64).reshape(V, C, S)
        fe.reset_saturations()

        rd_f = fe.rd(raw)
        rd_q = fe.q15_rd(raw)
        sqnr_rd = _sqnr_db(rd_f, fe.dequantize_rd(rd_q))

        power_f = fe.nci(rd_f)
        power_q = fe.q15_nci(rd_q)
        sqnr_power = _sqnr_db(power_f, power_q * fe.power_scale)

        mask_f = fe.cfar(power_f)
        mask_q = fe.q15_cfar(power_q)
        differing = int((mask_f != mask_q).sum())

        sat = fe.saturations()
        print(f"\nframe {i}: rd {sqnr_rd:.1f} dB, power {sqnr_power:.1f} dB, "
              f"mask differs {differing}/{cells} "
              f"(float {int(mask_f.sum())} vs q15 {int(mask_q.sum())} "
              f"detections), saturations {sat}")

        assert sqnr_rd > SQNR_FLOOR_RD_DB
        assert sqnr_power > SQNR_FLOOR_POWER_DB

        # What the cube's SQNR does not tell you: detections survive. CFAR
        # compares a cell against its neighbours, and quantization lifts
        # both, so a rounding-limited floor largely cancels. Measured here:
        # at most one differing cell, counts 10/10, 19/18, 12/12.
        assert differing <= 0.002 * cells
        n_f, n_q = int(mask_f.sum()), int(mask_q.sum())
        assert abs(n_f - n_q) <= max(2, 0.1 * n_f)

        # Nothing may clip: the exporter sets the ADC's full scale from
        # these very frames, with headroom.
        assert sat == (0, 0)

        # The integer facade must agree with its own per-stage chain, the
        # same contract the float facade is held to.
        det = fe.q15_pointcloud(rd_q, mask_q)
        assert np.array_equal(fe.q15_process_frame(raw), det)
