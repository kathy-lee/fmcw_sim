"""Tests for the CMSIS-DSP FFT comparison's results reader."""
import importlib.util
import sys
from pathlib import Path

READER = (Path(__file__).resolve().parents[1]
          / "embedded/nucleo_f446re_fft_cmsis/read_results.py")
_spec = importlib.util.spec_from_file_location("nucleo_fft_cmsis_reader", READER)
reader = importlib.util.module_from_spec(_spec)
# dataclass looks its module up here while the class is being created.
sys.modules[_spec.name] = reader
_spec.loader.exec_module(reader)

parse = reader.parse
check = reader.check
report = reader.report

FIXTURE = """\
# nucleo-fft-cmsis 1
# arith q15
# dsp_loopunroll 1
# clk 180000000
# fpu 1
# flash_ws 5
# flash_acr 0x00000705
OUT n=128 impl=own vec=dc bin0_re=1000 bin0_im=0 other_peak=0
OUT n=128 impl=cmsis vec=dc bin0_re=1000 bin0_im=0 other_peak=0
OUT n=32 impl=own vec=dc bin0_re=1000 bin0_im=0 other_peak=0
OUT n=32 impl=cmsis vec=dc bin0_re=1000 bin0_im=0 other_peak=0
CMP n=128 vec=radar max_abs_diff=2 rms_diff=0.71 own_peak=812 cmsis_peak=811
CMP n=32 vec=radar max_abs_diff=1 rms_diff=0.50 own_peak=700 cmsis_peak=700
TIME n=128 impl=own iters=1000 copy_cyc=900.0 total_cyc=31000.0 fft_cyc=30100.0
TIME n=128 impl=cmsis iters=1000 copy_cyc=900.0 total_cyc=15950.0 fft_cyc=15050.0
TIME n=32 impl=own iters=1000 copy_cyc=300.0 total_cyc=5300.0 fft_cyc=5000.0
TIME n=32 impl=cmsis iters=1000 copy_cyc=300.0 total_cyc=4300.0 fft_cyc=4000.0
MEM heap=1000 stack_peak=744 free_start=120000
END
"""

FLOAT_FIXTURE = (FIXTURE.replace("arith q15", "arith float")
                 .replace("bin0_re=1000", "bin0_re=128000.0"))


def test_valid_capture_has_no_problems():
    assert check(parse(FIXTURE), "q15") == []


def test_wrong_clock_refused():
    text = FIXTURE.replace("clk 180000000", "clk 84000000")
    assert any("clk" in p for p in check(parse(text), "q15"))


def test_fpu_off_refused():
    text = FIXTURE.replace("# fpu 1", "# fpu 0")
    assert any("fpu" in p for p in check(parse(text), "q15"))


def test_loopunroll_off_refused():
    text = FIXTURE.replace("dsp_loopunroll 1", "dsp_loopunroll 0")
    assert any("loopunroll" in p for p in check(parse(text), "q15"))


def test_arith_mismatch_fails():
    assert any("arith" in p for p in check(parse(FIXTURE), "float"))


def test_missing_row_fails():
    text = "\n".join(l for l in FIXTURE.splitlines()
                     if not l.startswith("TIME n=32 impl=cmsis")) + "\n"
    assert any("TIME" in p for p in check(parse(text), "q15"))


def test_missing_end_fails():
    text = FIXTURE.replace("END\n", "")
    assert any("END" in p for p in check(parse(text), "q15"))


def test_scale_mismatch_is_reported():
    text = FIXTURE.replace("n=128 impl=cmsis vec=dc bin0_re=1000",
                           "n=128 impl=cmsis vec=dc bin0_re=500")
    assert any("scale" in p for p in check(parse(text), "q15"))


def test_one_lsb_q15_dc_difference_is_not_a_scale_problem():
    text = FIXTURE.replace("n=128 impl=cmsis vec=dc bin0_re=1000",
                           "n=128 impl=cmsis vec=dc bin0_re=999")
    assert check(parse(text), "q15") == []


def test_float_scale_uses_relative_tolerance():
    close = FLOAT_FIXTURE.replace("n=128 impl=cmsis vec=dc bin0_re=128000.0",
                                  "n=128 impl=cmsis vec=dc bin0_re=128000.01")
    far = FLOAT_FIXTURE.replace("n=128 impl=cmsis vec=dc bin0_re=128000.0",
                                "n=128 impl=cmsis vec=dc bin0_re=128001.0")
    assert check(parse(close), "float") == []
    assert any("scale" in p for p in check(parse(far), "float"))


def test_last_complete_run_is_used():
    bad_run = FIXTURE.replace("clk 180000000", "clk 84000000")
    cut_short = "# nucleo-fft-cmsis 1\n# arith q15\n"
    assert check(parse(bad_run + FIXTURE + cut_short), "q15") == []


def test_report_shows_ratio_and_memory(tmp_path):
    (tmp_path / "stream_q15_nucleo.txt").write_text(FIXTURE)
    (tmp_path / "size_q15.txt").write_text(
        "   text    data     bss     dec     hex filename\n"
        "  37872     476    3024   41372    a19c build/q15/x.elf\n")
    text = report(tmp_path)
    assert "0.50" in text      # 128-point CMSIS/own cycles
    assert "0.80" in text      # 32-point CMSIS/own cycles
    assert "38348" in text     # flash = text + data
    assert "3500" in text      # static RAM = data + bss
    assert "744" in text       # stack peak


def test_report_keeps_long_difference_values_apart(tmp_path):
    text = FIXTURE.replace("max_abs_diff=2 rms_diff=0.71",
                           "max_abs_diff=0.0737305 rms_diff=0.00655588")
    (tmp_path / "stream_q15_nucleo.txt").write_text(text)
    row = next(l for l in report(tmp_path).splitlines() if l.split()[:1] == ["128"])
    assert len(row.split()) == 6
