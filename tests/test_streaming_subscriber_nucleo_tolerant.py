"""Timing, deadline and link helpers of the tolerant Nucleo subscriber:
pure functions and a serial double, no board."""
import numpy as np
import pytest

from simulator.streaming_subscriber_nucleo import BoardResult
from simulator.streaming_subscriber_nucleo_tolerant import (
    CANDIDATE_BAUDS, TimedLink, board_margin_db, compare_tolerant,
    deadline_table, margin_class, parse_banner_ext, parse_time, phase_means,
    resolve_compare, row_deadline_cyc)

TIME_LINE = ("TIME row_cyc 25600000 finish_cyc 900000000 plane_max_cyc 410000 "
             "power_cyc 3000000 cells_cyc 2000000 det_cyc 5000000 "
             "doa_cyc 890000000 emit_cyc 40000000")


class FakeBoard:
    """Serial double: the lines the board emits, one per readline."""

    def __init__(self, lines):
        self.out = [ln.encode("ascii") + b"\n" for ln in lines]
        self.is_open = True
        self.in_waiting = 0

    def readline(self):
        return self.out.pop(0) if self.out else b""


def _fake_link(board):
    link = object.__new__(TimedLink)
    link.ser = board
    return link


def test_time_fields_parsed():
    t = parse_time(["DET 0 0",
                    "TIME row_cyc 10 finish_cyc 20 plane_max_cyc 5 doa_cyc 7",
                    "END 1"])
    assert t == {"row_cyc": 10, "finish_cyc": 20, "plane_max_cyc": 5,
                 "doa_cyc": 7}


def test_no_time_line_gives_empty():
    assert parse_time(["DET 0 0", "END 1"]) == {}


def test_row_deadline_cyc():
    assert row_deadline_cyc(921600) == 1003906
    assert row_deadline_cyc(2000000) == 462600


def test_deadline_table_marks_misses():
    lines = deadline_table(500000)
    assert len(lines) == len(CANDIDATE_BAUDS)
    assert lines[0].split()[-1] == "ok"       # 921600: deadline 1003906
    assert lines[2].split()[-1] == "MISS"     # 1875000: deadline 493440


def test_deadline_at_the_limit_is_ok():
    assert deadline_table(1003906, bauds=(921600,))[0].split()[-1] == "ok"
    assert deadline_table(1003907, bauds=(921600,))[0].split()[-1] == "MISS"


def test_phase_means():
    assert phase_means([{"ready_s": 2.0, "send_s": 1.0},
                        {"ready_s": 1.0, "send_s": 3.0}]) \
        == {"ready_s": 1.5, "send_s": 2.0}
    assert phase_means([]) == {}


def test_timed_link_returns_result_and_times():
    board = FakeBoard(["DET 0 0", TIME_LINE,
                       "MEM heap 1328 stack_peak 1328 free_start 0",
                       "SAT 0", "END 7", "RDY"])
    res, times = _fake_link(board).read_result()
    assert res.frame_id == 7 and res.n == 0 and res.err is None
    assert (res.row_cyc, res.finish_cyc) == (25600000, 900000000)
    assert times["plane_max_cyc"] == 410000
    assert times["emit_cyc"] == 40000000


def test_timed_link_skips_interleaved_banner():
    # A banner block that ends in RDY without END is not the result.
    board = FakeBoard(["# onepass-frontend 1", "# clk 180000000", "RDY",
                       "DET 0 0", TIME_LINE, "SAT 0", "END 3", "RDY"])
    res, times = _fake_link(board).read_result()
    assert res.frame_id == 3 and times["doa_cyc"] == 890000000


def test_timed_link_stops_at_board_error():
    res, times = _fake_link(FakeBoard(["ERR row 3 17"])).read_result()
    assert res.err == "ERR row 3 17" and times == {}


def test_banner_doa_arith_parsed():
    b = parse_banner_ext(["# dims 12 32 128", "# doa_arith f32"])
    assert b["doa_arith"] == "f32"
    assert b["dims"] == (12, 32, 128)
    assert parse_banner_ext(["# doa_arith f32"])["doa_arith"] == "f32"


def test_banner_fft_impl_parsed():
    assert parse_banner_ext(["# fft_impl cmsis"])["fft_impl"] == "cmsis"


def row(rng, dop, az, snr):
    """One detection row: [range, doppler, az_deg, x, y, v_radial, snr_db]."""
    return [rng, dop, az, 1.5, -0.5, 2.0, snr]


def _ref():
    return np.array([row(10, 3, -20, 30.0), row(25, 7, 5, 25.0),
                     row(40, 12, 31, 18.0)], dtype=np.float64)


def _margins(n):
    return np.full(n, 10.0)


def test_tolerant_identical_is_match():
    verdict, _, counts = compare_tolerant(_ref(), _ref(), _margins(3))
    assert verdict == "MATCH" and counts["flips"] == 0
    empty = np.empty((0, 7))
    assert compare_tolerant(empty, empty, np.empty(0))[0] == "MATCH"


def test_tolerant_small_snr_difference_is_close():
    board = _ref()
    board[1, 6] += 0.05
    verdict, _, counts = compare_tolerant(board, _ref(), _margins(3))
    assert verdict == "CLOSE"
    assert counts["max_snr_diff_db"] == pytest.approx(0.05, abs=1e-9)


def test_tolerant_large_snr_difference_is_mismatch():
    board = _ref()
    board[1, 6] += 2.0
    assert compare_tolerant(board, _ref(), _margins(3))[0] == "MISMATCH"


def test_tolerant_az_two_steps_is_mismatch():
    board = _ref()
    board[0, 2] += 2
    board[0, 6] += 0.05          # so the exact comparison does not pass
    assert compare_tolerant(board, _ref(), _margins(3))[0] == "MISMATCH"
    board = _ref()
    board[0, 2] += 1
    board[0, 6] += 0.05
    verdict, _, counts = compare_tolerant(board, _ref(), _margins(3))
    assert verdict == "CLOSE" and counts["flips"] == 1


def _with_extra_board_row():
    return np.vstack([_ref(), [row(50, 20, 0, 12.0)]])


def test_tolerant_marginal_extra_board_detection_is_close():
    margins = np.array([10.0, 10.0, 10.0, 0.1])
    verdict, _, counts = compare_tolerant(_with_extra_board_row(), _ref(),
                                          margins)
    assert verdict == "CLOSE" and counts["board_only"] == 1
    assert counts["max_board_only_margin_db"] == pytest.approx(0.1)


def test_tolerant_strong_extra_board_detection_is_mismatch():
    margins = np.array([10.0, 10.0, 10.0, 6.0])
    assert compare_tolerant(_with_extra_board_row(), _ref(),
                            margins)[0] == "MISMATCH"


def test_tolerant_ref_only_limit():
    verdict, _, counts = compare_tolerant(_ref()[:2], _ref(), _margins(2))
    assert verdict == "CLOSE" and counts["ref_only"] == 1
    verdict, _, counts = compare_tolerant(_ref()[:1], _ref(), _margins(1))
    assert verdict == "MISMATCH" and counts["ref_only"] == 2


def test_board_margin_db():
    res = BoardResult(dets=[(10, 3, 70, 1.5, -0.5, 2.0, 30.0, 200, 100.0)])
    assert board_margin_db(res) == pytest.approx([3.0103], abs=1e-4)
    assert board_margin_db(BoardResult()).shape == (0,)


def test_resolve_compare():
    assert resolve_compare("auto", {"fft_impl": "cmsis"}) == "tolerant"
    assert resolve_compare("auto", {"fft_impl": "own"}) == "exact"
    assert resolve_compare("auto", {}) == "exact"
    assert resolve_compare("exact", {"fft_impl": "cmsis"}) == "exact"
    assert resolve_compare("tolerant", {"fft_impl": "own"}) == "tolerant"


def test_read_banner_returns_lines_up_to_rdy():
    board = FakeBoard(["# dims 12 32 128", "# fft_impl own", "RDY", "DET 0 0"])
    lines = _fake_link(board).read_banner(settle_s=0.0)
    assert lines == ["# dims 12 32 128", "# fft_impl own"]


def test_margin_class():
    assert margin_class(0.5) == "0-3 dB"
    assert margin_class(3.0) == "3-6 dB"
    assert margin_class(19.9) == "10-20 dB"
    assert margin_class(45.0) == ">=20 dB"


def test_tolerant_reports_snr_difference_by_margin():
    board = _ref()
    board[1, 6] += 0.3
    margins = np.array([25.0, 2.0, 25.0])
    _, _, counts = compare_tolerant(board, _ref(), margins)
    by = counts["snr_diff_by_margin"]
    assert by["0-3 dB"][0] == 1
    assert by["0-3 dB"][1] == pytest.approx(0.3, abs=1e-9)
    assert by[">=20 dB"] == [2, 0.0]
