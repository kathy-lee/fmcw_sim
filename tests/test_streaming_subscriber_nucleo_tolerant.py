"""Timing, deadline and link helpers of the tolerant Nucleo subscriber:
pure functions and a serial double, no board."""
from simulator.streaming_subscriber_nucleo_tolerant import (
    CANDIDATE_BAUDS, TimedLink, deadline_table, parse_time, phase_means,
    row_deadline_cyc)

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
