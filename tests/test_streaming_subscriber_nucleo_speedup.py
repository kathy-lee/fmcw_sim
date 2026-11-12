"""Timing, deadline and link helpers of the speed-up Nucleo subscriber:
pure functions and a serial double, no board."""
import numpy as np
import pytest

from simulator.streaming_subscriber_nucleo import BoardResult
from simulator.streaming_subscriber_nucleo_speedup import (
    TargetTally, TimedLink, _process_one, board_margin_db, check_banner_baud,
    parse_banner_ext, parse_time, phase_means, reference_lib_path,
    resolve_reference, row_deadline_cyc, score_targets, session_lines,
    slowest_row_text, summary_lines, truth_targets)

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


def test_slowest_row_text_shows_the_margin():
    assert slowest_row_text(481444, 1875000) == \
        "481444 of 493440 cycles at 1875000 baud (margin 2.4 %)"


def test_slowest_row_at_the_limit_is_no_warning():
    assert slowest_row_text(1003906, 921600) == \
        "1003906 of 1003906 cycles at 921600 baud (margin 0.0 %)"


def test_slowest_row_over_the_deadline_warns():
    assert slowest_row_text(500000, 1875000) == \
        "WARNING 500000 of 493440 cycles at 1875000 baud (over by 1.3 %)"


SUMMARY_COUNTS = {"processed": 10, "dropped": 307, "board_errors": 0,
                  "mismatches": 0, "flips": 0, "latency_s": 1.78,
                  "row_cyc": 13.7e6, "finish_cyc": 17.9e6,
                  "heap": 336, "stack": 912}
SUMMARY_PHASES = [{"ready_s": 0.09, "send_s": 1.55, "read_s": 0.11,
                   "ref_s": 0.01}]


def test_summary_lines_layout():
    lines = summary_lines(SUMMARY_COUNTS, [parse_time([TIME_LINE])],
                          SUMMARY_PHASES, baud=1875000)
    assert lines[0].startswith("== 10 frames processed =")
    assert lines[1:] == [
        "  frames        dropped 307 | board errors 0 | mismatches 0 | "
        "tie flips 0",
        "  latency       1.78 s per frame",
        "  laptop        ready 0.09 s | send 1.55 s | read 0.11 s | "
        "reference 0.01 s",
        "  board         rows 13.7 Mcyc (76 ms) | finish 17.9 Mcyc (99 ms)",
        "  finish        power 3.00 | cells 2.00 | det 5.00 | doa 890.00 | "
        "emit 40.00  (Mcyc)",
        "  slowest row   410000 of 493440 cycles at 1875000 baud "
        "(margin 16.9 %)",
        "  memory        heap 336 B | stack 912 B  (largest so far)",
    ]


def test_summary_without_stage_times_leaves_those_lines_out():
    # a board whose TIME line has only the two totals
    lines = summary_lines(SUMMARY_COUNTS, [{}], SUMMARY_PHASES, baud=921600)
    labels = [ln.split()[0] for ln in lines[1:]]
    assert labels == ["frames", "latency", "laptop", "board", "memory"]


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


def test_banner_baud_parsed():
    assert parse_banner_ext(["# baud 1875000"])["baud"] == 1875000
    assert "baud" not in parse_banner_ext(["# dims 12 32 128"])


def test_banner_baud_mismatch_refused():
    # 2 Mbaud cannot be exact on the board: it runs 2.3 % fast
    assert check_banner_baud({"baud": 2045454}, 2000000) is None
    assert "baud" in check_banner_baud({"baud": 921600}, 2000000)
    assert "baud" in check_banner_baud({}, 2000000)


def row(rng, dop, az, snr):
    """One detection row: [range, doppler, az_deg, x, y, v_radial, snr_db]."""
    return [rng, dop, az, 1.5, -0.5, 2.0, snr]


def test_board_margin_db():
    res = BoardResult(dets=[(10, 3, 70, 1.5, -0.5, 2.0, 30.0, 200, 100.0)])
    assert board_margin_db(res) == pytest.approx([3.0103], abs=1e-4)
    assert board_margin_db(BoardResult()).shape == (0,)


def test_resolve_reference():
    assert resolve_reference("auto", {"fft_impl": "cmsis"}) == "cmsis"
    assert resolve_reference("auto", {"fft_impl": "own"}) == "own"
    assert resolve_reference("auto", {}) == "own"
    assert resolve_reference("own", {"fft_impl": "cmsis"}) == "own"
    assert resolve_reference("cmsis", {"fft_impl": "own"}) == "cmsis"


def test_reference_lib_path():
    assert reference_lib_path("own") is None       # CFrontend's default
    path = reference_lib_path("cmsis")
    assert path.name == "libfmcw_cmsis.so"
    assert path.parent.name == "c_frontend"


def test_read_banner_returns_lines_up_to_rdy():
    board = FakeBoard(["# dims 12 32 128", "# fft_impl own", "RDY", "DET 0 0"])
    lines = _fake_link(board).read_banner(settle_s=0.0)
    assert lines == ["# dims 12 32 128", "# fft_impl own"]


# ---- judging detections against the frame's true targets ----

V_AXIS = [-13.0 + 0.8125 * i for i in range(32)]      # bin 16 is 0 m/s


def _truth(range_m, velocity_mps, azimuth_deg):
    return {"range_m": range_m, "velocity_mps": velocity_mps,
            "azimuth_deg": azimuth_deg}


def _targets(*truth):
    return truth_targets(list(truth), range_per_bin=0.15, velocity_axis=V_AXIS,
                         n_samples=128, min_range_bin=6)


def test_truth_bins_from_range_and_velocity():
    (t,) = _targets(_truth(3.0, 1.625, 5.0))
    assert t["range_bin"] == pytest.approx(20.0)
    assert t["doppler_bin"] == pytest.approx(18.0)
    assert t["az_deg"] == 5.0 and t["in_map"]


def test_truth_range_wraps_like_the_signal():
    # 20 m is bin 133.3 of a 128-bin range FFT: it comes back in at bin 5.3,
    # below the first kept bin
    near, far = _targets(_truth(20.0, 0.0, 0.0), _truth(12.0, 0.0, 0.0))
    assert near["range_bin"] == pytest.approx(133.0 + 1.0 / 3.0 - 128.0)
    assert not near["in_map"]
    assert far["range_bin"] == pytest.approx(80.0) and not far["in_map"]


def test_target_found_with_its_cluster():
    targets = _targets(_truth(3.0, 1.625, 5.0))
    det = np.array([row(20, 18, 5, 40.0), row(21, 18, 6, 30.0)])
    s = score_targets(det, targets, n_chirps=32, cluster_bins=(2, 2),
                      margin_db=np.array([25.0, 4.0]))
    assert s["found"] == [True]
    assert s["az_err_deg"] == [0.0]
    assert s["peak_margin_db"] == [25.0]
    assert (s["cluster"], s["away"]) == (2, 0)


def test_target_missed():
    targets = _targets(_truth(3.0, 1.625, 5.0))
    s = score_targets(np.empty((0, 7)), targets, n_chirps=32, cluster_bins=(2, 2))
    assert s["found"] == [False]
    assert s["az_err_deg"] == [None] and s["peak_margin_db"] == [None]


def test_target_outside_the_map_is_not_judged():
    targets = _targets(_truth(12.0, 0.0, 0.0))
    s = score_targets(np.empty((0, 7)), targets, n_chirps=32, cluster_bins=(2, 2))
    assert s["found"] == [None]


def test_detection_away_from_any_target_is_counted():
    targets = _targets(_truth(3.0, 1.625, 5.0))
    det = np.array([row(20, 18, 5, 40.0), row(50, 16, 0, 20.0)])
    s = score_targets(det, targets, n_chirps=32, cluster_bins=(2, 2))
    assert s["found"] == [True]
    assert (s["cluster"], s["away"]) == (1, 1)


def test_angle_error_is_of_the_strongest_matching_detection():
    targets = _targets(_truth(3.0, 1.625, 5.0))
    det = np.array([row(20, 18, 9, 40.0), row(20, 19, 5, 30.0)])
    s = score_targets(det, targets, n_chirps=32, cluster_bins=(2, 2))
    assert s["found"] == [True] and s["az_err_deg"] == [4.0]


def test_cluster_window_is_per_axis():
    # the target is at range bin 20, Doppler bin 18
    targets = _targets(_truth(3.0, 1.625, 5.0))
    in_range = np.array([row(23, 18, 5, 20.0)])      # 3 range bins off
    in_doppler = np.array([row(20, 21, 5, 20.0)])    # 3 Doppler bins off

    def counts(det, window):
        s = score_targets(det, targets, n_chirps=32, cluster_bins=window)
        return s["cluster"], s["away"]

    assert counts(in_range, (4, 2)) == (1, 0)
    assert counts(in_range, (2, 2)) == (0, 1)
    assert counts(in_doppler, (4, 2)) == (0, 1)
    assert counts(in_doppler, (2, 2)) == (0, 1)
    assert counts(in_doppler, (2, 4)) == (1, 0)


def _tally():
    targets = _targets(_truth(3.0, 1.625, 5.0))
    board = score_targets(
        np.array([row(20, 18, 5, 40.0), row(50, 16, 0, 20.0)]), targets,
        n_chirps=32, margin_db=np.array([25.0, 4.0]), cluster_bins=(4, 2))
    ref = score_targets(np.array([row(20, 18, 6, 40.0)]), targets,
                        n_chirps=32, cluster_bins=(4, 2))
    tally = TargetTally("own-FFT", (4, 2))
    tally.add(board, ref)
    return tally


def test_tally_states_its_cluster_window():
    assert "within 4 range and 2 Doppler bins" in "\n".join(_tally().lines())


def test_tally_rows():
    lines = _tally().lines()
    assert ("    target 0 |      1 |     1 |         1 |   0.0 deg |"
            "   1.0 deg | 25.0 to 25.0 dB") in lines
    assert "    near a target          |     1 |         1" in lines
    assert "    away from every target |     1 |         0" in lines
    assert any("own-FFT" in ln and "board" in ln for ln in lines)


def test_tally_columns_line_up():
    """Within each table, every separator stands in the same column; a
    heading that spans two columns may leave one out."""
    lines = _tally().lines()
    tables, cur = [], []
    for ln in lines:
        if "|" in ln or "+" in ln:      # a row, or the rule under a heading
            cur.append([i for i, ch in enumerate(ln) if ch in "|+"])
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    assert len(tables) == 2
    for rows in tables:
        full = max(rows, key=len)
        assert all(set(r) <= set(full) for r in rows)
        assert sum(r == full for r in rows) >= 3    # heading, rule, data


def test_tally_without_targets_prints_nothing():
    assert TargetTally("own-FFT", (4, 2)).lines() == []


def test_session_lines():
    lines = session_lines("cmsis", processed=10, mismatches=0, flips=0,
                          board_errors=0, plane_max_cyc=481444,
                          baud=1875000, tally=_tally())
    assert lines[0].startswith("== session result =")
    assert lines[1:4] == [
        "  comparison    cmsis-FFT reference: 10 MATCH, 0 MISMATCH, "
        "0 tie flips",
        "  board errors  0 frames (not processed, not compared)",
        "  slowest row   481444 of 493440 cycles at 1875000 baud "
        "(margin 2.4 %)",
    ]
    assert lines[4] == ""
    assert lines[5].startswith("  true targets (found: a detection within")


def test_session_lines_without_a_slowest_row():
    lines = session_lines("own", processed=0, mismatches=0, flips=0,
                          board_errors=2, plane_max_cyc=0, baud=921600,
                          tally=TargetTally())
    assert [ln.split()[0] for ln in lines[1:]] == ["comparison", "board"]


def test_doppler_distance_wraps():
    # 12.6 m/s is bin 31.5; a detection in Doppler bin 0 is half a bin away
    targets = _targets(_truth(3.0, -13.0 + 0.8125 * 31.5, 0.0))
    det = np.array([row(20, 0, 0, 40.0)])
    assert score_targets(det, targets, n_chirps=32, cluster_bins=(2, 2))["found"] == [True]


class _OneResultLink:
    """Link double: hands back one prepared board result."""

    def __init__(self, res):
        self.res = res

    def wait_ready(self):
        pass

    def send_frame(self, frame_id, rows_q, row_pace):
        pass

    def read_result(self):
        return self.res, {}


class _NoQuantizer:
    last_sat = 0

    def quantize(self, frame):
        return None


class _FixedChain:
    """Reference double: the same detections for every frame."""

    def __init__(self, det):
        self.det = det

    def q15_process_frame(self, frame):
        return self.det


def _one_frame(fe_truth=None, ref_snr=40.0):
    targets = _targets(_truth(3.0, 1.625, 5.0))
    board = BoardResult(
        n=1, dets=[(20, 18, 95, 1.5, -0.5, 2.0, 40.0, 200, 100.0)])
    same = np.array([row(20, 18, 5, ref_snr)])
    frame = np.zeros((12, 32, 128), dtype=np.complex64)
    return _process_one(_OneResultLink(board), _FixedChain(same),
                        _NoQuantizer(), frame, 0, None, targets=targets,
                        fe_truth=fe_truth, cluster_bins=(4, 2))[4]


def test_targets_scored_against_the_comparison_reference_by_default():
    out = _one_frame()
    assert out["verdict"] == "MATCH"
    assert out["score_board"]["found"] == [True]
    assert out["score_ref"]["found"] == [True]


def test_targets_scored_against_a_separate_reference_when_given():
    # the verdict still comes from the comparison reference
    out = _one_frame(fe_truth=_FixedChain(np.empty((0, 7))))
    assert out["verdict"] == "MATCH"
    assert out["score_board"]["found"] == [True]
    assert out["score_ref"]["found"] == [False]


def test_any_difference_from_the_reference_is_a_mismatch():
    out = _one_frame(ref_snr=40.5)
    assert out["verdict"] == "MISMATCH"
    assert set(out) == {"verdict", "flips", "score_board", "score_ref"}
