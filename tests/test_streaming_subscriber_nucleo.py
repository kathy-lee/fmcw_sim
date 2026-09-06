"""Wire, parser, and diff machinery of the Nucleo streaming subscriber —
pure functions, no serial port, no board. The loopback test is the bridge's own reference
formatted as a board transcript and fed back through the parser: it must
MATCH itself, and a tampered copy must not."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.streaming_subscriber_nucleo import (
    MAGIC, NucleoLink, _process_one, compare_detections, det_rows,
    format_transcript, frame_header_bytes, parse_banner, parse_result,
    row_bytes, row_checksum)


def _ref():
    # [range_bin, doppler_bin, az_deg, x, y, v_radial, snr_db]
    return np.array([
        [40.0, 20.0,  4.0, 2.99270, 0.20927, 1.03, 55.1234567],
        [93.0, 24.0, 25.0, 6.34400, 2.95800, 1.81, 38.7654321],
    ])


def test_header_layout():
    h = frame_header_bytes(7, 12, 32, 128)
    assert len(h) == 14
    assert h[:4] == MAGIC
    assert h[4:8] == (7).to_bytes(4, "little")
    assert h[8:10] == (12).to_bytes(2, "little")
    assert h[10:12] == (32).to_bytes(2, "little")
    assert h[12:14] == (128).to_bytes(2, "little")


def test_row_bytes_and_checksum():
    rows = np.arange(12 * 32 * 128 * 2, dtype=np.int64) % 251
    rows = (rows - 125).astype(np.int16).reshape(12, 32, 128, 2)
    b = row_bytes(rows, 3, 17)
    assert len(b) == 514
    payload, ck = b[:512], int.from_bytes(b[512:], "little")
    assert payload == rows[3, 17].tobytes()
    assert ck == row_checksum(payload) == sum(payload) % 65536


def test_banner_gate():
    lines = ["# onepass-frontend 1", "# clk 180000000", "# fpu 1",
             "# flash_ws 5", "# flash_acr 0x00000705", "# dims 12 32 128",
             "# cfar 2 4 4 8 14.712345678901234 6", "# doa 181 1",
             "# maxdet 64"]
    b = parse_banner(lines)
    assert b["dims"] == (12, 32, 128)
    assert b["alpha"] == pytest.approx(14.712345678901234, rel=1e-15)
    assert b["maxdet"] == 64


def test_loopback_match_and_tamper():
    ref = _ref()
    lines = format_transcript(ref, frame_id=5)
    res = parse_result(lines)
    assert res.err is None and res.frame_id == 5 and res.n == 2
    ok, msgs, flips = compare_detections(det_rows(res), ref)
    assert ok and flips == 0

    bad = ref.copy()
    bad[0, 3] += 1e-3                       # x off by far more than 1e-6
    ok, msgs, _ = compare_detections(det_rows(res), bad)
    assert not ok and msgs


def test_tie_flip_tolerated_once():
    ref = _ref()
    board = ref.copy()
    board[1, 2] += 1.0                      # one grid step
    board[1, 3:5] = 0.0                     # x/y move with the angle: skipped
    ok, msgs, flips = compare_detections(board, ref)
    assert ok and flips == 1
    board[1, 2] += 1.0                      # two grid steps: failure
    ok, _, _ = compare_detections(board, ref)
    assert not ok


def test_count_mismatch_fails():
    ref = _ref()
    ok, msgs, _ = compare_detections(ref[:1], ref)
    assert not ok


class FakeBoard:
    """Serial double: `pending` is the OS buffer's stale backlog (what
    in_waiting counts and reset_input_buffer clears), `blocks` the lines
    the board emits live from now on. What the conversation logic does
    with block boundaries and staleness is the behavior under test."""

    def __init__(self, blocks, pending=()):
        self.out = [ln.encode("ascii") + b"\n" for ln in blocks]
        self.pending = [ln.encode("ascii") + b"\n" for ln in pending]
        self.written = b""
        self.resets = 0
        self.is_open = True

    @property
    def in_waiting(self):
        return sum(len(b) for b in self.pending)

    def readline(self):
        if self.pending:
            return self.pending.pop(0)
        return self.out.pop(0) if self.out else b""

    def write(self, data):
        self.written += bytes(data)

    def reset_input_buffer(self):
        self.resets += 1
        self.pending.clear()


def _fake_link(board):
    link = object.__new__(NucleoLink)
    link.ser = board
    return link


def test_open_discards_stale_backlog(monkeypatch):
    board = FakeBoard([], pending=["# stale banner", "RDY"] * 3)
    mod = types.ModuleType("serial")
    mod.Serial = lambda *a, **k: board
    monkeypatch.setitem(sys.modules, "serial", mod)
    NucleoLink("/dev/fake", 921600)
    assert board.resets == 1 and not board.pending


def test_frame_sent_only_after_live_ready():
    # Stale banners sit in the buffer (the board cycled while the host was
    # busy) and a live banner precedes the result. The frame must go out
    # only against the live RDY: trusting a stale one desynchronizes the
    # block reader, which then misreads a banner as an all-defaults,
    # zero-detection result.
    ref = _ref()
    banner = ["# onepass-frontend 1", "# dims 12 32 128", "RDY"]
    board = FakeBoard(banner + format_transcript(ref, frame_id=0),
                      pending=["# stale banner", "RDY"] * 2)
    rows_q = np.zeros((12, 32, 128, 2), dtype=np.int16)
    fe = SimpleNamespace(q15_process_frame=lambda frame: ref)
    op = SimpleNamespace(quantize=lambda frame: rows_q, last_sat=0)
    line, res = _process_one(_fake_link(board), fe, op, None, 0, None)
    assert "MISMATCH" not in line and "MATCH" in line
    assert res.n == len(ref) and res.frame_id == 0
    assert len(board.written) == 14 + 12 * 32 * (512 + 2)


def test_read_result_skips_interleaved_banner():
    # The board may banner again before its answer lands in our buffer;
    # a block with neither END nor ERR is not the result.
    ref = _ref()
    board = FakeBoard(["# onepass-frontend 1", "# clk 180000000", "RDY"]
                      + format_transcript(ref, frame_id=3))
    res = _fake_link(board).read_result()
    assert res.n == 2 and res.frame_id == 3 and res.err is None


def test_silent_board_raises_instead_of_spinning():
    with pytest.raises(IOError):
        _fake_link(FakeBoard([])).wait_ready()
    with pytest.raises(IOError):
        _fake_link(FakeBoard(["DET 0 0"])).read_result()
