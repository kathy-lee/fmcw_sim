"""Wire, parser, and diff machinery of the Nucleo streaming subscriber —
pure functions, no serial port, no board. The loopback test is the bridge's own reference
formatted as a board transcript and fed back through the parser: it must
MATCH itself, and a tampered copy must not."""
import numpy as np
import pytest

from simulator.streaming_subscriber_nucleo import (
    MAGIC, compare_detections, det_rows, format_transcript,
    frame_header_bytes, parse_banner, parse_result, row_bytes, row_checksum)


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
