#!/usr/bin/env python3
"""Streaming subscriber whose front end runs ON the Nucleo, over UART.

The lineage seat: where streaming_subscriber_mimo_hybrid.py hands each
frame to the C front end in-process through ctypes, this bridge quantizes
the frame to the exact bytes a target ADC would produce, ships them to the
board row by row, and reads the board's detections back as text — then
diffs them live against the frozen chain running on the laptop
(CFrontend.q15_process_frame) on the very same complex64 input. MATCH per
frame is the acceptance signal; a MISMATCH never stops the stream, it is
counted and shown.

Pure helpers first (wire framing, banner/result parsing, the detection
diff) — importable and tested without pyserial, zmq, or the built library
— then the serial conversation, then main().
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from simulator.streaming_subscriber import detect_gap
from simulator.wire import deserialize_frame

MAGIC = b"OPF1"
GRID_MIN_DEG = -90


def frame_header_bytes(frame_id: int, v: int, c: int, s: int) -> bytes:
    return MAGIC + struct.pack("<IHHH", frame_id, v, c, s)


def row_checksum(payload: bytes) -> int:
    return sum(payload) & 0xFFFF


def row_bytes(rows_q: np.ndarray, v: int, c: int) -> bytes:
    payload = np.ascontiguousarray(rows_q[v, c]).tobytes()
    return payload + struct.pack("<H", row_checksum(payload))


@dataclass
class BoardResult:
    frame_id: int = -1
    n: int = 0
    overflow: int = 0
    dets: list = field(default_factory=list)   # 9-tuples, D-line order
    row_cyc: int = 0
    finish_cyc: int = 0
    heap: int = 0
    stack_peak: int = 0
    free_start: int = 0
    sat: int = 0
    err: str | None = None


def parse_banner(lines: list[str]) -> dict:
    out: dict = {}
    for ln in lines:
        t = ln.split()
        if ln.startswith("# dims"):
            out["dims"] = (int(t[2]), int(t[3]), int(t[4]))
        elif ln.startswith("# cfar"):
            out["cfar"] = [int(t[2]), int(t[3]), int(t[4]), int(t[5])]
            out["alpha"] = float(t[6])
            out["min_range_bins"] = int(t[7])
        elif ln.startswith("# maxdet"):
            out["maxdet"] = int(t[2])
        elif ln.startswith("# clk"):
            out["clk"] = int(t[2])
        elif ln.startswith("# fpu"):
            out["fpu"] = int(t[2])
    return out


def parse_result(lines: list[str]) -> BoardResult:
    r = BoardResult()
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "ERR":
            r.err = ln
        elif t[0] == "DET":
            r.n, r.overflow = int(t[1]), int(t[2])
        elif t[0] == "D":
            # power arrives as an integer-valued %.17g (newlib-nano printf
            # has no %lld); exact below 2^53, so int(float()) is lossless
            r.dets.append((int(t[1]), int(t[2]), int(t[3]),
                           float(t[4]), float(t[5]), float(t[6]),
                           float(t[7]), int(float(t[8])), float(t[9])))
        elif t[0] == "TIME":
            r.row_cyc, r.finish_cyc = int(t[2]), int(t[4])
        elif t[0] == "MEM":
            r.heap, r.stack_peak, r.free_start = \
                int(t[2]), int(t[4]), int(t[6])
        elif t[0] == "SAT":
            r.sat = int(t[1])
        elif t[0] == "END":
            r.frame_id = int(t[1])
    return r


def det_rows(result: BoardResult) -> np.ndarray:
    """Board detections in CFrontend's 7-column order (az_idx -> degrees)."""
    if not result.dets:
        return np.empty((0, 7))
    return np.array([[d[0], d[1], d[2] - 90, d[3], d[4], d[5], d[6]]
                     for d in result.dets], dtype=np.float64)


def format_transcript(det: np.ndarray, frame_id: int) -> list[str]:
    """Reference detections rendered as a board result block — the
    loopback fixture. power/thresh carry placeholders; they are
    diagnostics, not compared."""
    lines = [f"DET {len(det)} 0"]
    for r in det:
        lines.append(
            f"D {int(r[0])} {int(r[1])} {int(r[2]) + 90} "
            f"{r[3]:.9e} {r[4]:.9e} {r[5]:.9e} {r[6]:.9e} 1 1.0")
    lines += ["TIME row_cyc 0 finish_cyc 0",
              "MEM heap 0 stack_peak 0 free_start 0",
              "SAT 0", f"END {frame_id}", "RDY"]
    return lines


def _sorted(det: np.ndarray) -> np.ndarray:
    if det.shape[0] == 0:
        return det
    order = np.lexsort((det[:, 2], det[:, 0], det[:, 1]))
    return det[order]


def compare_detections(board: np.ndarray, ref: np.ndarray,
                       tol: float = 1e-6):
    """MATCH verdict: counts equal; per row az grid-exact (a single grid
    step is tolerated but counted as a tie flip, and x/y are then skipped
    since they move with the angle); x, y, v_radial, snr_db within tol
    relative. Rows are compared after sorting by (doppler, range, az)."""
    msgs: list[str] = []
    flips = 0
    if board.shape[0] != ref.shape[0]:
        return False, [f"count {board.shape[0]} != {ref.shape[0]}"], 0
    b, r = _sorted(board), _sorted(ref)

    def close(a, c):
        return abs(a - c) <= tol * max(abs(a), abs(c), 1e-12)

    for i in range(b.shape[0]):
        if b[i, 0] != r[i, 0] or b[i, 1] != r[i, 1]:
            msgs.append(f"row {i}: cell ({b[i,1]:.0f},{b[i,0]:.0f}) != "
                        f"({r[i,1]:.0f},{r[i,0]:.0f})")
            continue
        d_az = abs(b[i, 2] - r[i, 2])
        cols = [(5, "v_radial"), (6, "snr_db")]
        if d_az == 0:
            cols += [(3, "x"), (4, "y")]
        elif d_az == 1:
            flips += 1
            msgs.append(f"row {i}: tie flip az {b[i,2]:.0f} vs {r[i,2]:.0f}")
        else:
            msgs.append(f"row {i}: az {b[i,2]:.0f} != {r[i,2]:.0f}")
            continue
        for j, name in cols:
            if not close(b[i, j], r[i, j]):
                msgs.append(f"row {i}: {name} {b[i,j]!r} != {r[i,j]!r}")
    hard = [m for m in msgs if "tie flip" not in m]
    return not hard, msgs, flips


class NucleoLink:
    """Blocking serial conversation with the board. The board paces: we
    only ever transmit immediately after reading an RDY."""

    def __init__(self, port: str, baud: int, timeout: float = 5.0):
        import serial                     # local: hardware-only dependency
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # an idle board banners into the void; whatever accumulated before
        # open is history, and any RDY in it is expired permission
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def wait_ready(self) -> list[str]:
        """Discard the backlog, then read lines until an RDY that arrived
        after the discard; return everything seen before it (the banner,
        usually). Freshness is the point: the board banners on a cycle
        whether or not anyone reads, so a buffered RDY is expired
        permission — only one that arrives live proves the board is
        entering its listen window now."""
        while self.ser.in_waiting:
            self.ser.readline()
        seen: list[str] = []
        empties = 0
        while True:
            ln = self.ser.readline().decode("ascii", "replace").strip()
            if not ln:
                empties += 1
                if empties >= 6 or not self.ser.is_open:
                    raise IOError("no RDY from the board")
                continue
            empties = 0
            if ln == "RDY":
                return seen
            seen.append(ln)

    def send_frame(self, frame_id: int, rows_q: np.ndarray,
                   row_pace: float = 0.0) -> None:
        """row_pace > 0 sleeps that long after each row — long enough to
        cover the row's own wire time plus a quiet gap, so the board's
        between-row compute happens on a silent wire. A diagnostic: if
        pacing cures checksum errors, the board is deaf between rows."""
        V, C = rows_q.shape[0], rows_q.shape[1]
        self.ser.write(frame_header_bytes(frame_id, V, C,
                                          rows_q.shape[2]))
        for v in range(V):
            for c in range(C):
                self.ser.write(row_bytes(rows_q, v, c))
                if row_pace:
                    time.sleep(row_pace)

    def read_result(self) -> BoardResult:
        """Read one result block: through END and its trailing RDY, or stop
        right after an ERR line (main then calls wait_ready() to resync).
        A block with neither — a banner the board printed while our
        transmission was still in flight — is discarded, not misread as an
        empty result."""
        lines: list[str] = []
        done = False
        empties = 0
        while True:
            ln = self.ser.readline().decode("ascii", "replace").strip()
            if not ln:
                empties += 1
                if empties >= 6 or not self.ser.is_open:
                    raise IOError("no result from the board")
                continue
            empties = 0
            if ln == "RDY":
                if done:
                    break
                lines.clear()
                continue
            lines.append(ln)
            if ln.startswith("END"):
                done = True
            if ln.startswith("ERR"):
                break
        return parse_result(lines)


def _process_one(link, fe, op, frame, frame_id, t_pub,
                 row_pace: float = 0.0) -> tuple[str, BoardResult]:
    """Quantize, await RDY, send, read, diff one frame. Returns
    (verdict_line, result)."""
    rows_q = op.quantize(frame)
    if op.last_sat:
        print(f"warning: {op.last_sat} samples clipped at the ADC grid")
    # a live RDY: the board prints one at the top of every cycle, banner
    # first, and only then listens — sending against an earlier RDY races
    # the banner transmit and desynchronizes the block reader
    link.wait_ready()
    t0 = time.time()
    link.send_frame(frame_id, rows_q, row_pace)
    res = link.read_result()
    if res.err:
        link.wait_ready()
        return f"[frame {frame_id}] BOARD {res.err}", res
    ref = fe.q15_process_frame(frame)
    ok, msgs, flips = compare_detections(det_rows(res), ref)
    lat = time.time() - (t_pub if t_pub else t0)
    verdict = "MATCH" if ok else "MISMATCH"
    line = (f"[frame {frame_id}] {verdict} ({res.n} dets, tie flips {flips}) "
            f"latency {lat:.2f} s")
    if not ok or flips:
        line += "\n  " + "\n  ".join(msgs)
    if not ok:
        line += (f"\n  board: {np.array2string(det_rows(res), precision=6)}"
                 f"\n  ref:   {np.array2string(ref, precision=6)}")
    return line, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://127.0.0.1:5555")
    ap.add_argument("--topic", default="fmcw")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--params", required=True)
    ap.add_argument("--scenario",
                    default="scenarios/scenario_mimo_stream_128.json")
    ap.add_argument("--one-shot", action="store_true")
    ap.add_argument("--row-pace", type=float, default=0.0,
                    help="seconds to sleep after each row (diagnostic)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = forever")
    ap.add_argument("--summary-every", type=int, default=10)
    args = ap.parse_args()

    from processing.c_frontend import CFrontend
    from processing.c_frontend_onepass import CFrontendOnepass, read_params

    cfg = read_params(args.params)
    link = NucleoLink(args.port, args.baud)
    b = parse_banner(link.wait_ready())
    want = (int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"]))
    alpha_ok = abs(b.get("alpha", 0) - cfg["cfar_alpha"]) \
        <= 1e-9 * abs(cfg["cfar_alpha"])
    if b.get("dims") != want or not alpha_ok:
        print(f"refusing: board {b} vs params dims {want} "
              f"alpha {cfg['cfar_alpha']}")
        raise SystemExit(1)
    print(f"banner: dims {b['dims']} alpha {b['alpha']}  [accepted]")

    fe = CFrontend(args.params)
    op = CFrontendOnepass(args.params)

    if args.one_shot:
        from simulator.beat_signal_generator import (FMCWBeatSignalGenerator,
                                                     load_scenario)
        rcfg, targets = load_scenario(args.scenario)
        gen = FMCWBeatSignalGenerator(rcfg)
        frame = gen.generate_frame_mimo(targets, frame_index=0)
        frame = np.ascontiguousarray(frame.astype(np.complex64))
        line, res = _process_one(link, fe, op, frame, 0, None, args.row_pace)
        print(line)
        print(f"TIME row_cyc {res.row_cyc} finish_cyc {res.finish_cyc}   "
              f"MEM heap {res.heap} stack_peak {res.stack_peak} "
              f"free_start {res.free_start}  SAT {res.sat}")
        link.close()
        raise SystemExit(0 if "MATCH" in line and "MIS" not in line else 1)

    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    print(f"receiving from {args.connect}, feeding {args.port} "
          f"at {args.baud} baud")

    processed = mismatches = flips_total = drops = 0
    heap_max = stack_max = 0            # session watermarks, board-reported
    last_fid = None
    lat_sum = row_sum = fin_sum = 0.0
    clk = b.get("clk", 180000000)
    try:
        while args.max_frames == 0 or processed < args.max_frames:
            if not sock.poll(1000, zmq.POLLIN):
                continue
            parts = sock.recv_multipart()
            while sock.poll(0, zmq.POLLIN):        # drain: keep only newest
                parts = sock.recv_multipart()
            header = json.loads(parts[1].decode("utf-8"))
            if len(header["shape"]) != 3:
                continue
            fid = int(header["frame_id"])
            drops += detect_gap(fid, last_fid)
            last_fid = fid
            frame = deserialize_frame(header, parts[2])
            frame = np.ascontiguousarray(frame.astype(np.complex64))
            line, res = _process_one(link, fe, op, frame, fid,
                                     float(header["timestamp_unix_s"]),
                                     args.row_pace)
            print(line, flush=True)
            if res.err:
                continue
            processed += 1
            if "MISMATCH" in line:
                mismatches += 1
            flips_total += line.count("tie flip az")
            lat_sum += time.time() - float(header["timestamp_unix_s"])
            row_sum += res.row_cyc
            fin_sum += res.finish_cyc
            heap_max = max(heap_max, res.heap)
            stack_max = max(stack_max, res.stack_peak)
            if processed % args.summary_every == 0:
                n = args.summary_every
                print(f"-- {processed} processed, {drops} dropped, "
                      f"{mismatches} mismatches, {flips_total} tie flips | "
                      f"latency {lat_sum / n:.2f} s | "
                      f"row {row_sum / n / 1e6:.1f} Mcyc "
                      f"({row_sum / n / clk * 1e3:.0f} ms) "
                      f"finish {fin_sum / n / 1e6:.1f} Mcyc "
                      f"({fin_sum / n / clk * 1e3:.0f} ms) | "
                      f"heap<={heap_max} stack_peak<={stack_max}", flush=True)
                lat_sum = row_sum = fin_sum = 0.0
    except KeyboardInterrupt:
        print("\nbridge stopped.")
    finally:
        link.close()
        sock.close(linger=0)


if __name__ == "__main__":
    main()
