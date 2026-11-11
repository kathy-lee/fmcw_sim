#!/usr/bin/env python3
"""Streaming subscriber for the speed-up build of the Nucleo front end
(embedded/nucleo_f446re_frontend_speedup).

Differences from streaming_subscriber_nucleo.py:

1. Full timing. The board's TIME line is read in full, so the summaries
   show each finish stage (power, cells, detection, DoA, printing) and the
   slowest single row, plane_max_cyc.
2. Row deadline. With ping-pong reception a row must be processed before
   the next one has arrived. The summaries warn when plane_max_cyc exceeds
   the time one row takes on the wire at the baud in use, and the final
   summary lists that deadline for every candidate baud rate.
3. PC-side phase times. Each frame's wait for the board's RDY, its
   transmission, the wait for the result and the reference computation
   are timed and averaged.

4. DoA arithmetic. The banner says whether the board runs the double- or
   the single-precision DoA scan, and the accepted-banner line shows it.

Imported from streaming_subscriber_nucleo.py unchanged: the serial
conversation (NucleoLink), frame and row packing, banner and result
parsing, the dimension and CFAR check at start-up, and the exact
comparison (compare_detections). The command-line arguments are the same.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from simulator.streaming_subscriber import detect_gap
from simulator.streaming_subscriber_nucleo import (
    BoardResult, NucleoLink, compare_detections, det_rows, parse_banner,
    parse_result)
from simulator.wire import deserialize_frame

CLK_HZ = 180_000_000
ROW_WIRE_BYTES = 514            # 512 B of samples plus the 2-byte checksum
BITS_PER_BYTE = 10              # start bit, 8 data bits, stop bit
CANDIDATE_BAUDS = (921600, 1500000, 1875000, 2000000)
STAGES = ("power_cyc", "cells_cyc", "det_cyc", "doa_cyc", "emit_cyc")
PHASES = ("ready_s", "send_s", "read_s", "ref_s")


def parse_banner_ext(lines: list[str]) -> dict:
    """The certified banner fields plus the ones only this build prints."""
    out = parse_banner(lines)
    for ln in lines:
        t = ln.split()
        if ln.startswith("# doa_arith") and len(t) >= 3:
            out["doa_arith"] = t[2]
    return out


def parse_time(lines: list[str]) -> dict[str, int]:
    """Every name/value pair of the block's TIME line; empty without one."""
    for ln in lines:
        t = ln.split()
        if t and t[0] == "TIME":
            return {t[i]: int(t[i + 1]) for i in range(1, len(t) - 1, 2)}
    return {}


def row_deadline_cyc(baud: int, clk: int = CLK_HZ,
                     row_bytes: int = ROW_WIRE_BYTES) -> int:
    """CPU cycles one row takes on the wire: the time the board has to
    process a row before the next one has fully arrived."""
    return row_bytes * BITS_PER_BYTE * clk // baud


def deadline_table(plane_max_cyc: int,
                   bauds: tuple[int, ...] = CANDIDATE_BAUDS) -> list[str]:
    """One line per baud rate: whether the slowest row fits its deadline."""
    lines = []
    for b in bauds:
        d = row_deadline_cyc(b)
        verdict = "MISS" if plane_max_cyc > d else "ok"
        lines.append(f"baud {b} deadline {d} plane_max_cyc {plane_max_cyc} "
                     f"{verdict}")
    return lines


def phase_means(frames: list[dict[str, float]]) -> dict[str, float]:
    """The mean of each key over the frames."""
    if not frames:
        return {}
    return {k: sum(f[k] for f in frames) / len(frames) for k in frames[0]}


class TimedLink(NucleoLink):
    """NucleoLink whose result also carries the full TIME line."""

    def read_result(self) -> tuple[BoardResult, dict[str, int]]:
        """Read one result block as NucleoLink.read_result does: through
        END and its trailing RDY, or up to an ERR line. A block with
        neither, a banner printed while the frame was still in flight, is
        discarded."""
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
        return parse_result(lines), parse_time(lines)


def _process_one(link, fe, op, frame, frame_id, t_pub, row_pace: float = 0.0):
    """Quantize, await RDY, send, read, diff one frame. Returns
    (verdict_line, result, time_fields, phase_seconds).

    send_s ends when the last row is handed to the serial driver, which
    may still hold bytes; read_s runs from there to the end of the result
    block. Together they cover the rows' wire time and the board's finish.
    """
    rows_q = op.quantize(frame)
    if op.last_sat:
        print(f"warning: {op.last_sat} samples clipped at the ADC grid")
    t0 = time.time()
    link.wait_ready()
    t1 = time.time()
    link.send_frame(frame_id, rows_q, row_pace)
    t2 = time.time()
    res, times = link.read_result()
    t3 = time.time()
    phases = {"ready_s": t1 - t0, "send_s": t2 - t1, "read_s": t3 - t2,
              "ref_s": 0.0}
    if res.err:
        link.wait_ready()
        return f"[frame {frame_id}] BOARD {res.err}", res, times, phases
    ref = fe.q15_process_frame(frame)
    ok, msgs, flips = compare_detections(det_rows(res), ref)
    phases["ref_s"] = time.time() - t3
    lat = time.time() - (t_pub if t_pub else t0)
    verdict = "MATCH" if ok else "MISMATCH"
    line = (f"[frame {frame_id}] {verdict} ({res.n} dets, tie flips {flips}) "
            f"latency {lat:.2f} s")
    if not ok or flips:
        line += "\n  " + "\n  ".join(msgs)
    if not ok:
        line += (f"\n  board: {np.array2string(det_rows(res), precision=6)}"
                 f"\n  ref:   {np.array2string(ref, precision=6)}")
    return line, res, times, phases


def _timing_lines(time_rows: list[dict[str, int]],
                  phase_rows: list[dict[str, float]], baud: int) -> list[str]:
    """The stage, slowest-row and PC-phase lines of one summary."""
    lines = []
    if time_rows and all(s in time_rows[0] for s in STAGES):
        n = len(time_rows)
        means = "  ".join(
            f"{s[:-4]} {sum(t[s] for t in time_rows) / n / 1e6:.2f}"
            for s in STAGES)
        lines.append(f"   stages, Mcyc per frame: {means}")
    plane_max = max((t.get("plane_max_cyc", 0) for t in time_rows), default=0)
    if plane_max:
        deadline = row_deadline_cyc(baud)
        lines.append(f"   slowest row: plane_max_cyc {plane_max} "
                     f"(deadline {deadline} at {baud} baud)")
        if plane_max > deadline:
            lines.append(f"   WARNING plane_max_cyc {plane_max} exceeds row "
                         f"deadline {deadline} at {baud} baud")
    pm = phase_means(phase_rows)
    if pm:
        lines.append("   pc phases, s per frame: " + "  ".join(
            f"{k[:-2]} {pm[k]:.2f}" for k in PHASES))
    return lines


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
    link = TimedLink(args.port, args.baud)
    b = parse_banner_ext(link.wait_ready())
    want = (int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"]))
    alpha_ok = abs(b.get("alpha", 0) - cfg["cfar_alpha"]) \
        <= 1e-9 * abs(cfg["cfar_alpha"])
    if b.get("dims") != want or not alpha_ok:
        print(f"refusing: board {b} vs params dims {want} "
              f"alpha {cfg['cfar_alpha']}")
        raise SystemExit(1)
    print(f"banner: dims {b['dims']} alpha {b['alpha']} "
          f"doa {b.get('doa_arith', 'not reported')}  [accepted]")

    fe = CFrontend(args.params)
    op = CFrontendOnepass(args.params)

    if args.one_shot:
        from simulator.beat_signal_generator import (FMCWBeatSignalGenerator,
                                                     load_scenario)
        rcfg, targets = load_scenario(args.scenario)
        gen = FMCWBeatSignalGenerator(rcfg)
        frame = gen.generate_frame_mimo(targets, frame_index=0)
        frame = np.ascontiguousarray(frame.astype(np.complex64))
        line, res, times, phases = _process_one(link, fe, op, frame, 0, None,
                                                args.row_pace)
        print(line)
        print(f"TIME row_cyc {res.row_cyc} finish_cyc {res.finish_cyc}   "
              f"MEM heap {res.heap} stack_peak {res.stack_peak} "
              f"free_start {res.free_start}  SAT {res.sat}")
        print("\n".join(_timing_lines([times] if times else [], [phases],
                                      args.baud)))
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
    plane_max_session = 0
    last_fid = None
    lat_sum = row_sum = fin_sum = 0.0
    time_rows: list[dict[str, int]] = []       # since the last summary
    phase_rows: list[dict[str, float]] = []
    clk = b.get("clk", CLK_HZ)
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
            line, res, times, phases = _process_one(
                link, fe, op, frame, fid, float(header["timestamp_unix_s"]),
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
            time_rows.append(times)
            phase_rows.append(phases)
            plane_max_session = max(plane_max_session,
                                    times.get("plane_max_cyc", 0))
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
                for ln in _timing_lines(time_rows, phase_rows, args.baud):
                    print(ln, flush=True)
                lat_sum = row_sum = fin_sum = 0.0
                time_rows, phase_rows = [], []
    except KeyboardInterrupt:
        print("\nbridge stopped.")
    finally:
        link.close()
    if plane_max_session:
        print("row deadline by baud rate, for the session's slowest row:")
        for ln in deadline_table(plane_max_session):
            print("   " + ln)


if __name__ == "__main__":
    main()
