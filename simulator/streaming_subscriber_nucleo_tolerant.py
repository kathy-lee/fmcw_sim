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

   The banner also gives the UART rate the board actually runs. The
   subscriber refuses to start when it differs from --baud by more than
   3 %.

5. Reference with the board's FFT. The banner says which FFT the board
   runs. For the CMSIS-DSP FFT the laptop reference is the same batch
   chain built on that FFT (c_frontend/libfmcw_cmsis.so), so the exact
   comparison applies to either build. --reference overrides the choice.

   Against a reference with a different FFT the results are close but not
   bit-identical. For that case --compare tolerant gives one of three
   verdicts: MATCH (the exact comparison passes), CLOSE (every difference
   is within the stated limits) or MISMATCH.

6. True targets. Every published frame carries the true range, velocity
   and azimuth of its targets. The board's detections and the laptop
   reference's are each judged against them: whether each target inside
   the map was found, the angle error, and how many detections lie away
   from every target, that is, outside the CFAR guard region of each. This judges the detection result itself, whichever
   FFT produced it. The reference in this report is always the chain on
   the project FFT, so for a CMSIS-DSP board it shows what the change of
   FFT does to the detections.

Imported from streaming_subscriber_nucleo.py unchanged: the serial
conversation (NucleoLink), frame and row packing, banner and result
parsing, the dimension and CFAR check at start-up, and the exact
comparison (compare_detections). The other command-line arguments are the
same.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
BAUD_TOL = 0.03                 # board rate against the rate in use
STAGES = ("power_cyc", "cells_cyc", "det_cyc", "doa_cyc", "emit_cyc")
PHASES = ("ready_s", "send_s", "read_s", "ref_s")

# Limits of the tolerant comparison.
SNR_TOL_DB = 0.5        # largest SNR difference of a paired detection
EDGE_DB = 1.0           # a detection only the board has must be this close
                        # to its own CFAR threshold
MAX_REF_ONLY = 1        # detections only the reference has, per frame
REL_TOL = 1e-6          # x, y and v_radial of a paired detection
MAX_MSGS = 6            # difference lines printed per frame
# Upper edges, in dB above the CFAR threshold, of the classes the SNR
# differences are reported in: weak detections first.
MARGIN_EDGES_DB = (3.0, 6.0, 10.0, 20.0)


def margin_class(margin_db: float) -> str:
    """The class of a detection that is margin_db above its threshold."""
    lo = 0.0
    for hi in MARGIN_EDGES_DB:
        if margin_db < hi:
            return f"{lo:g}-{hi:g} dB"
        lo = hi
    return f">={lo:g} dB"


def parse_banner_ext(lines: list[str]) -> dict:
    """The certified banner fields plus the ones only this build prints."""
    out = parse_banner(lines)
    for ln in lines:
        t = ln.split()
        if ln.startswith("# doa_arith") and len(t) >= 3:
            out["doa_arith"] = t[2]
        elif ln.startswith("# fft_impl") and len(t) >= 3:
            out["fft_impl"] = t[2]
        elif ln.startswith("# baud") and len(t) >= 3:
            out["baud"] = int(t[2])
    return out


def check_banner_baud(banner: dict, baud: int) -> str | None:
    """None when the board's UART rate is within BAUD_TOL of baud, else
    the reason to refuse. The board cannot produce every rate exactly, so
    a small difference is expected; a large one means the firmware was
    built for another rate."""
    got = banner.get("baud")
    if got is None:
        return "the board reports no baud rate (firmware without # baud)"
    if abs(got - baud) > BAUD_TOL * baud:
        return (f"the board's baud rate is {got}, the link is opened "
                f"at {baud}")
    return None


def board_margin_db(result: BoardResult) -> np.ndarray:
    """Per board detection, how far its power is above its CFAR threshold,
    in dB, in the order of det_rows."""
    if not result.dets:
        return np.empty(0)
    return np.array([10.0 * np.log10(d[7] / d[8]) for d in result.dets])


def resolve_reference(mode: str, banner: dict) -> str:
    """The FFT of the laptop reference: "own" or "cmsis". "auto" follows
    the FFT the board reports, so the two sides run the same arithmetic."""
    if mode != "auto":
        return mode
    return "cmsis" if banner.get("fft_impl") == "cmsis" else "own"


def reference_lib_path(fft: str) -> Path | None:
    """The library holding the batch chain on that FFT; None is CFrontend's
    default, the chain on the project FFT."""
    if fft == "own":
        return None
    return (Path(__file__).resolve().parents[1] / "c_frontend"
            / "libfmcw_cmsis.so")


def compare_tolerant(board: np.ndarray, ref: np.ndarray,
                     margin_db: np.ndarray,
                     snr_tol_db: float = SNR_TOL_DB,
                     edge_db: float = EDGE_DB,
                     max_ref_only: int = MAX_REF_ONLY):
    """Verdict for a board whose arithmetic is close to the reference's but
    not identical. Returns (verdict, messages, counts).

    MATCH when compare_detections passes. Otherwise detections are paired
    by their (range, doppler) cell, and the verdict is CLOSE when
    - each pair agrees in az within one grid step (one step is counted as
      a flip), in snr_db within snr_tol_db, in v_radial within REL_TOL
      relative, and in x and y within REL_TOL relative when az is equal;
    - each detection only the board has is at most edge_db above its own
      threshold (margin_db, one value per board row);
    - at most max_ref_only detections are only in the reference. The
      reference reports no threshold, so these cannot be checked one by
      one.
    MISMATCH otherwise."""
    ok, msgs, flips = compare_detections(board, ref)
    counts = {"flips": flips, "board_only": 0, "ref_only": 0,
              "max_snr_diff_db": 0.0, "max_board_only_margin_db": 0.0,
              "failed": [],          # the rules a MISMATCH frame broke
              "snr_diff_by_margin": {}}   # class -> [pairs, largest diff]
    if ok:
        return "MATCH", msgs, counts

    def close(a, c):
        return abs(a - c) <= REL_TOL * max(abs(a), abs(c), 1e-12)

    def cell(r):
        return (int(r[0]), int(r[1]))

    msgs, hard, flips = [], 0, 0
    failed = set()
    ref_at = {cell(r): r for r in ref}
    seen = set()
    for i, b in enumerate(board):
        key = cell(b)
        if key in seen:
            msgs.append(f"cell {key}: reported twice by the board")
            hard += 1
            failed.add("other")
            continue
        seen.add(key)
        r = ref_at.get(key)
        if r is None:
            m = float(margin_db[i])
            counts["board_only"] += 1
            counts["max_board_only_margin_db"] = max(
                counts["max_board_only_margin_db"], m)
            msgs.append(f"cell {key}: board only, {m:.2f} dB above threshold")
            if m > edge_db:
                hard += 1
                failed.add("board_only")
            continue
        d_snr = abs(b[6] - r[6])
        counts["max_snr_diff_db"] = max(counts["max_snr_diff_db"], d_snr)
        cls = counts["snr_diff_by_margin"].setdefault(
            margin_class(float(margin_db[i])), [0, 0.0])
        cls[0] += 1
        cls[1] = max(cls[1], d_snr)
        if d_snr > snr_tol_db:
            msgs.append(f"cell {key}: snr_db {b[6]:.4f} vs {r[6]:.4f}")
            hard += 1
            failed.add("snr")
        if not close(b[5], r[5]):
            msgs.append(f"cell {key}: v_radial {b[5]!r} != {r[5]!r}")
            hard += 1
            failed.add("other")
        d_az = abs(b[2] - r[2])
        if d_az == 1:
            flips += 1
            msgs.append(f"cell {key}: tie flip az {b[2]:.0f} vs {r[2]:.0f}")
        elif d_az > 1:
            msgs.append(f"cell {key}: az {b[2]:.0f} != {r[2]:.0f}")
            hard += 1
            failed.add("az")
        else:
            for j, name in ((3, "x"), (4, "y")):
                if not close(b[j], r[j]):
                    msgs.append(f"cell {key}: {name} {b[j]!r} != {r[j]!r}")
                    hard += 1
                    failed.add("other")
    for key in ref_at:
        if key not in seen:
            counts["ref_only"] += 1
            msgs.append(f"cell {key}: reference only")
    if counts["ref_only"] > max_ref_only:
        hard += 1
        failed.add("ref_only")
    counts["flips"] = flips
    counts["failed"] = sorted(failed)
    return ("MISMATCH" if hard else "CLOSE"), msgs, counts


MATCH_BINS = 1.0        # a target is found by a detection this close


def truth_targets(ground_truth: list[dict], range_per_bin: float,
                  velocity_axis, n_samples: int,
                  min_range_bin: int) -> list[dict]:
    """The frame's true targets in the board's bin coordinates.

    The bins are computed from the true range and velocity with the
    board's own range_per_bin and velocity axis. The range wraps at
    n_samples bins, as the sampled beat signal does; a target is in the
    map when its wrapped range lies in the kept bins."""
    v0, dv = velocity_axis[0], velocity_axis[1] - velocity_axis[0]
    out = []
    for t in ground_truth:
        rb = (t["range_m"] / range_per_bin) % n_samples
        db = ((t["velocity_mps"] - v0) / dv) % len(velocity_axis)
        out.append({"range_bin": rb, "doppler_bin": db,
                    "az_deg": t["azimuth_deg"],
                    "in_map": min_range_bin <= rb <= n_samples // 2 - 1})
    return out


def score_targets(det: np.ndarray, targets: list[dict], n_chirps: int,
                  margin_db: np.ndarray | None = None, *,
                  cluster_bins: tuple[float, float]) -> dict:
    """Judge detections (det_rows order) against the true targets.

    Per target: found is True when a detection lies within MATCH_BINS of
    it in range and in Doppler, False when none does, None when the
    target is outside the map. az_err_deg and peak_margin_db describe the
    strongest such detection (None without one, or without margin_db).

    cluster counts the detections that belong to some target: within
    cluster_bins = (range bins, Doppler bins) of it. away counts the
    rest. The caller passes the CFAR guard cells, the extent over which
    the front end itself expects a target to spread: a strong target's
    skirt is detected several bins from its peak. Doppler distance wraps
    around the axis."""
    def offsets(d, t):
        dd = abs(d[1] - t["doppler_bin"])
        return abs(d[0] - t["range_bin"]), min(dd, n_chirps - dd)

    found, az_err, peak = [], [], []
    for t in targets:
        if not t["in_map"]:
            found.append(None), az_err.append(None), peak.append(None)
            continue
        best = None
        for i, d in enumerate(det):
            dr, dd = offsets(d, t)
            if dr <= MATCH_BINS and dd <= MATCH_BINS and \
                    (best is None or d[6] > det[best][6]):
                best = i
        found.append(best is not None)
        az_err.append(None if best is None
                      else float(abs(det[best][2] - t["az_deg"])))
        peak.append(None if best is None or margin_db is None
                    else float(margin_db[best]))
    cluster = 0
    for d in det:
        for t in targets:
            dr, dd = offsets(d, t)
            if dr <= cluster_bins[0] and dd <= cluster_bins[1]:
                cluster += 1
                break
    return {"found": found, "az_err_deg": az_err, "peak_margin_db": peak,
            "cluster": cluster, "away": len(det) - cluster}


class TargetTally:
    """Session totals of score_targets, for the board and the reference."""

    def __init__(self, ref_name: str = "reference",
                 cluster_bins: tuple[float, float] | None = None):
        self.ref_name = ref_name
        self.cluster_bins = cluster_bins    # as given to score_targets
        self.per_target: dict[int, dict] = {}
        self.cluster = {"board": 0, "ref": 0}
        self.away = {"board": 0, "ref": 0}

    def add(self, board: dict, ref: dict) -> None:
        for side, sc in (("board", board), ("ref", ref)):
            self.cluster[side] += sc["cluster"]
            self.away[side] += sc["away"]
            for k, hit in enumerate(sc["found"]):
                if hit is None:
                    continue
                t = self.per_target.setdefault(k, {
                    "in_map": 0,
                    "found": {"board": 0, "ref": 0},
                    "az_max": {"board": 0.0, "ref": 0.0},
                    "peak": []})
                if side == "board":
                    t["in_map"] += 1
                    if sc["peak_margin_db"][k] is not None:
                        t["peak"].append(sc["peak_margin_db"][k])
                if hit:
                    t["found"][side] += 1
                    t["az_max"][side] = max(t["az_max"][side],
                                            sc["az_err_deg"][k])

    def lines(self) -> list[str]:
        if not self.per_target:
            return []
        out = ["true targets (found: a detection within "
               f"{MATCH_BINS:g} bin in range and Doppler):"]
        for k in sorted(self.per_target):
            t = self.per_target[k]
            peak = (f"{min(t['peak']):.1f} to {max(t['peak']):.1f} dB"
                    if t["peak"] else "n/a")
            out.append(
                f"   target {k}: in map {t['in_map']} frames | found: "
                f"board {t['found']['board']}, "
                f"{self.ref_name} {t['found']['ref']} | "
                f"largest angle error: "
                f"board {t['az_max']['board']:.1f} deg, "
                f"{self.ref_name} {t['az_max']['ref']:.1f} deg | "
                f"board peak above threshold: {peak}")
        window = "" if self.cluster_bins is None else (
            f" (within {self.cluster_bins[0]:g} range and "
            f"{self.cluster_bins[1]:g} Doppler bins)")
        out.append(
            f"   detections near a target{window}: "
            f"board {self.cluster['board']}, "
            f"{self.ref_name} {self.cluster['ref']} | "
            f"away from every target: board {self.away['board']}, "
            f"{self.ref_name} {self.away['ref']}")
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

    def read_banner(self, settle_s: float = 0.5) -> list[str]:
        """The banner the board is printing now. Output from before the
        port was opened can still be on its way from the ST-LINK, and it
        may come from a firmware that has since been replaced, so
        everything that arrives during settle_s is discarded first."""
        t_end = time.time() + settle_s
        while time.time() < t_end:
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
            else:
                time.sleep(0.01)
        return self.wait_ready()

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


def _process_one(link, fe, op, frame, frame_id, t_pub, row_pace: float = 0.0,
                 compare: str = "exact", targets: list[dict] | None = None,
                 fe_truth=None,
                 cluster_bins: tuple[float, float] | None = None):
    """Quantize, await RDY, send, read, diff one frame. Returns
    (verdict_line, result, time_fields, phase_seconds, outcome), where
    outcome is the comparison's counts plus "verdict", or None when the
    board reported an error. With targets (truth_targets), outcome also
    holds score_targets for the board and for a reference: fe_truth's
    detections when it is given, else those of fe, the chain the verdict
    comes from. cluster_bins is score_targets' window and is needed with
    targets.

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
        return (f"[frame {frame_id}] BOARD {res.err}", res, times, phases,
                None)
    ref = fe.q15_process_frame(frame)
    if compare == "tolerant":
        verdict, msgs, outcome = compare_tolerant(det_rows(res), ref,
                                                  board_margin_db(res))
    else:
        ok, msgs, flips = compare_detections(det_rows(res), ref)
        verdict = "MATCH" if ok else "MISMATCH"
        outcome = {"flips": flips, "board_only": 0, "ref_only": 0,
                   "max_snr_diff_db": 0.0, "max_board_only_margin_db": 0.0,
                   "failed": [], "snr_diff_by_margin": {}}
    outcome["verdict"] = verdict
    if targets is not None:
        n_chirps = frame.shape[1]
        outcome["score_board"] = score_targets(
            det_rows(res), targets, n_chirps, board_margin_db(res),
            cluster_bins=cluster_bins)
        truth_ref = ref if fe_truth is None \
            else fe_truth.q15_process_frame(frame)
        outcome["score_ref"] = score_targets(truth_ref, targets, n_chirps,
                                             cluster_bins=cluster_bins)
    phases["ref_s"] = time.time() - t3
    lat = time.time() - (t_pub if t_pub else t0)
    detail = f"{res.n} dets, tie flips {outcome['flips']}"
    if verdict == "CLOSE":
        detail += (f", board only {outcome['board_only']}, "
                   f"ref only {outcome['ref_only']}, "
                   f"max snr diff {outcome['max_snr_diff_db']:.3f} dB")
    line = f"[frame {frame_id}] {verdict} ({detail}) latency {lat:.2f} s"
    if compare == "tolerant":
        # the messages name every differing cell; the full arrays would
        # only repeat them
        shown = msgs[:MAX_MSGS]
        if len(msgs) > MAX_MSGS:
            shown.append(f"... and {len(msgs) - MAX_MSGS} more differences")
        if verdict == "MISMATCH":
            shown.append(f"board only {outcome['board_only']}, "
                         f"ref only {outcome['ref_only']}, "
                         f"max snr diff {outcome['max_snr_diff_db']:.3f} dB, "
                         f"max board-only margin "
                         f"{outcome['max_board_only_margin_db']:.2f} dB, "
                         f"reference has {len(ref)} dets, "
                         f"board overflow {res.overflow}")
        msgs = shown
    if msgs:
        line += "\n  " + "\n  ".join(msgs)
    if verdict == "MISMATCH" and compare != "tolerant":
        line += (f"\n  board: {np.array2string(det_rows(res), precision=6)}"
                 f"\n  ref:   {np.array2string(ref, precision=6)}")
    return line, res, times, phases, outcome


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
    ap.add_argument("--reference", choices=("auto", "own", "cmsis"),
                    default="auto",
                    help="FFT of the laptop reference; auto: the FFT the "
                         "board reports")
    ap.add_argument("--compare", choices=("exact", "tolerant"),
                    default="exact",
                    help="tolerant: for a reference whose FFT differs "
                         "from the board's")
    args = ap.parse_args()

    from processing.c_frontend import CFrontend
    from processing.c_frontend_onepass import CFrontendOnepass, read_params

    cfg = read_params(args.params)
    link = TimedLink(args.port, args.baud)
    b = parse_banner_ext(link.read_banner())
    want = (int(cfg["n_virtual"]), int(cfg["n_chirps"]), int(cfg["n_samples"]))
    alpha_ok = abs(b.get("alpha", 0) - cfg["cfar_alpha"]) \
        <= 1e-9 * abs(cfg["cfar_alpha"])
    if b.get("dims") != want or not alpha_ok:
        print(f"refusing: board {b} vs params dims {want} "
              f"alpha {cfg['cfar_alpha']}")
        raise SystemExit(1)
    baud_err = check_banner_baud(b, args.baud)
    if baud_err:
        print(f"refusing: {baud_err}")
        raise SystemExit(1)
    print(f"banner: dims {b['dims']} alpha {b['alpha']} "
          f"doa {b.get('doa_arith', 'not reported')} "
          f"fft {b.get('fft_impl', 'not reported')} "
          f"baud {b['baud']}  [accepted]")
    compare = args.compare
    ref_fft = resolve_reference(args.reference, b)
    lib_path = reference_lib_path(ref_fft)
    if lib_path is not None and not lib_path.exists():
        print(f"refusing: {lib_path} is not built — "
              f"make -C c_frontend lib-cmsis")
        raise SystemExit(1)
    print(f"reference: {ref_fft} FFT, comparison: {compare}")

    fe = CFrontend(args.params, lib_path=lib_path)
    # the true-target report keeps the project-FFT chain as its reference
    fe_truth = None if ref_fft == "own" else CFrontend(args.params)
    op = CFrontendOnepass(args.params)
    velocity_axis = [float(x) for x in cfg["velocity_axis"].split(",")]
    # a detection belongs to a target inside the CFAR guard region
    cluster_bins = (int(cfg["cfar_guard_r"]), int(cfg["cfar_guard_d"]))
    tally = TargetTally("reference" if fe_truth is None
                        else "own-FFT reference", cluster_bins)

    def targets_of(ground_truth):
        return truth_targets(ground_truth, cfg["range_per_bin"],
                             velocity_axis, int(cfg["n_samples"]),
                             int(cfg["min_range_bins"]))

    if args.one_shot:
        from simulator.beat_signal_generator import (FMCWBeatSignalGenerator,
                                                     load_scenario)
        rcfg, targets = load_scenario(args.scenario)
        gen = FMCWBeatSignalGenerator(rcfg)
        frame = gen.generate_frame_mimo(targets, frame_index=0)
        frame = np.ascontiguousarray(frame.astype(np.complex64))
        line, res, times, phases, outcome = _process_one(
            link, fe, op, frame, 0, None, args.row_pace, compare,
            targets_of(gen.ground_truth(targets, frame_index=0)), fe_truth,
            cluster_bins)
        print(line)
        print(f"TIME row_cyc {res.row_cyc} finish_cyc {res.finish_cyc}   "
              f"MEM heap {res.heap} stack_peak {res.stack_peak} "
              f"free_start {res.free_start}  SAT {res.sat}")
        print("\n".join(_timing_lines([times] if times else [], [phases],
                                      args.baud)))
        if outcome:
            tally.add(outcome["score_board"], outcome["score_ref"])
            print("\n".join(tally.lines()))
        link.close()
        raise SystemExit(
            0 if outcome and outcome["verdict"] != "MISMATCH" else 1)

    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    print(f"receiving from {args.connect}, feeding {args.port} "
          f"at {args.baud} baud")

    processed = mismatches = close_total = flips_total = drops = 0
    board_errors = 0                    # frames the board answered with ERR
    board_only_total = ref_only_total = ref_only_frame_max = 0
    snr_diff_max = board_only_margin_max = 0.0
    failed_frames: dict[str, int] = {}     # rule -> MISMATCH frames
    snr_by_margin: dict[str, list] = {}    # class -> [pairs, largest diff]
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
            line, res, times, phases, outcome = _process_one(
                link, fe, op, frame, fid, float(header["timestamp_unix_s"]),
                args.row_pace, compare,
                targets_of(header.get("ground_truth", [])), fe_truth,
                cluster_bins)
            print(line, flush=True)
            if res.err:
                board_errors += 1
                continue
            processed += 1
            tally.add(outcome["score_board"], outcome["score_ref"])
            mismatches += outcome["verdict"] == "MISMATCH"
            close_total += outcome["verdict"] == "CLOSE"
            flips_total += outcome["flips"]
            board_only_total += outcome["board_only"]
            ref_only_total += outcome["ref_only"]
            ref_only_frame_max = max(ref_only_frame_max, outcome["ref_only"])
            for cls, (n_pairs, d_max) in \
                    outcome["snr_diff_by_margin"].items():
                acc = snr_by_margin.setdefault(cls, [0, 0.0])
                acc[0] += n_pairs
                acc[1] = max(acc[1], d_max)
            for rule in outcome["failed"]:
                failed_frames[rule] = failed_frames.get(rule, 0) + 1
            snr_diff_max = max(snr_diff_max, outcome["max_snr_diff_db"])
            board_only_margin_max = max(board_only_margin_max,
                                        outcome["max_board_only_margin_db"])
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
                      f"{board_errors} board errors, "
                      f"{mismatches} mismatches, {close_total} close, "
                      f"{flips_total} tie flips | "
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
    print(f"board errors: {board_errors} frames (not processed, "
          f"not compared)")
    print(f"comparison {compare}: "
          f"{processed - close_total - mismatches} MATCH, "
          f"{close_total} CLOSE, {mismatches} MISMATCH | "
          f"tie flips {flips_total}, board only {board_only_total}, "
          f"ref only {ref_only_total} (at most {ref_only_frame_max} "
          f"in a frame) | "
          f"max snr diff {snr_diff_max:.3f} dB, "
          f"max board-only margin {board_only_margin_max:.2f} dB")
    if failed_frames:
        print("   MISMATCH frames by broken rule (a frame can break "
              "several): " + ", ".join(
                  f"{k} {v}" for k, v in sorted(failed_frames.items())))
    if snr_by_margin:
        print("   largest snr difference by how far the board's detection "
              "is above its threshold (frames that are not MATCH):")
        order = [margin_class(e - 0.01) for e in MARGIN_EDGES_DB] \
            + [margin_class(MARGIN_EDGES_DB[-1])]
        for cls in order:
            if cls in snr_by_margin:
                n_pairs, d_max = snr_by_margin[cls]
                print(f"      {cls:>9}: {n_pairs} pairs, "
                      f"largest {d_max:.3f} dB")
    for ln in tally.lines():
        print(ln)
    if plane_max_session:
        print("row deadline by baud rate, for the session's slowest row:")
        for ln in deadline_table(plane_max_session):
            print("   " + ln)


if __name__ == "__main__":
    main()
