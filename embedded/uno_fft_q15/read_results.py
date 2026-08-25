"""Compare the board's stream against the host's and report the cost.

Precision here is a verification, not a measurement. Q15 arithmetic is
deterministic — same input integers, same shifts, same rounding, same
output on any conforming compiler — so the board must match the host to the
last count. If it does, the transform's precision on the board is whatever
evaluation/sqnr_fixed_point.py already measured, and there is nothing new
to compute. The float column cannot make that claim, and the asymmetry is
itself worth printing.

Usage:  python read_results.py --build-dir build
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FLOAT_RTOL = 1e-4      # avr-gcc's 32-bit double against the host's 64-bit


@dataclass
class Stream:
    arith: str = ""
    n: int = 0
    twiddle: list[tuple[float, float]] = field(default_factory=list)
    vectors: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    mem: dict[str, int] = field(default_factory=dict)
    complete: bool = False


def parse(text: str) -> Stream:
    s = Stream()
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# arith "):
            s.arith = line.split()[2]
        elif line.startswith("# n "):
            s.n = int(line.split()[2])
        elif line.startswith("TW "):
            _, _, re_, im_ = line.split()
            s.twiddle.append((float(re_), float(im_)))
        elif line.startswith("VEC "):
            current = line.split()[1]
            s.vectors[current] = []
        elif line.startswith("OUT ") and current is not None:
            _, _, re_, im_ = line.split()
            s.vectors[current].append((float(re_), float(im_)))
        elif line.startswith("TIME "):
            f = line.split()[1:]
            s.timing = {f[i]: float(f[i + 1]) for i in range(0, len(f), 2)}
        elif line.startswith("MEM "):
            f = line.split()[1:]
            s.mem = {f[i]: int(f[i + 1]) for i in range(0, len(f), 2)}
        elif line == "END":
            s.complete = True
    return s


def compare_exact(a, b):
    """Count mismatches and the worst absolute difference."""
    bad = 0
    worst = 0.0
    for (ar, ai), (br, bi) in zip(a, b):
        if ar != br or ai != bi:
            bad += 1
            worst = max(worst, abs(ar - br), abs(ai - bi))
    return bad, worst


def compare_tol(a, b, rtol):
    bad = 0
    worst = 0.0
    scale = max((max(abs(r), abs(i)) for r, i in b), default=1.0) or 1.0
    for (ar, ai), (br, bi) in zip(a, b):
        d = max(abs(ar - br), abs(ai - bi))
        worst = max(worst, d)
        if d > rtol * scale:
            bad += 1
    return bad, worst / scale


def parse_size(path: Path) -> dict[str, int]:
    """Pull flash and static RAM out of arduino-cli's compile output."""
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    m = re.search(r"Sketch uses (\d+) bytes", text)
    if m:
        out["flash"] = int(m.group(1))
    m = re.search(r"Global variables use (\d+) bytes", text)
    if m:
        out["static_ram"] = int(m.group(1))
    return out


def verdict(uno: Stream, host: Stream) -> list[str]:
    lines = []
    if uno.arith == "q15":
        tw_bad, tw_worst = compare_exact(uno.twiddle, host.twiddle)
        if tw_bad == 0:
            lines.append("  twiddle table       exact match")
        else:
            lines.append(f"  twiddle table       {tw_bad} of {len(host.twiddle)} differ, "
                         f"max {tw_worst:g} count(s)")
            lines.append("                      this is the floating-point setup path, "
                         "not the integer transform")
    else:
        tw_bad, tw_worst = compare_tol(uno.twiddle, host.twiddle, FLOAT_RTOL)
        lines.append(f"  twiddle table       max relative diff {tw_worst:.2e}")

    for name in ("dc", "radar"):
        if name not in uno.vectors or name not in host.vectors:
            lines.append(f"  {name:<20}missing from one stream")
            continue
        if uno.arith == "q15":
            bad, worst = compare_exact(uno.vectors[name], host.vectors[name])
            if bad == 0:
                lines.append(f"  {name:<20}bit-exact vs host")
            else:
                cause = " (explained by the twiddle table)" if tw_bad else ""
                lines.append(f"  {name:<20}{bad} of {len(host.vectors[name])} bins "
                             f"differ, max {worst:g}{cause}")
        else:
            bad, worst = compare_tol(uno.vectors[name], host.vectors[name], FLOAT_RTOL)
            lines.append(f"  {name:<20}max relative diff {worst:.2e} "
                         f"({bad} beyond {FLOAT_RTOL:g})")

    # Independent of any twiddle rounding: the constant's path never rotates
    # by anything but w[0], so this must hold even when the tables differ.
    dc = uno.vectors.get("dc")
    if dc and uno.arith == "q15":
        ok = dc[0] == (1000.0, 0.0) and all(v == (0.0, 0.0) for v in dc[1:])
        lines.append(f"  dc identity         {'holds' if ok else 'BROKEN'}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the Uno FFT harness results.")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--ram", type=int, default=2048, help="target SRAM in bytes")
    args = ap.parse_args()
    bd = Path(args.build_dir)

    cols = {}
    for arith in ("q15", "float"):
        uno_p, host_p = bd / f"stream_{arith}_uno.txt", bd / f"stream_{arith}_host.txt"
        if not uno_p.exists() or not host_p.exists():
            continue
        uno, host = parse(uno_p.read_text()), parse(host_p.read_text())
        if not uno.complete:
            print(f"{arith}: device stream is truncated — no END line. "
                  f"Saw {len(uno.vectors)} vector(s), "
                  f"{'a' if uno.timing else 'no'} TIME line.", file=sys.stderr)
            sys.exit(1)
        cols[arith] = (uno, host, parse_size(bd / f"size_{arith}.txt"))

    if not cols:
        print(f"no streams found in {bd}", file=sys.stderr)
        sys.exit(1)

    names = list(cols)
    print("\nUno FFT harness\n")
    print(f"  {'metric':<26}" + "".join(f"{n:>14}" for n in names))

    def row(label, fn):
        cells = "".join(f"{fn(*cols[n]):>14}" for n in names)
        print(f"  {label:<26}{cells}")

    row("time per transform [us]", lambda u, h, s: f"{u.timing.get('fft_us', 0):.1f}")
    row("  of which reload [us]", lambda u, h, s: f"{u.timing.get('copy_us', 0):.1f}")
    row("host per transform [us]", lambda u, h, s: f"{h.timing.get('fft_us', 0):.3f}")
    row("flash [bytes]", lambda u, h, s: str(s.get("flash", "-")))
    row("static RAM [bytes]", lambda u, h, s: str(s.get("static_ram", "-")))
    row("heap [bytes]", lambda u, h, s: str(u.mem.get("heap", "-")))
    row("stack peak [bytes]", lambda u, h, s: str(u.mem.get("stack_peak", "-")))
    # Withheld unless every component is present: a sum that quietly drops a
    # missing term still looks like a complete answer.
    def peak_ram(u, h, s):
        if "static_ram" not in s or "stack_peak" not in u.mem:
            return "-"
        return str(s["static_ram"] + u.mem.get("heap", 0) + u.mem["stack_peak"])

    row(f"peak RAM of {args.ram}", peak_ram)

    if len(names) == 2:
        q = cols["q15"][0].timing.get("fft_us", 0)
        f = cols["float"][0].timing.get("fft_us", 0)
        qh = cols["q15"][1].timing.get("fft_us", 0)
        fh = cols["float"][1].timing.get("fft_us", 0)
        if q:
            print(f"\n  float/Q15 on the Uno   {f / q:.2f}x")
        if qh:
            print(f"  float/Q15 on the host  {fh / qh:.2f}x")

    for arith, (uno, host, _) in cols.items():
        print(f"\n  {arith} against the host:")
        for line in verdict(uno, host):
            print(line)
    print()


if __name__ == "__main__":
    main()
