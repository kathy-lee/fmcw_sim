"""Compare the board's stream against the host's and report the cost.

Precision here is a verification, not a measurement. Q15 arithmetic is
deterministic — same input integers, same shifts, same rounding, same
output on any conforming compiler — so the board must match the host to the
last count. If it does, the transform's precision on the board is whatever
evaluation/sqnr_fixed_point.py already measured, and there is nothing new
to compute. The float column cannot make that claim, and the asymmetry is
itself worth printing.

The board also reports the configuration it ran under. A wrong clock or a
soft-float build gives numbers that look right and measure the wrong
thing, so those two are refused outright rather than reported.

Usage:  python read_results.py --build-dir build
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FLOAT_RTOL = 1e-4
EXPECT_CLK = 180_000_000
EXPECT_WS = 5


@dataclass
class Stream:
    arith: str = ""
    n: int = 0
    config: dict[str, int] = field(default_factory=dict)
    twiddle: list[tuple[float, float]] = field(default_factory=list)
    vectors: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    mem: dict[str, int] = field(default_factory=dict)
    complete: bool = False


def parse_block(lines: list[str]) -> Stream:
    """One run. Malformed lines are skipped rather than raising: a capture
    can begin part-way through a line, and one mangled sample should not
    cost the whole run."""
    s = Stream()
    current = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        f = line.split()
        try:
            if line.startswith("# arith "):
                s.arith = f[2]
            elif line.startswith("# n "):
                s.n = int(f[2])
            elif line.startswith("# ") and len(f) == 3:
                s.config[f[1]] = int(f[2], 0)
            elif f[0] == "TW" and len(f) == 4:
                s.twiddle.append((float(f[2]), float(f[3])))
            elif f[0] == "VEC" and len(f) == 2:
                current = f[1]
                s.vectors[current] = []
            elif f[0] == "OUT" and len(f) == 4 and current is not None:
                s.vectors[current].append((float(f[2]), float(f[3])))
            elif f[0] == "TIME" and len(f) % 2 == 1:
                s.timing = {f[i]: float(f[i + 1]) for i in range(1, len(f), 2)}
            elif f[0] == "MEM" and len(f) % 2 == 1:
                s.mem = {f[i]: int(f[i + 1]) for i in range(1, len(f), 2)}
            elif line == "END":
                s.complete = True
        except (ValueError, IndexError):
            continue
    return s


def parse(text: str) -> Stream:
    """The last complete run in a capture.

    The board reports on a timer, so a capture usually holds several runs,
    the first of them starting part-way through. Splitting on the header and
    keeping the last block that reached END discards both the fragment at the
    front and any run the capture cut short at the back. A host stream, being
    a single run, falls through this unchanged.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("# nucleo-fft"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    if not blocks:
        return Stream()

    for block in reversed(blocks):
        s = parse_block(block)
        if s.complete:
            return s
    return parse_block(blocks[-1])


def check_config(s: Stream) -> list[str]:
    """Reasons to refuse a device stream. Empty means it ran as intended."""
    bad = []
    if s.config.get("clk") != EXPECT_CLK:
        bad.append(f"clk is {s.config.get('clk')}, expected {EXPECT_CLK}")
    if s.config.get("fpu") != 1:
        bad.append("fpu is not 1 — soft-float build or FPU not enabled")
    if s.config.get("flash_ws") != EXPECT_WS:
        bad.append(f"flash_ws is {s.config.get('flash_ws')}, expected {EXPECT_WS}")
    acr = s.config.get("flash_acr", 0)
    if (acr >> 8) & 0x7 != 0x7:
        bad.append(f"flash_acr 0x{acr:x}: prefetch/I-cache/D-cache not all enabled")
    return bad


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
    """Flash and static RAM from arm-none-eabi-size's Berkeley table:
    flash = text + data (data is stored in flash and copied out at
    startup), static RAM = data + bss."""
    if not path.exists():
        return {}
    m = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+[0-9a-fA-F]+\s+\S+",
                  path.read_text(), re.M)
    if not m:
        return {}
    text, data, bss = (int(g) for g in m.groups())
    return {"flash": text + data, "static_ram": data + bss}


def butterflies(n: int) -> int:
    return (n // 2) * int(math.log2(n)) if n >= 2 else 0


def verdict(dev: Stream, host: Stream) -> list[str]:
    lines = []
    if dev.arith == "q15":
        tw_bad, tw_worst = compare_exact(dev.twiddle, host.twiddle)
        if tw_bad == 0:
            lines.append("  twiddle table       exact match")
        else:
            lines.append(f"  twiddle table       {tw_bad} of {len(host.twiddle)} differ, "
                         f"max {tw_worst:g} count(s)")
            lines.append("                      this is the floating-point setup path, "
                         "not the integer transform")
    else:
        tw_bad, tw_worst = compare_tol(dev.twiddle, host.twiddle, FLOAT_RTOL)
        lines.append(f"  twiddle table       max relative diff {tw_worst:.2e}")

    for name in ("dc", "radar"):
        if name not in dev.vectors or name not in host.vectors:
            lines.append(f"  {name:<20}missing from one stream")
            continue
        if dev.arith == "q15":
            bad, worst = compare_exact(dev.vectors[name], host.vectors[name])
            if bad == 0:
                lines.append(f"  {name:<20}bit-exact vs host")
            else:
                cause = " (explained by the twiddle table)" if tw_bad else ""
                lines.append(f"  {name:<20}{bad} of {len(host.vectors[name])} bins "
                             f"differ, max {worst:g}{cause}")
        else:
            bad, worst = compare_tol(dev.vectors[name], host.vectors[name], FLOAT_RTOL)
            lines.append(f"  {name:<20}max relative diff {worst:.2e} "
                         f"({bad} beyond {FLOAT_RTOL:g})")

    # Independent of any twiddle rounding: the constant's path never rotates
    # by anything but w[0], so this must hold even when the tables differ.
    dc = dev.vectors.get("dc")
    if dc and dev.arith == "q15":
        ok = dc[0] == (1000.0, 0.0) and all(v == (0.0, 0.0) for v in dc[1:])
        lines.append(f"  dc identity         {'holds' if ok else 'BROKEN'}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the Nucleo FFT harness results.")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--ram", type=int, default=131072, help="target SRAM in bytes")
    args = ap.parse_args()
    bd = Path(args.build_dir)

    cols = {}
    for arith in ("q15", "float"):
        dev_p, host_p = bd / f"stream_{arith}_nucleo.txt", bd / f"stream_{arith}_host.txt"
        if not dev_p.exists() or not host_p.exists():
            continue
        dev, host = parse(dev_p.read_text()), parse(host_p.read_text())
        if not dev.complete:
            print(f"{arith}: device stream is truncated — no END line. "
                  f"Saw {len(dev.vectors)} vector(s), "
                  f"{'a' if dev.timing else 'no'} TIME line.", file=sys.stderr)
            sys.exit(1)
        problems = check_config(dev)
        if problems:
            print(f"{arith}: device configuration refused:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            sys.exit(1)
        cols[arith] = (dev, host, parse_size(bd / f"size_{arith}.txt"))

    if not cols:
        print(f"no streams found in {bd}", file=sys.stderr)
        sys.exit(1)

    names = list(cols)
    print("\nNucleo-F446RE FFT harness\n")
    print(f"  {'metric':<26}" + "".join(f"{n:>14}" for n in names))

    def row(label, fn):
        cells = "".join(f"{fn(*cols[n]):>14}" for n in names)
        print(f"  {label:<26}{cells}")

    def per_butterfly(d, h, s):
        b = butterflies(d.n)
        return f"{d.timing['fft_cyc'] / b:.1f}" if b and "fft_cyc" in d.timing else "-"

    row("cycles per transform", lambda d, h, s: f"{d.timing.get('fft_cyc', 0):.0f}")
    row("cycles per butterfly", per_butterfly)
    row("time per transform [us]", lambda d, h, s: f"{d.timing.get('fft_us', 0):.2f}")
    row("  of which reload [us]", lambda d, h, s: f"{d.timing.get('copy_us', 0):.2f}")
    row("host per transform [us]", lambda d, h, s: f"{h.timing.get('fft_us', 0):.3f}")
    row("flash [bytes]", lambda d, h, s: str(s.get("flash", "-")))
    row("static RAM [bytes]", lambda d, h, s: str(s.get("static_ram", "-")))
    row("heap [bytes]", lambda d, h, s: str(d.mem.get("heap", "-")))
    row("stack peak [bytes]", lambda d, h, s: str(d.mem.get("stack_peak", "-")))

    # Withheld unless every component is present: a sum that quietly drops a
    # missing term still looks like a complete answer.
    def peak_ram(d, h, s):
        if "static_ram" not in s or "stack_peak" not in d.mem:
            return "-"
        return str(s["static_ram"] + d.mem.get("heap", 0) + d.mem["stack_peak"])

    row(f"peak RAM of {args.ram}", peak_ram)

    if len(names) == 2:
        q = cols["q15"][0].timing.get("fft_cyc", 0)
        f = cols["float"][0].timing.get("fft_cyc", 0)
        qh = cols["q15"][1].timing.get("fft_us", 0)
        fh = cols["float"][1].timing.get("fft_us", 0)
        if q:
            print(f"\n  float/Q15 on the Nucleo  {f / q:.2f}x")
        if qh:
            print(f"  float/Q15 on the host    {fh / qh:.2f}x")

    for arith, (dev, host, _) in cols.items():
        print(f"\n  {arith} against the host:")
        for line in verdict(dev, host):
            print(line)
    print()


if __name__ == "__main__":
    main()
