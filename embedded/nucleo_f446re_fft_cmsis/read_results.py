"""Report the board's FFT comparison between the project and CMSIS-DSP.

The board reports the configuration it ran under. A wrong clock, a disabled
FPU or CMSIS-DSP built without loop unrolling gives numbers that look right
and measure the wrong thing, so a capture with any of those is refused.

A scale difference is different: the two FFTs may legitimately scale their
output differently. It is listed as a problem, because the CMP differences
are not comparable until it is resolved, but the numbers are still shown.

Usage:  python read_results.py --build-dir build
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPECT_CLK = 180_000_000
EXPECT_WS = 5
SIZES = (128, 32)
IMPLS = ("own", "cmsis")
SCALE_ABS_Q15 = 1
SCALE_REL_FLOAT = 1e-6
SCALE_PREFIX = "scale"

OUT_KEYS = {"n", "impl", "vec", "bin0_re", "bin0_im", "other_peak"}
CMP_KEYS = {"n", "vec", "max_abs_diff", "rms_diff", "own_peak", "cmsis_peak"}
TIME_KEYS = {"n", "impl", "iters", "copy_cyc", "total_cyc", "fft_cyc"}
MEM_KEYS = {"heap", "stack_peak", "free_start"}


@dataclass
class Stream:
    arith: str = ""
    config: dict[str, int] = field(default_factory=dict)
    out: dict[tuple[int, str], dict] = field(default_factory=dict)
    cmp: dict[int, dict] = field(default_factory=dict)
    time: dict[tuple[int, str], dict] = field(default_factory=dict)
    mem: dict = field(default_factory=dict)
    complete: bool = False


def _value(text: str):
    try:
        return int(text, 0)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _fields(tokens: list[str], required: set[str]) -> dict | None:
    """key=value tokens, or None when the row is mangled or incomplete."""
    pairs = dict(t.split("=", 1) for t in tokens if "=" in t)
    if not required <= pairs.keys():
        return None
    return {k: _value(v) for k, v in pairs.items()}


def _parse_block(lines: list[str]) -> Stream:
    """One run. A mangled line is skipped; the row check then reports it."""
    s = Stream()
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        kind = tokens[0]
        if kind == "#" and len(tokens) == 3:
            if tokens[1] == "arith":
                s.arith = tokens[2]
            elif isinstance(_value(tokens[2]), int):
                s.config[tokens[1]] = _value(tokens[2])
        elif kind == "OUT":
            row = _fields(tokens[1:], OUT_KEYS)
            if row and row["vec"] == "dc":
                s.out[(row["n"], row["impl"])] = row
        elif kind == "CMP":
            row = _fields(tokens[1:], CMP_KEYS)
            if row and row["vec"] == "radar":
                s.cmp[row["n"]] = row
        elif kind == "TIME":
            row = _fields(tokens[1:], TIME_KEYS)
            if row:
                s.time[(row["n"], row["impl"])] = row
        elif kind == "MEM":
            s.mem = _fields(tokens[1:], MEM_KEYS) or {}
        elif kind == "END":
            s.complete = True
    return s


def parse(text: str) -> Stream:
    """The last complete run in a capture.

    The board reports repeatedly, so a capture usually holds several runs and
    may begin or end part-way through one. If no run is complete, the last
    one is returned so that check can say what is missing.
    """
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("# nucleo-fft-cmsis"):
            blocks.append([])
        if blocks:
            blocks[-1].append(line)
    if not blocks:
        return Stream()
    for block in reversed(blocks):
        s = _parse_block(block)
        if s.complete:
            return s
    return _parse_block(blocks[-1])


def _check_config(s: Stream, expected_arith: str) -> list[str]:
    bad = []
    if s.arith != expected_arith:
        bad.append(f"arith is {s.arith!r}, expected {expected_arith!r}")
    if s.config.get("dsp_loopunroll") != 1:
        bad.append("dsp_loopunroll is not 1: CMSIS-DSP built without loop unrolling")
    if s.config.get("clk") != EXPECT_CLK:
        bad.append(f"clk is {s.config.get('clk')}, expected {EXPECT_CLK}")
    if s.config.get("fpu") != 1:
        bad.append("fpu is not 1: soft-float build or FPU not enabled")
    if s.config.get("flash_ws") != EXPECT_WS:
        bad.append(f"flash_ws is {s.config.get('flash_ws')}, expected {EXPECT_WS}")
    if (s.config.get("flash_acr", 0) >> 8) & 0x7 != 0x7:
        bad.append("flash_acr: prefetch, I-cache and D-cache not all enabled")
    return bad


def _check_rows(s: Stream) -> list[str]:
    bad = [] if s.complete else ["no END line: capture is truncated"]
    for n in SIZES:
        if n not in s.cmp:
            bad.append(f"missing CMP row for n={n}")
        for impl in IMPLS:
            if (n, impl) not in s.out:
                bad.append(f"missing OUT row for n={n} impl={impl}")
            if (n, impl) not in s.time:
                bad.append(f"missing TIME row for n={n} impl={impl}")
    if not s.mem:
        bad.append("missing MEM row")
    return bad


def _check_scale(s: Stream, arith: str) -> list[str]:
    bad = []
    for n in SIZES:
        own, cmsis = s.out.get((n, "own")), s.out.get((n, "cmsis"))
        if not own or not cmsis:
            continue
        a, b = own["bin0_re"], cmsis["bin0_re"]
        if arith == "q15":
            differs = abs(a - b) > SCALE_ABS_Q15
        else:
            differs = abs(a - b) > SCALE_REL_FLOAT * max(abs(a), abs(b))
        if differs:
            bad.append(f"{SCALE_PREFIX} mismatch at n={n}: DC bin own={a}, cmsis={b}")
    return bad


def check(stream: Stream, expected_arith: str) -> list[str]:
    """Problems with a capture. Empty means it can be trusted."""
    return (_check_rows(stream)
            + _check_config(stream, expected_arith)
            + _check_scale(stream, expected_arith))


def _parse_size(path: Path) -> dict[str, int]:
    """Flash is text + data, static RAM is data + bss, in arm-none-eabi-size's
    table: data is stored in flash and copied out at startup."""
    if not path.exists():
        return {}
    m = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+[0-9a-fA-F]+\s+\S+",
                  path.read_text(), re.M)
    if not m:
        return {}
    text, data, bss = (int(g) for g in m.groups())
    return {"flash": text + data, "static_ram": data + bss}


def _section(arith: str, s: Stream, problems: list[str], size: dict[str, int]) -> list[str]:
    lines = [f"{arith}:"]
    lines += [f"  problem: {p}" for p in problems]
    if any(not p.startswith(SCALE_PREFIX) for p in problems):
        lines.append("  capture refused, no results shown")
        return lines
    lines.append(f"  {'n':>5}{'own cyc':>12}{'cmsis cyc':>12}{'cmsis/own':>11}"
                 f"{'max diff':>10}{'rms diff':>10}")
    for n in SIZES:
        own = s.time[(n, "own")]["fft_cyc"]
        cmsis = s.time[(n, "cmsis")]["fft_cyc"]
        ratio = f"{cmsis / own:.2f}" if own else "-"
        c = s.cmp[n]
        lines.append(f"  {n:>5}{own:>12.0f}{cmsis:>12.0f}{ratio:>11}"
                     f"{c['max_abs_diff']:>10g}{c['rms_diff']:>10g}")
    lines.append(f"  flash {size.get('flash', '-')} B, "
                 f"static RAM {size.get('static_ram', '-')} B, "
                 f"heap {s.mem['heap']} B, stack peak {s.mem['stack_peak']} B")
    return lines


def _load(build_dir: Path) -> dict[str, tuple[Stream, list[str]]]:
    loaded = {}
    for arith in ("q15", "float"):
        path = build_dir / f"stream_{arith}_nucleo.txt"
        if path.exists():
            s = parse(path.read_text())
            loaded[arith] = (s, check(s, arith))
    return loaded


def report(build_dir: Path) -> str:
    """The results of every capture found in build_dir."""
    loaded = _load(build_dir)
    if not loaded:
        return f"no captures found in {build_dir}"
    lines = ["NUCLEO-F446RE FFT, project vs CMSIS-DSP", ""]
    for arith, (s, problems) in loaded.items():
        size = _parse_size(build_dir / f"size_{arith}.txt")
        lines += _section(arith, s, problems, size) + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the CMSIS-DSP FFT comparison.")
    ap.add_argument("--build-dir", default="build", type=Path)
    args = ap.parse_args()
    print(report(args.build_dir))
    loaded = _load(args.build_dir)
    if not loaded or any(problems for _, problems in loaded.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
