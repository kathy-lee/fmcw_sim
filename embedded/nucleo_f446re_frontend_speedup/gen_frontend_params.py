#!/usr/bin/env python3
"""Render params.cfg as a compile-time header for the board.

This script reads params.cfg and writes frontend_params.h, a C header 
containing the same values as a const FrontendParams struct (stored in 
flash) plus #define FRONTEND_V 12 etc. that size the static arrays. 
Python's repr(float) guarantees every double survives the trip through text exactly.

The board has no filesystem, so the flat key=value contract becomes a
const FrontendParams in flash plus the #defines that size the static
regions. Floats are emitted via repr(), which round-trips exactly through
a correctly rounded strtod — the board's parameters are the laptop's to
the last bit.

The header also carries FRONTEND_DOA_W, the 181-entry steering table of the
single-precision DoA scan: one complex factor per grid angle, computed here
in double and rounded to float. That scan needs a uniform array, so element
positions that are not multiples of one spacing are refused."""
import argparse
import math
import struct
from pathlib import Path

DOA_N_GRID = 181


def parse_cfg(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, val = line.split("=", 1)
        out[key] = val
    return out


def _f32(x):
    """x rounded to single precision, as a C float literal."""
    return repr(struct.unpack("f", struct.pack("f", x))[0]) + "f"


def doa_table(pos_wl):
    """Lines of the FRONTEND_DOA_W initializer for element positions pos_wl
    (wavelengths), which must be e * d for one spacing d."""
    d = pos_wl[1] - pos_wl[0] if len(pos_wl) > 1 else 0.0
    for e, x in enumerate(pos_wl):
        if abs(x - e * d) > 1e-12:
            raise SystemExit(
                f"pos_wl is not a uniform array: element {e} is at {x!r}, "
                f"expected {e * d!r} for spacing {d!r}")
    lines = [f"static const cplx FRONTEND_DOA_W[{DOA_N_GRID}] = {{"]
    for g in range(DOA_N_GRID):
        ph = 2.0 * math.pi * d * math.sin(math.radians(g - 90))
        lines.append(f"    {{{_f32(math.cos(ph))}, {_f32(math.sin(ph))}}},")
    lines.append("};")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = parse_cfg(a.cfg)

    ints = ["n_virtual", "n_chirps", "n_samples", "n_tx", "n_rx",
            "doa_n_sources", "cfar_guard_d", "cfar_guard_r",
            "cfar_training_d", "cfar_training_r", "min_range_bins"]
    dbls = ["range_per_bin", "pfa", "cfar_alpha", "adc_full_scale"]
    vel = [repr(float(x)) for x in c["velocity_axis"].split(",")]
    pos = [repr(float(x)) for x in c["pos_wl"].split(",")]
    assert len(vel) == int(c["n_chirps"]) and len(pos) == int(c["n_virtual"])
    table = doa_table([float(x) for x in c["pos_wl"].split(",")])

    lines = [f"/* generated from {a.cfg} — do not edit */",
             "#ifndef FRONTEND_PARAMS_H",
             "#define FRONTEND_PARAMS_H",
             '#include "fmcw.h"',
             '#include "params.h"',
             f"#define FRONTEND_V {int(c['n_virtual'])}",
             f"#define FRONTEND_C {int(c['n_chirps'])}",
             f"#define FRONTEND_S {int(c['n_samples'])}",
             "static const FrontendParams FRONTEND_PARAMS = {"]
    for k in ints:
        lines.append(f"    .{k} = {int(c[k])},")
    for k in dbls:
        lines.append(f"    .{k} = {repr(float(c[k]))},")
    lines.append("    .velocity_axis = { " + ", ".join(vel) + " },")
    lines.append(f"    .n_velocity = {len(vel)},")
    lines.append("    .pos_wl = { " + ", ".join(pos) + " },")
    lines.append(f"    .n_pos = {len(pos)},")
    lines.append("};")
    lines += table
    lines += ["#endif", ""]
    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
