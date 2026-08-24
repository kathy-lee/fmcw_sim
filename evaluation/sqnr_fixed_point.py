"""Per-stage signal quality of the C fixed-point front end vs the float one.

Both chains run through a single CFrontend handle on the same exported
frames, so the only difference between the columns is the arithmetic.

What it reports, and why more than one number is needed:

- SQNR per stage, 10*log10(sum|ref|^2 / sum|ref - fixed*scale|^2), for the
  RD cube and the power map. These two disagree sharply and both are
  honest. The cube's figure sums over every cell while the error is about
  one LSB in each, so the tens of thousands of near-empty noise cells
  dominate the error and contribute almost nothing to the signal. The power
  map weights by |rd|^4, concentrating the measure on target cells where
  quantization is negligible.

- Peak |int16| and the noise floor in counts, which show the mechanism the
  SQNR figures only hint at: how much of the 16-bit container the data
  actually occupies. Shift-every-stage divides by S*C unconditionally,
  while real data grows by rather less — one target's amplitude is well
  below the composite ADC peak, so the range FFT concentrates it by far
  less than S. The cube therefore sits well below full scale, a typical
  cell carries only a few bits above the quantization step, and that is
  what holds the cube's SQNR down. It is also precisely what block
  floating point would reclaim, by shifting only when the block max
  demands it.

- CFAR mask agreement and detection counts, because a poor cube SQNR does
  not imply lost detections: CFAR is a ratio test and quantization lifts a
  cell and its neighbours together.

- Saturation counts. Nonzero means an input assumption failed, not that
  precision was merely lost.

Usage:
    (uv run) python -m evaluation.sqnr_fixed_point --export data/c_export_bench

Build the library first:  make -C c_frontend lib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from processing.c_frontend import CFrontend

# An ideal n-bit converter driven by a full-scale sine reaches
# 6.02n + 1.76 dB. Quoted beside the measurements as the ceiling the ADC
# alone would impose, before the FFT's per-stage rounding is counted.
ADC_BITS = 12
IDEAL_SQNR_DB = 6.02 * ADC_BITS + 1.76


def read_cfg(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, val = line.split("=", 1)
            out[key] = val
    return out


def sqnr_db(ref: np.ndarray, got: np.ndarray) -> float:
    err = float(np.sum(np.abs(ref - got) ** 2))
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(np.sum(np.abs(ref) ** 2)) / err))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fixed-point vs float signal quality, per stage.")
    ap.add_argument("--export", default="data/c_export_bench",
                    help="directory holding params.cfg and frames/")
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = every frame in the directory")
    args = ap.parse_args()

    root = Path(args.export)
    cfg = read_cfg(root / "params.cfg")
    paths = sorted((root / "frames").glob("frame_*.bin"))
    if args.frames:
        paths = paths[:args.frames]
    if not paths:
        raise SystemExit(f"no frames found in {root / 'frames'}")

    fe = CFrontend(root / "params.cfg")
    V, C, S = fe.dims
    rows = []

    for path in paths:
        raw = np.fromfile(path, dtype=np.complex64).reshape(V, C, S)
        fe.reset_saturations()

        rd_f = fe.rd(raw)
        rd_q = fe.q15_rd(raw)
        power_f = fe.nci(rd_f)
        power_q = fe.q15_nci(rd_q)
        mask_f = fe.cfar(power_f)
        mask_q = fe.q15_cfar(power_q)

        # Magnitude of every cube cell in raw counts, for the occupancy and
        # floor figures. The median stands in for the noise floor: the cube
        # is overwhelmingly noise cells, so it is not pulled by targets.
        mag = np.hypot(rd_q[..., 0].astype(np.float64),
                       rd_q[..., 1].astype(np.float64))

        rows.append(dict(
            frame=path.stem,
            sqnr_rd=sqnr_db(rd_f, fe.dequantize_rd(rd_q)),
            sqnr_power=sqnr_db(power_f, power_q * fe.power_scale),
            peak=float(mag.max()),
            floor=float(np.median(mag)),
            agree=100.0 * float((mask_f == mask_q).mean()),
            det_f=int(mask_f.sum()),
            det_q=int(mask_q.sum()),
            sat=fe.saturations(),
        ))

    fe.close()

    print(f"fixed-point signal quality ({len(rows)} frames, "
          f"V={V} C={C} S={S}, adc_full_scale={cfg['adc_full_scale']})")
    print(f"  ideal {ADC_BITS}-bit converter: {IDEAL_SQNR_DB:.1f} dB "
          f"(full-scale sine, before any FFT rounding)\n")
    head = (f"  {'frame':<12}{'rd dB':>9}{'power dB':>10}{'peak':>9}"
            f"{'floor':>8}{'mask %':>9}{'det f/q':>10}{'sat':>10}")
    print(head)
    for r in rows:
        print(f"  {r['frame']:<12}{r['sqnr_rd']:>9.1f}{r['sqnr_power']:>10.1f}"
              f"{r['peak']:>9.0f}{r['floor']:>8.1f}{r['agree']:>9.3f}"
              f"{r['det_f']:>5d}/{r['det_q']:<4d}"
              f"{r['sat'][0] + r['sat'][1]:>10d}")

    mean = lambda k: float(np.mean([r[k] for r in rows]))          # noqa: E731
    print(f"\n  {'mean':<12}{mean('sqnr_rd'):>9.1f}{mean('sqnr_power'):>10.1f}"
          f"{mean('peak'):>9.0f}{mean('floor'):>8.1f}{mean('agree'):>9.3f}"
          f"{sum(r['det_f'] for r in rows):>5d}"
          f"/{sum(r['det_q'] for r in rows):<4d}"
          f"{sum(sum(r['sat']) for r in rows):>10d}")

    peak, floor = mean('peak'), mean('floor')
    bits_used = np.log2(max(peak, 1.0))
    print(f"\n  cube occupancy: peak {peak:.0f} counts of 32767 "
          f"(~{bits_used:.1f} of 16 bits), noise floor {floor:.1f} counts")
    print(f"  {16.0 - bits_used:.1f} bits of the container never carry a "
          f"value, discarded by the unconditional per-stage shift. A typical "
          f"cell holds {floor:.1f} counts against roughly half a count of "
          f"accumulated rounding error, which is what puts the cube near "
          f"{mean('sqnr_rd'):.0f} dB while the power map, weighted toward "
          f"target cells, reads {mean('sqnr_power'):.0f} dB.")


if __name__ == "__main__":
    main()
