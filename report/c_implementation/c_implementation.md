# C Implementation of the MIMO Detection Front End

**Date:** 2026-08-21
**Branch:** `feat/c-implementation-front-end`
**Pipeline under test:** `c_frontend/` — dependency-free C11 port of the MIMO
detection front end (per-channel range-Doppler → NCI → CA-CFAR → Bartlett
DoA → point cloud), validated stage-by-stage against the Python reference
and timed against it on identical frames.

---

## 1. Scope

The C front end (`c_frontend/`) reimplements the Python MIMO chain up to the
raw point cloud: ADC cube in, detections CSV out. Clustering and tracking
stay outside — the point cloud is the architectural interface a radar DSP
exports. Validation is two-layered: C unit tests against committed golden
vectors (`make -C c_frontend test`), and a pytest comparison against the
live Python pipeline on exported frames (`tests/test_c_frontend.py` —
power map, CFAR mask with a borderline-cell exemption, point cloud). All
pass; the numbers below are from that validated state.

## 2. Timing method

Both implementations process the same 50 frames of `scenario_mimo.json`
(12 virtual channels, 64 chirps, 256 samples). Frames are exported once;
the Python timer regenerates them in-process (same generator, seed, and
frame indices), and runs one warm-up frame first so the one-time CFAR alpha
calibration stays out of the measurement. Both timing reports bracket
compute only (file I/O excluded).

```bash
uv run python scripts/export_for_c.py \
    --scenario scenarios/scenario_mimo.json --out data/c_export_bench --frames 50

./c_frontend/fmcw_frontend \
    --config data/c_export_bench/params.cfg --frames data/c_export_bench/frames

uv run python -m evaluation.time_mimo_pipeline \
    --scenario scenarios/scenario_mimo.json --frames 50
```

Environment: laptop, single-threaded, `gcc -O2`, no SIMD tuning, CPU
frequency scaling active — absolute numbers are indicative, ratios are the
robust part.

## 3. Results

Mean microseconds per frame over 50 frames:

| stage            | Python              | C      | ratio        |
|------------------|--------------------:|-------:|--------------|
| rd               | 2949 (numpy.fft)    |   2420 | C 1.2x       |
| integrate / nci  | 265 (numpy)         |     71 | C 3.7x       |
| cfar             | 336 (scipy.ndimage) |    101 | C 3.3x       |
| doa_pc           | 1696 (numpy)        |     52 | C 33x        |
| total (no cluster) | ~5246             |   2644 | **C ~2x**    |

Python's `cluster` stage (668 us) has no C counterpart, so totals are
compared without it. Stage names map as `integrate` = `nci`. The
parenthesized library is what dominates each Python stage: `rd` is
`numpy.fft` (pocketfft) calls, `integrate` one vectorized NumPy sum,
`cfar` two `scipy.ndimage.uniform_filter` passes, while `doa_pc` is a
Python-level loop over detected cells making small NumPy calls — which is
exactly where the C ratio explodes. The C column uses no libraries; every
stage is hand-written C (libm supplies only scalar math functions such as
`cos` and `log10`).

## 4. Analysis

- **`doa_pc` (33x)** — Python loops per detected cell over tiny arrays
  (12-element snapshots, 181x12 steering products), where interpreter and
  wrapper overhead dwarf the arithmetic. C runs the same loop with a
  precomputed steering matrix and no per-call overhead.
- **`cfar` / `nci` (~3x)** — Python is one vectorized scipy/NumPy call
  here; C's win is mostly the removed wrapper overhead. The C CFAR uses
  separable sliding-window sums (O(1) per cell); its first, naive version
  (O(window) per cell, kept as `cfar_ca_mask_ref`) ran at ~6400 us —
  18x *slower* than scipy. The algorithm, not the language, made the
  difference.
- **`rd` (~1.2x)** — nearly a tie: `np.fft` is pocketfft (SIMD-optimized),
  which the hand-rolled textbook radix-2 cannot match per transform; C
  claws the difference back on the per-chirp/per-channel call overhead.
- **Tails** — C's max stays within ~15% of its mean (fixed buffers, single
  code path); Python's max runs ~1.4x its mean (allocator/GC jitter). The
  tail is what a real-time budget cares about.
- **Budget context** — 2.6 ms per frame against a typical 50–100 ms radar
  frame period leaves ~20–40x headroom on one laptop core.

## 5. Sliding-window CFAR

The C CFAR was first written as the naive definition: for every cell, walk
the full training window and sum it — O(window) per cell, ~325 fused ops at
this geometry. Correct (it matched the golden masks cell for cell), but it
timed at ~6400 us: 18x *slower* than scipy, whose `uniform_filter` is a
separable running average doing O(1) work per cell. A naive algorithm in a
fast language lost to a good algorithm in a slow one.

The production version now uses the same idea, in the form practice uses:

1. Copy the power map (+1e-12) once into a border-replicated padded buffer,
   realizing scipy's `mode="nearest"` edge semantics up front instead of
   clamping indices per access.
2. Separable sliding sums: per row, a running window over columns (add the
   entering cell, subtract the leaving one); then per column, a running
   window over those row sums. O(1) per cell regardless of window size.
   Run twice — full window and guard window — then threshold
   `(full − guard)/n_train` per cell as before.

The sliding form was chosen over a summed-area table deliberately: its
accumulators hold at most one window's worth of data (a SAT entry grows
with the whole map), which is the property that survives a future
fixed-point port, and it is the shape hardware CFAR engines stream.

The two summation orders differ only by float rounding (~1e-12 relative),
far inside the margin of any non-borderline cell, so the detection masks
are identical in practice. That is enforced, not assumed: the naive loop
is kept as `cfar_ca_mask_ref`, and `test_cfar` holds both implementations
to cell-for-cell agreement with the golden mask and with each other; the
pytest comparison against live scipy re-passed unchanged. Result:
~6400 us → ~101 us (~50x), flipping the pipeline totals in C's favor.

## 6. Limitations

Single machine, single run of 50 frames, frequency scaling uncontrolled; no
attempt at SIMD, threading, or FFT library substitution on the C side.
Ratios were stable across repeated runs; absolute times varied by up to 2x
with CPU clock state.
