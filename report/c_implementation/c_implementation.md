# C Implementation of the MIMO Detection Front End

**Date:** float port 2026-08-21, fixed-point port 2026-08-24
**Branch:** `feat/c-implementation-front-end`
**Pipeline under test:** dependency-free C11 port of the MIMO
detection front end (per-channel range-Doppler → NCI → CA-CFAR → Bartlett
DoA → point cloud), in two arithmetics: float32, and a true-integer Q15
path. Both are validated stage-by-stage against the Python reference and
measured against it on identical frames.

---

## 1. Scope and structure

The C front end (`c_frontend/`) reimplements the Python MIMO chain up to the
raw point cloud: ADC cube in, detections CSV out. Clustering and tracking
stay outside — the point cloud is the architectural interface a radar DSP
exports. Validation is two-layered: C unit tests against committed golden
vectors (`make -C c_frontend test`), and a pytest comparison against the
live Python pipeline on exported frames (`tests/test_c_frontend.py` —
power map, CFAR mask with a borderline-cell exemption, point cloud). All
pass; the numbers below are from that validated state.

The report has two halves. **Sections 2–5** cover the float port: how it
compares against the Python pipeline, the one optimization that decided
those numbers, and the shared library that made in-process comparison
possible. **Sections 6–9** cover the fixed-point port: int16 arithmetic
through CFAR, what it costs in precision, in detections, and in time and
memory. Section 10 closes with limitations that apply to both.

The fixed-point half is the more interesting one, and its result is not
the one the bit-width label suggests. Storing the radar cube in 16 bits
turns out not to be the binding constraint; how the arithmetic is *scaled*
between stages is, and it costs nearly half the weak-target detections.

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

./c_frontend/fmcw_frontend \
    --config data/c_export_bench/params.cfg --frames data/c_export_bench/frames --fixed

uv run python -m evaluation.time_mimo_pipeline \
    --scenario scenarios/scenario_mimo.json --frames 50
```

Environment: laptop, single-threaded, `gcc -O2`, no SIMD tuning, CPU
frequency scaling active — absolute numbers are indicative, ratios are the
robust part. Each comparison table below pairs columns measured **in one
back-to-back session**, because absolute times drift with clock state while
ratios hold; the float column in §3 and the float column in §9 therefore
differ by a few percent despite measuring the same code.

## 3. Float implementation: results and analysis

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

Reading the rows:

- **`doa_pc` (33x)** — Python loops per detected cell over tiny arrays
  (12-element snapshots, 181x12 steering products), where interpreter and
  wrapper overhead dwarf the arithmetic. C runs the same loop with a
  precomputed steering matrix and no per-call overhead.
- **`cfar` / `nci` (~3x)** — Python is one vectorized scipy/NumPy call
  here; C's win is mostly the removed wrapper overhead. The C CFAR uses
  separable sliding-window sums (O(1) per cell); its first, naive version
  (O(window) per cell, kept as `cfar_ca_mask_ref`) ran at ~6400 us —
  18x *slower* than scipy. The algorithm, not the language, made the
  difference. §4 tells that story.
- **`rd` (~1.2x)** — nearly a tie: `np.fft` is pocketfft (SIMD-optimized),
  which the hand-rolled textbook radix-2 cannot match per transform; C
  claws the difference back on the per-chirp/per-channel call overhead.
- **Tails** — C's max stays within ~15% of its mean (fixed buffers, single
  code path); Python's max runs ~1.4x its mean (allocator/GC jitter). The
  tail is what a real-time budget cares about.
- **Budget context** — 2.6 ms per frame against a typical 50–100 ms radar
  frame period leaves ~20–40x headroom on one laptop core.

## 4. Sliding-window CFAR

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

That fixed-point claim was later cashed in, and improved on: the integer
sliding sums agree with the naive reference **exactly**, with no tolerance
and no borderline exemption, because integer addition is exact where float
running sums accumulate drift (§6).

## 5. Python binding and the SIL bench

The same kernels are also built as a shared library, `libfmcw.so`, behind
an opaque-handle API — the vendor-SDK idiom: `fmcw_create(params.cfg)`,
then per-stage or whole-frame calls, then `fmcw_destroy`. No struct crosses
the FFI boundary in either direction, and the batch binary and the library
share one set of kernels.

`processing/c_frontend.py` wraps it with stdlib `ctypes` — no codegen, no
build dependency, mechanics visible. Every array's dtype, shape and
contiguity is validated on the Python side *before* any pointer crosses, on
the principle that a `ValueError` beats a segfault. NumPy `complex64`
arrays are byte-identical to the C `cplx` layout, so ADC and RD cubes pass
with zero copy and zero conversion.

Two things this bought:

**Software-in-the-loop validation.** `test_c_binding_matches_python`
compares the C library against the Python reference **per stage,
in-process**, with no export-and-dump file round-trip. This is the step
embedded teams use to verify deployment C against a reference model, and
it is the infrastructure the entire fixed-point half of this report runs
on: §7's and §8's measurements are only possible because both arithmetics
are callable on the same frame from one process.

**A live A/B.** `simulator/streaming_subscriber_mimo_hybrid.py
--frontend {python,c}` runs the full tracked demo — ZMQ transport,
Cartesian clustering, CTRV-EKF, three-panel plot — with either front end
doing the math, so one flag against the same seeded publisher switches
implementations. Observed over live streaming: **C ~3–4 ms per frame
against Python's ~4–7 ms**, both comfortably real-time with zero dropped
frames. The more interesting detail is the shape rather than the mean:
Python's `doa_pc` swings between ~0.1 and ~1.9 ms as the detection count
varies, while C stays flat at ~0.05 ms. That is §3's table and its tail
argument reproduced in a live streaming context rather than a batch
benchmark.

## 6. Fixed-point implementation: scope and arithmetic

The integer path is **true fixed-point arithmetic**, not the
quantize-in-float model of report 5: int16 I/Q data, int32 registers, int64
accumulators, and no floating-point operation anywhere between the ADC and
the detection mask.

| stage | float path | fixed path |
|---|---|---|
| ADC frame | `cplx` (float32 I/Q) | `cplx_q15` (int16 I/Q), 12-bit quantized |
| range FFT | float butterflies | int16 storage, int32 registers, `>>1` per stage |
| clutter / Doppler | float | int16, saturating |
| RD cube | `cplx` | `cplx_q15` — half the memory |
| NCI power | `double` | `int64` |
| CFAR | `double` sliding sums | `int64` sliding sums, Q16 alpha |
| DoA + point cloud | float | *the same float code*, fed converted values |

DoA stays float deliberately. Only a few dozen cells survive CFAR, each
needing a 12-element snapshot, against a Bartlett search doing 181×12
complex multiply-accumulates per detection — so the conversion is cheap,
and more importantly both paths then run the *identical* DoA function. Any
angle or SNR difference between them is therefore attributable to the
integer front end alone, with no second quantized implementation to
confound it. That partition also matches how the hardware is built: a
fixed-point accelerator for FFT, integration and CFAR, with angle
estimation and tracking in float on a DSP afterwards.

**Bit growth is managed by shifting every stage** — an unconditional
rounding `>>1` after each butterfly. The safety argument has to be made on
complex *magnitude*, not per-component: rotation preserves magnitude, so a
twiddle multiply's output component is bounded by `|a|`, and a value with
both components near full scale has magnitude 46341, past int16. Stated
properly it is exact: given `|a|, |b| ≤ 32767` and `|w| = 1`, then
`|a ± w·b| ≤ 65534`, and halving returns it inside 32767 — the invariant
reproduces itself stage after stage. The consequence downstream is that
the output sits at a *statically known* `1/(S·C)`, so the whole scale
ledger is one compile-time constant rather than a per-frame exponent.

Two results fell out that are worth stating as results rather than
implementation notes:

- **CFAR needs no scale information at all.** It compares a cell against a
  scaled mean of its neighbours, so the global factor cancels and the
  ledger simply stops at NCI. Only alpha becomes fixed-point (Q16). The
  integer form also multiplies where the float form divides, which
  **removes the `+1e-12` epsilon entirely** — the float path needs it to
  avoid dividing by a zero noise estimate, and the integer path has no such
  hazard.
- **Integer NCI is exact.** `Σ(re² + im²)` in int64 involves no rounding
  whatsoever, so of every stage in the chain, integration is the one the
  fixed-point port costs nothing at all.

## 7. Fixed-point: signal quality

Measured by `evaluation/sqnr_fixed_point.py` over the same 50 frames of
`scenario_mimo`, comparing both chains cell by cell through one library
handle. SQNR is `10·log10(Σ|float|² / Σ|float − integer·scale|²)`.

| quantity | value |
|---|---|
| RD cube SQNR | 23.5 dB |
| power map SQNR | 61.9 dB |
| CFAR mask agreement | 99.999% |
| detections, float vs integer | 694 vs 693 |
| saturation events | 0 |
| cube peak | 679 counts of 32767 (~9.4 of 16 bits) |
| cube noise floor | 7.2 counts |

*(An ideal 12-bit converter driven by a full-scale sine reaches 74.0 dB;
that is the ceiling the ADC alone would impose before any FFT rounding.)*

**The two SQNR figures disagree by nearly 40 dB and both are honest.** The
cube's measure sums over all 98,304 cells while the error is roughly one
count in each, so the tens of thousands of near-empty noise cells dominate
the error term and contribute almost nothing to the signal term. The power
map weights by `|rd|⁴`, which concentrates the measure on target cells
where quantization is negligible. Same data, two questions.

**The cube's 23.5 dB is a headroom problem, and the occupancy figures name
it.** The cube peaks at 679 counts of a possible 32767 — about 9.4 of 16
bits — so **6.6 bits of the container never carry a value**. A typical
cell holds 7.2 counts against roughly half a count of accumulated rounding
error, which is where 23 dB comes from directly. The cause is
shift-every-stage: it divides by `S·C = 16384` unconditionally, while this
data does not grow by anything like that much, because a single target's
amplitude is far below the composite ADC peak that sets the input scale.
We are discarding bits that were never needed.

Zero saturation across all 50 frames confirms the input assumptions hold:
nothing exceeded the ADC's full scale, and the clutter-removal subtraction
never clipped.

**Do not read 694 vs 693 as "fixed point is free."** It says that on *this*
scenario, whose targets are strong, quantization cannot plausibly lose
them. The next section runs the same implementation on a scenario built to
test the opposite case, and the answer changes completely.

## 8. Fixed-point: detection performance

Measured by `evaluation/fixedpoint_c_eval.py` on
`scenario_mimo_fixedpoint_dynrange.json` — two targets 62 dB apart
(amplitudes 0.3 and 0.00025), so the weak one sits near threshold, roughly
1.7 ADC codes per sample. Metrics, gating and seeds are report 5's
`frame_metrics` reused unchanged, so these numbers are directly comparable
with that study's bit-width sweep.

10 frames × 5 seeds = 50 trials:

| metric | Python float | C float | C int16 |
|---|---:|---:|---:|
| weak-target Pd | 1.000 | 1.000 | **0.540** |
| Pfa per cell | 3.42e-05 | 3.42e-05 | 1.71e-05 |
| weak SNR [dB] | 6.92 | 6.92 | 5.20 |
| SFDR [dB] | 27.43 | 27.43 | 27.37 |

**The middle column is a control, not a result.** Per-stage equivalence
between the Python and C float chains is already proven by the test suite,
so it had to land within Monte-Carlo noise of the first column — it in fact
matches to every printed digit, which is what makes the third column
trustworthy.

**The integer chain loses the weak target in nearly half the trials.**
Output SNR on that target falls 1.72 dB, and Pd is a steep function of SNR
at threshold, so a marginal detection goes from certain to a coin flip.
Meanwhile **SFDR is unchanged** (27.43 → 27.37): the strong target and its
spurs are untouched. That is the predicted shape exactly — damage
invisible on strong returns, concentrated on weak ones near the threshold,
which is precisely where Pd lives.

The halved Pfa is the same mechanism rather than an improvement: a coarser
power map crosses the threshold less often, for genuine detections and
false alarms alike.

**This answers two questions report 5 left open in its own limitations
section.** That study swept bit widths using per-frame block scaling
(`scaling='block'`) and wrote that its recommended widths were
"conditional on BFP", with "whether a static Q-format or BFP better fits
this pipeline" left unanswered; it also noted that computing the FFT in
float made its knees "optimistic by perhaps a bit or two", deferring a
bit-true implementation to a "sign-off". This port is that sign-off, and it
implements exactly the static Q-format that was never compared. At the same
nominal 16 bits, the static scaling costs **46% of weak-target
detections** — so the widths really were conditional on BFP, and the
optimism was worth considerably more than a bit or two at the margin.

The mechanism is measured rather than inferred: §7 found the cube using 9.4
of its 16 bits, so weak signals live in the bottom few bits and are the
first thing rounding destroys.

## 9. Fixed-point: cost in time and memory

Both binaries run back to back on the same 50 frames, same thermal state:

| stage | float | fixed | ratio |
|---|---:|---:|---|
| quant | — | 1633.0 | *simulation only, see below* |
| rd | 2557.6 | 5054.7 | 1.98x slower |
| nci | 74.0 | 76.5 | 1.03x — a tie |
| cfar | 103.8 | 129.3 | 1.25x slower |
| doa_pc | 52.5 | 52.0 | 1.01x faster |
| **total** (excl. quant) | **2787.9** | **5312.5** | **1.91x slower** |

`quant` is excluded from the comparison because a real converter *emits*
integers: `adc_to_q15` exists only because this pipeline's input is a float
simulation, so charging the integer path for it would measure scaffolding
the hardware never runs. It belongs in the error budget, which §7 covers,
not the compute budget.

**Fixed point is nearly twice as slow here, and that is the expected
result on this machine.** Count the operations in one butterfly: float
needs about 10 (four multiplies and two adds for the complex multiply, four
adds for sum and difference); Q15 needs about 22 (four integer multiplies,
two adds, two rounding adds, two shifts and two saturation checks in the
twiddle multiply, then four add-and-shift pairs). Roughly double the work,
and the measurement says 1.98x. Fixed point is paying for the `>>15`
rescale after every multiply and the rounding shift every stage — both of
which float gets free from the hardware's exponent.

`nci` tying at ~75 µs is the other half of that story: pure
multiply-accumulate with no rescaling, nothing for fixed point to pay
extra for, and int64 and double issue at the same rate.

Inspecting the generated assembly sharpens the point. Neither path uses AVX
or FMA — the build targets the x86-64 baseline — but gcc does apply SLP
vectorization to the float butterfly's `(re, im)` pair, emitting packed
`mulps`/`addps`/`subps`, while the Q15 butterfly comes out **entirely
scalar** (`imull`, `sarl`, `movswl`, no packed integer instructions at
all), because the widening to int32 and the rescaling shift defeat the same
transformation. So this measures scalar C fixed-point against
pair-vectorized float on a machine with a fast FPU. Real fixed-point DSP
earns its keep through packed 16-bit multiply-accumulate instructions —
CMSIS-DSP uses `__SMLAD` and relatives on Cortex-M4/M7 to do two at once —
and portable C expresses none of that.

Memory, from the same runs:

| buffer | float | Q15 |
|---|---:|---:|
| ADC cube | 1536 KiB | 768 KiB |
| RD cube | 768 KiB | 384 KiB |
| power map | 64 KiB | 64 KiB |

Both cubes halve exactly, as the int16 width promises. The power map does
not, because `int64` and `double` are the same width — worth stating rather
than rounding the claim up. On a part where the radar cube must live in
on-chip SRAM, halving it is the difference between fitting and spilling to
external memory, which is the argument fixed point actually wins on.

**The lesson of this half is the counterpart to §3's.** There, a naive
algorithm in a fast language lost to a good algorithm in a slow one:
algorithm beats language. Here, the same arithmetic is 1.91x slower on a
desktop CPU with an FPU and 2x smaller in memory — the platform decides
whether fixed point pays, and this platform is not the one it is for.

## 10. Limitations and future work

**Measurement.** Single machine, single run of 50 frames, frequency scaling
uncontrolled; no attempt at SIMD, threading, or FFT library substitution on
the C side. Ratios were stable across repeated runs; absolute times varied
by up to 2x with CPU clock state, which is why each table's columns are
measured back to back and only ratios are compared across tables.

**Detection statistics.** §8's Pd figures rest on 50 trials, so the
resolution is 1/50 and the 0.540 figure carries a standard error near 0.07.
That is ample for the effect measured — 1.000 against 0.540 is not a
borderline call — but small differences in the other metrics should not be
over-read.

**Fixed-point scope.** The integer path stops at CFAR; DoA and the point
cloud remain float (§6), so nothing here characterizes an integer angle
estimator. Bit widths are fixed at int16/int64 rather than configurable,
so this report measures one point rather than a sweep — report 5 covers
the sweep, under a different scaling model.

**Where the fixed-point work should go next**, in priority order:

1. **Block floating point.** §7 showed the cube using 9.4 of 16 bits and
   §8 showed what that costs. BFP shifts only when the block maximum
   demands it rather than unconditionally every stage, so it should
   reclaim most of the 6.6 unused bits — worth roughly 30 dB of cube SQNR,
   and the direct remedy for the weak-target Pd loss. It is no longer a
   generic next step but the specific fix for a specific measured failure,
   and report 5's block-scaled sweep suggests where it should land.
2. **Integer DoA**, reusing the existing integer FFT for an angle FFT,
   which would complete the fixed-point chain.
3. **SIMD**, which is what would make the timing comparison represent
   fixed point rather than scalar C.

**One caveat on the Hann window**, which costs the integer path something
float does not notice: it attenuates most samples by around 6 dB on
average, so roughly a bit of the container goes unused after windowing.
Recovering it with a post-window left shift is not possible, since the
window peak is ~1.0 and would overflow. It is a real cost of the format,
documented rather than fixed.
