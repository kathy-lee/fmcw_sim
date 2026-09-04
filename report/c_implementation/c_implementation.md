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

The C front end reimplements the Python MIMO chain up to the
raw point cloud: ADC cube in, detections CSV out. Clustering and tracking
stay outside — the point cloud is the architectural interface a radar DSP
exports. Validation is two-layered: C unit tests against committed golden
vectors (`make -C c_frontend test`), and a pytest comparison against the
live Python pipeline on exported frames (`tests/test_c_frontend.py` —
power map, CFAR mask with a borderline-cell exemption, point cloud).

### How the pieces fit

One set of kernels is built into two shells — a command-line binary and a
shared library — so the batch timing runs and the Python benches measure
the same compute reached two different ways. No radar arithmetic lives in
either shell.
  
compute kernels: in `$(SRC)`, do no I/O, no config, no timing:
- `float`:    fft  rd  nci  cfar  doa  pointcloud
- `integer`:  fft_q15  rd_q15  nci_q15  cfar_q15  pointcloud_q15  convert_q15

`io.c` does following: 
- params.cfg parsing
- buffer allocation
- memory summary
- frame loading
- CSV and per-stage dump writing
```
            the kernels and io.c go down each side
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                            │
        ▼                                            ▼
   src/main.c                                  src/fmcw_api.c
   (args, frame loop, timers)                 (opaque handle, flat types)
        │                                            │
        ▼                                            ▼
   ./fmcw_frontend  (static)                   libfmcw.so  (-fPIC -shared)
   (memory summary, timing table,                    │
    detection count)                                 ▼
                                         processing/c_frontend.py
                                        (ctypes wrapper, validates arrays)
                                                     │
                                                     ▼
                                       evaluation/sqnr_fixed_point   (§7)
                                       evaluation/fixedpoint_c_eval  (§8)
                                    streaming_subscriber_mimo_hybrid.py (§5)
```

The two shells serve different consumers: `fmcw_frontend` is the
standalone batch entry point, existing so the chain can run and be timed
without Python — frames in from disk, detections out as CSV, per-stage
timing table at exit. `libfmcw.so` exists for in-process use: Python loads
it to run the same kernels on arrays it already holds, which is what makes
the frame-for-frame comparisons in this report possible.

`make` builds the binary, `make lib` builds the shared library; both
compile the same source list: libfmcw.so compiles `$(SRC)`, fmcw_frontend compiles `$(SRC)` plus `src/main.c`.

The two shells use `io.c` unevenly: `main.c` uses all of it, while
`fmcw_api.c` touches only config parsing and buffer allocation, since a
library caller supplies its own arrays and does its own I/O. Both
arithmetics are present in both products — the choice is made per call
(`--fixed` on the binary, the `q15_*` methods in Python), not at build
time, which is what allows a single process to run both chains on one
frame and compare them.

In this report, **Sections 2–4** cover the float port: the shared
library that made in-process comparison possible, how the port compares
against the Python pipeline, and the one optimization that decided those
numbers. **Section 5** covers the fixed-point port: int16 arithmetic
through CFAR, what it costs in precision, in detections, and in time and
memory. **Section 6** takes the same integer kernel to an 8-bit
microcontroller, where the same comparison comes out the other way, and
**Section 7** takes it to a Cortex-M4F, the class of part a front end
ships on, where it comes out the laptop's way again.

## 2. Python binding and the SIL bench

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


## 3. Float implementation
### 3.1 Comparison method

Three chains are timed: the Python pipeline, the C float chain, and the C
fixed-point chain. All three process the same 50 frames of
`scenario_mimo.json` (12 virtual channels, 64 chirps, 256 samples), but
they obtain those frames differently.

The two C runs read them from disk, exported once by
`scripts/export_for_c.py`. The Python run — `evaluation/time_mimo_pipeline.py`
— regenerates them in memory instead, from the same generator with the same
seed and the same frame indices, so it sees byte-identical data without
being charged for file reads the C timers exclude anyway. It also processes
one warm-up frame before measuring, because the first `cfar_2d` call
triggers a one-time empirical alpha calibration that would otherwise swamp
the real `cfar` figure; the C side has no equivalent cost, its alpha
arriving precalibrated in `params.cfg`.

Every timer brackets compute only — file I/O sits outside the measurement
on both sides.

```bash
# setup: 50 frames + params.cfg, consumed by both C runs below
uv run python scripts/export_for_c.py \
    --scenario scenarios/scenario_mimo.json --out data/c_export_bench --frames 50

# C, float          -> the C column of §3, and the float column of §9
./c_frontend/fmcw_frontend \
    --config data/c_export_bench/params.cfg --frames data/c_export_bench/frames

# C, fixed-point    -> the fixed column of §9
./c_frontend/fmcw_frontend \
    --config data/c_export_bench/params.cfg --frames data/c_export_bench/frames --fixed

# Python, float     -> the Python column of §3
uv run python -m evaluation.time_mimo_pipeline \
    --scenario scenarios/scenario_mimo.json --frames 50
```

Environment: laptop, single-threaded, `gcc -O2`, no SIMD tuning, CPU
frequency scaling active — absolute numbers are indicative, ratios are the
robust part. Each comparison table below pairs columns measured **in one
back-to-back session**, because absolute times drift with clock state while
ratios hold; the float column in §3 and the float column in §9 therefore
differ by a few percent despite measuring the same code.

### 3.2 Comparison results

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


## 5. Fixed-point implementation
### 5.1 scope and arithmetic

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

### 5.2 Signal quality evaluation

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

### 5.3 Detection performance

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

### 5.4 Cost in time and memory

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

## 6. The same kernel on an 8-bit microcontroller

§5 closed by saying the platform decides whether fixed point pays, and that
a desktop CPU with an FPU is not the platform it is for. That claim is
testable. `embedded/uno_fft_q15/` runs the same `fft_q15.c`, unmodified, on
an ATmega328P — 8-bit registers, 16 MHz, no FPU, no barrel shifter, 2 KB of
RAM — and compares it against the same file compiled for the host.

The comparison is a diff of two identically shaped streams. A host
reference compiles the kernel with `gcc` and prints its twiddle table,
both transforms, and its timing; the sketch prints the same over serial.
Input is a 64-sample chirp segment compiled into flash, so both machines
start from byte-identical integers and the only variable is the machine.

N = 64, both builds at `-O2`:

| | Q15 | float | ratio |
|---|---:|---:|---:|
| time per transform, Uno | 7974.4 µs | 17755.0 µs | **2.23x** |
| time per transform, host | 0.764 µs | 0.313 µs | **0.41x** |
| flash | 8366 B | 8064 B | |
| static RAM | 467 B | 725 B | |
| heap | 260 B | 388 B | |
| stack peak | 136 B | 139 B | |
| peak RAM of 2048 | 863 B | 1252 B | 0.69x |
| agreement with host | bit-exact | 2.09e-07 relative | |

**The sign flips.** Same source, same input, opposite verdict: fixed point
wins by 2.23x on the microcontroller and loses by 2.4x on the laptop. The
mechanism is not subtle — on a part with no FPU every float multiply is a
software routine, measured here at about 148 cycles per operation, while
the integer path pays 664 cycles per butterfly for arithmetic the AVR does
8 bits at a time. Neither figure is impressive; the ratio between them is
the point, and it is the ratio §5 could not obtain from one machine.

**Why the ratio is only 2.23x, and why it is a floor.** Disassembling the
integer image accounts for the gap. avr-gcc does not inline a 16x16->32
signed multiply; it emits a libgcc call, `__mulhisi3`, and there are four of
them in every butterfly — the four component products of the twiddle
multiply — each paying a call, a return and operand marshalling on top of
the `MUL` sequence itself. Around them sits 32-bit arithmetic carried out a
byte at a time, and a `>>15` that walks down one bit per instruction because
the part has no barrel shifter.

That is §5's x86 finding arriving from the opposite direction. There the
compiler vectorized the float butterfly and left the integer one scalar;
here it will not even inline the integer multiply. Both times the operation
dedicated hardware makes single-cycle — `PMULHRSW` on x86, `SMLAD` on a
Cortex-M4 — is precisely the one portable C cannot reach, so the compiler
substitutes something generic. So 2.23x is a floor for this kernel as
written in C rather than a property of the arithmetic, and it is the case
for CMSIS-DSP's intrinsics made twice, on two architectures that share
nothing else.

**Q15 is bit-exact across the two instruction sets.** The twiddle table and
both transforms match the host count for count. This was the result most at
risk: the twiddle table is the one part of a fixed-point build computed in
floating point, and avr-gcc makes `double` a 32-bit type, so a twiddle near
a rounding boundary could have landed one count off and propagated through
every stage. At this size it does not. That exactness is what lets the
integer chain serve as a reference model — develop on the host, trust the
target. The float column makes no such claim and does not need to: one ULP
of float32 separates it from the host, which is all IEEE 754 promises
across compilers.

**Memory is the structural result, and it does not depend on a benchmark.**
863 bytes against 1252, on a part with 2048. At N = 128 the integer build
would need about 1375 bytes and still fit; the float build would need about
2148 and would not. That is the same argument §5 made about the radar cube
and on-chip SRAM, at a scale small enough to verify by counting.

**What this does not establish.** An 8-bit AVR is not what FMCW front ends
run on; the contested case is a Cortex-M4F, with both a single-cycle FPU
and the `SMLAD` DSP instructions, and nothing here speaks to it. The full
front end cannot run on this part at all — 1.2 MiB of working set against
2 KB. And at 7974 µs per 64-point transform, one frame's transforms would
take about 30 seconds, so nothing here is a feasibility claim. The result
is a bracket: §3 and §5 measure one end, this measures the other, and the
hardware the pipeline would actually ship on sits between them.

## 7. The same kernel on a Cortex-M4F

§6 ended on a bracket: the laptop at one end, the AVR at the other, and
the hardware FMCW front ends actually run on somewhere between. That
hardware is a Cortex-M4F — a single-cycle FPU *and* the `SMLAD` family of
packed 16-bit multiply-accumulates — so both arithmetics have a unit built
for them, and which one wins is genuinely open. `embedded/nucleo_f446re_fft/`
puts the same two files on an STM32F446RE at 180 MHz, bare metal, and
measures. The harness is the Uno's with its timer, serial and stack
symbols exchanged, plus one addition the AVR did not need: the board
prints the clock, FPU and flash-cache configuration it is running under,
and the reader refuses a stream whose configuration is wrong, since a
16 MHz boot clock or a soft-float build gives correct output and numbers
eleven or several times off.

N = 256, both builds at `-O2`, cycles from `DWT->CYCCNT` with the input
reload subtracted:

| | Q15 | float | ratio |
|---|---:|---:|---:|
| cycles per transform | 78140 | 40870 | **0.52x** |
| cycles per butterfly | 76.3 | 39.9 | |
| time per transform, Nucleo | 434.11 µs | 227.06 µs | |
| time per transform, host | 3.773 µs | 1.773 µs | 0.47x |
| flash | 35824 B | 35604 B | |
| static RAM | 3500 B | 4524 B | |
| heap | 1552 B | 2064 B | |
| stack peak | 744 B | 832 B | |
| peak RAM of 131072 | 5796 B | 7420 B | 1.28x |
| agreement with host | bit-exact | 8.2e-08 relative | |

One kernel, three machines, cycles per butterfly: Uno Q15 664, Nucleo Q15
76, Nucleo float 40.

**The sign flips back, and by the predicted amount.** The design note for
this harness recorded, before the run, an expectation that portable-C
float would be 1.5–2.5x faster on this core; it is 1.9x. So on the part
the pipeline would ship on, fixed point written in plain C loses on time
by nearly the same ratio as on the laptop, and the AVR result of §6 is the
exception among the three, not the rule. What Q15 keeps is what it kept
there: bit-exactness, and 1.6 KiB less RAM at this size from half-width
buffers.

**Why: the compiler reached one unit and not the other.** The float
`fft_inplace` compiles to 87 lines with no calls, its butterfly 18 VFP
instructions — six loads, four stores, the complex twiddle product as
`vmul` + `vnmul` + two `vfma`, then two `vadd` and two `vsub`. Multiply
and add fuse into one instruction with one rounding, and there is no
explicit rounding step anywhere. The Q15 `fft_q15_inplace` compiles to
207 lines, also without a single library call: the barrel shifter and the
single-cycle 32x32 multiplier remove every one of the AVR's `__mulhisi3`
costs, which is why 664 cycles became 76. But the twiddle product is four
ordinary `mul`/`mla`/`mls` on sign-extended halfwords, and around it sit
six rounding shifts per butterfly that the format requires. Not one
`smulbb`, `smuad` or `smlad` appears. The instruction that does a Q15
complex multiply-accumulate in two operations exists on this core, and
`-O2` never emitted it — the third time in this report that the operation
dedicated hardware makes single-cycle is exactly the one portable C does
not reach, after `PMULHRSW` on x86 (§5) and the inlined multiply on AVR
(§6). The fixed-point deficit here is therefore a statement about the
compiler, not the arithmetic, and CMSIS-DSP's `arm_cfft_q15`, written
around those intrinsics, is where the arithmetic's own cost would be
measured. That column is deferred, and named as the one thing that would
change this section's verdict.

**Q15 is bit-exact across the three instruction sets.** 128 twiddles and
both 256-point transforms match the host count for count, as 32 twiddles
did on the Uno. The concern §6 raised — a twiddle rounded one count off
during the floating-point setup — does not arise here, since
arm-none-eabi-gcc keeps `double` at 64 bits, and the dump confirms it.
The float column differs from the host by 8.2e-08 relative, one float32
ULP, and the cause is visible in the disassembly: `vfma` rounds once
where the host's separate multiply and add round twice. Same source, two
IEEE-conforming results — which is the reason the integer chain, and not
the float one, can serve as a reference model across machines.

**What this does not establish.** Nothing about CMSIS-DSP, as above.
Nothing about the pipeline: this is one transform on one row, and the
front end as written needs ~1.57 MiB of working set against 128 KiB. The
restructure that would fit it — static storage, a streaming range stage,
one channel's range–Doppler plane at a time, a detection cap, and a second
pass over the input for the DoA snapshots, about 100 KiB at the full
scenario — is designed but not built. And 434 µs per transform is a
number about a kernel, not a real-time claim about a frame.

## 8. Limitations and future work

**Measurement.** Three machines — a desktop x86-64, an 8-bit AVR and a
Cortex-M4F — but the pipeline numbers come from the first alone; the other
two measure one FFT. Single run of 50 frames on the desktop, frequency
scaling uncontrolled; no attempt at SIMD, threading, or FFT library
substitution on the C side, and on the Cortex-M4F no CMSIS-DSP column, so
§7 measures portable C rather than the arithmetic's ceiling there. Ratios were stable across repeated runs; absolute times varied
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
4. **CMSIS-DSP on the Cortex-M4F.** §7 settled the portable-C case —
   float wins by 1.9x — and showed why: the `SMLAD` instructions never
   appear in the compiled Q15 butterfly. `arm_cfft_q15` is written around
   them, so a third column in `embedded/nucleo_f446re_fft/` is what would
   separate the arithmetic's cost from the compiler's. Its rounding and
   scaling differ from this kernel's, so that column is a comparison of
   two kernels, not a bit-exact check of one.
5. **The pipeline on the Cortex-M4F**, which needs the two-pass
   restructure §7 describes before it fits in 128 KiB.

**One caveat on the Hann window**, which costs the integer path something
float does not notice: it attenuates most samples by around 6 dB on
average, so roughly a bit of the container goes unused after windowing.
Recovering it with a post-window left shift is not possible, since the
window peak is ~1.0 and would overflow. It is a real cost of the format,
documented rather than fixed.
