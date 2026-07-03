# Fixed-Point MIMO Front-End — Evaluation Report

**Pipeline stage under test:** the MIMO detection front end (ADC → range-Doppler → non-coherent integration → CFAR input)
**Fidelity:** boundary quantization (stages compute in float with values re-gridded at boundaries)
**Golden reference:** the float pipeline (`fp=None`)

---

## 1. Introduction

This study exercises an
opt-in fixed-point mode on our own MIMO front end to show the realistic process:
which steps go fixed-point, how each step's bit width is chosen, and what it
costs against the float reference. The result is an early estimate of how many bits each stage needs, not the exact,
final check that would run before committing a design to hardware, §5 lists the
limitations.

The evaluation uses one **dynamic-range scene**
(`scenarios/scenario_mimo_fixedpoint_dynrange.json`): a strong close-in return
together with a weak, distant, off-boresight target that sits just above the float
detection threshold. This is the hard case for finite precision — the strong return
sets the quantization scale, so the weak target is the first thing lost as bits are
starved.

Each candidate bit width is judged against the float pipeline by a set of metrics:

**Primary (the operational gate):**
- **Pd** — weak-target detection probability; must drop by ≤ 2 percentage points vs float.
- **Pfa** — false-alarm rate; must stay ≤ 2× the float baseline at the same CFAR threshold.

**Supporting:**
- **SNR loss** — drop in the weak target's output SNR from quantization; the per-stage bit budget.
- **SFDR** — spurious-free dynamic range (worst quantization spur vs the dominant signal).
- **ADC saturation** — pass/fail: the strong return must stay within the ADC full-scale.

## 2. Implementation

The front end is quantized at four boundaries as following, everything else (the FFT butterflies,
the CFAR sliding-window arithmetic, the whole back end) computes in float.

```
generate_frame_mimo            ← ideal "analog" truth, UNCHANGED (no fp here)
   │   complex (n_virtual, n_chirps, n_range)
   ▼ [Q_adc]   digitizer: complex I/Q → adc_bits, FIXED scale ±adc_full_scale (clips)
   │
   ├─ per virtual channel: range_doppler(fp)   (float64 inside)
   │     DC-removal · Hann · range FFT · keep n_range//2
   │        [Q_cube]  range-FFT output → cube_bits        ← radar-cube storage word
   │     clutter-removal · Hann · Doppler FFT · fftshift
   │        [Q_rd]    Doppler-FFT output → rd_bits
   │
   ▼ integrate(fp):  Σ_v |rd_v|²
   │     each |rd_v|² formed EXACTLY from the Q_rd cube (squaring finite I/Q is
   │     lossless — no separate power word)
   │     [Q_acc]  running sum  → acc_bits   (the integrated power map = CFAR input)
   │
   ▼ CFAR  (float compute on the quantized integer power map; output = boolean mask)
   │
   ▼ detection list → DoA · clustering · CTRV-EKF tracking     ← FLOAT, untouched
```

| Knob | Boundary | Hardware analog |
|------|----------|-----------------|
| `adc_bits` (+ `adc_full_scale`) | `Q_adc` (digitizer) | 12–16-bit ADC |
| `cube_bits` | range-FFT output | 16-bit I/Q radar cube (SRAM-dominant) |
| `rd_bits` | Doppler-FFT output | derived (≈ cube word + FFT bit growth) |
| `acc_bits` | NCI accumulator (summed power) | 24–40-bit accumulator |

The ADC uses a **fixed** full-scale, sized to contain the strong return (an
over-range input would clip, but here the strong return sits at full-scale and
does not — so saturation is a checked guard, not a triggered effect); the internal
stages default to **per-frame block scaling**
(a block-floating-point exponent at frame granularity, applied at the boundary).
CFAR consumes the quantized integer power map and emits a boolean mask, so it is
fixed-point at its input boundary but not quantized internally.

Fixed-point has two failure modes, split here by boundary. **Overflow** at the
MSB (Most Significant Bit) end is bounded by construction — the ADC's fixed
full-scale, per-stage block scaling, and the accumulator's bit-growth headroom
(§3) — so it never triggers in this scene. What remains free to vary is
**quantization / underflow** at the LSB (Least Significant Bit) end, the weak
target dropping below the LSB, which is what the §3 sweeps study.

## 3. Per-stage evaluation

**Goal.** Determine each swept stage's *knee* — the fewest bits at which the
front end still matches the float pipeline — and identify which metric binds each
stage. All four boundaries are swept (`adc`, `cube`, `rd`, `acc`). Each swept stage is run across a bit range while the others are held wide
(≈float), so every curve isolates that one stage's cost; statistics are
Monte-Carlo over `element_noise_seed`. The dashed line in each figure is the
float-golden value. All four
figures are produced by one run:

CMD: `python -m evaluation.fixedpoint_sweep`

![Weak-target Pd vs bits, per stage](fixedpoint_pd_vs_bits.png)

**Pd is the binding metric.** Pd is the weak-target detection rate on the
dynamic-range scene, gated per frame at the moving target's cell. The **ADC** stage
shows a clean Pd knee (~8 bits) — its fixed full-scale means too few bits push the weak target
below the LSB. The block-scaled internal stages behave differently from one another:
`rd` recovers by ~10 bits, while `cube` and `acc` carry their own binding limits: The cube Pd curve is non-monotonic (8 bits scores worse than 6), and the acc Pd curve recovers only at ~24–28 bits — roughly double every other stage. The `acc` knee is bit-growth arithmetic (below); the cube dip is spur flooding (§3.1).

**Why `acc` needs ~24 bits.** The accumulator width is the integer bit-growth of
`Σ_v (I_v² + Q_v²)`. Starting from `I, Q` at `rd_bits` each: squaring doubles the
word, so `I_v² + Q_v²` is `~2·rd_bits` wide; summing `N_v = n_tx·n_rx = 12` channels
adds `ceil(log2 N_v) = 4`. So `2·rd_bits + 4 ≈ 2·10 + 4 = 24` bits — the width that
carries the sum *losslessly*. (Equivalently in dB: a linear *power* word buys only
~3 dB/bit versus ~6 dB/bit for amplitude, so the scene's 61.6 dB dynamic range (the
strong/weak amplitude ratio, `20·log10(0.3/0.00025)`) costs ~20 bits —
twice the voltage stages — even though squaring leaves the dB range itself unchanged;
the `+4` is the accumulation headroom.) Since `rd_bits ≈ 10` already sits the weak
target one LSB above zero, *lossless* means *the weak target survives*, so this is
exactly where Pd recovers — 0.98 at 24 bits, full at 28 — while below it the weak
target's power falls under the LSB and Pd stalls at 0.63–0.69 (16–20 bits).

![False-alarm rate vs bits, per stage](fixedpoint_pfa_vs_bits.png)

**Pfa is a bit-robust sanity check.** Pfa is the classical false-alarm rate,
measured on a **noise-only companion** (same config and seeds, targets removed). It
holds at the **designed constant rate** across *all* bit widths — CFAR sets its
threshold from the local noise, so quantizing the noise doesn't move it. A flat
curve binds no stage, so the bit-dependent false-alarm cost lives in spur flooding
instead (§3.1).

![SNR loss vs bits, per stage](fixedpoint_snrloss_vs_bits.png)

SNR loss is how far the weak target's output SNR falls below the float pipeline —
the dB "budget" view of the detection cost. It **mirrors Pd**, and the knees agree:
`adc` by ~8 bits, `cube` by ~10, `acc` only by ~24–28. One wrinkle: the `acc`
SNR-loss curve is **non-monotonic** (0.0 dB at 20 bits → 1.3 at 24 → ~0 at 28) — the
**0.0 reading at 20 bits is a noise-floor-estimator artifact**, not a genuine recovery.
The metric is `Loss = SNR_float − SNR_fp` (clamped at 0), and each SNR is the
weak-target *peak* minus the **median off-DC power** (the noise-floor estimate);
`SNR_float ≈ 7 dB` (measured on the float summed-power map: weak-cell peak +11 dB,
median off-DC floor +4 dB). At `acc = 20` bits **two things break
together**: (i) the noise floor collapses — ≈99% of off-target cells quantize to zero
(measured 98.7% → the −120 dB log-floor) — so the median noise estimate is garbage;
and (ii) the weak target sits right at the accumulator LSB (effective LSB = 31.9,
weak-target = 13.2 → weak / LSB = 0.41, derivation as below), so per frame it is a
**coin-flip** — its peak is bimodal: alive in ~50% of frames (measured 49%; a survivor
rounds up to ≥1 LSB, ≈ +15 dB) and dead (−120 dB) in the rest, nothing in between.
Averaging `peak − median` (a dB quantity) across that
collapsed, bimodal regime yields a mean far above `SNR_float`, so the loss clamps to
0 — even though the exact weak cell survives only ~half the frames and gated Pd is just
0.69. By 24 bits the floor is back (measured 4.5% dead, median +6 dB) and the peak is
intact every frame, so `SNR_fp ≈ 5.6 dB` (peak +11.2 − floor +5.6, both measured) gives
the genuine ≈1.3 dB loss (matching the figure) and Pd jumps to 0.98.
**Pd — a per-frame detection count, not an averaged dB — is the truth** (monotonic
0.63 → 0.69 → 0.98 → 0.99).

*Derivation of those two numbers.* The accumulator block-scales to the strong-target
peak, so its quantization step (from `fixed_point.quantize`) is
`LSB = block_max / 2^(acc_bits − 1)`. The strong return sets `block_max ≈ 1.67×10⁷`
(the measured peak of the fixed-point summed-power map) and `acc_bits = 20`, giving
`LSB = 1.67×10⁷ / 2¹⁹ ≈ 31.9` in linear power — matching the measured smallest nonzero
output quantum exactly. The weak-target level is its measured float power, ≈ `13.2`
(= +11.2 dB, the +11 above). Hence `weak / LSB ≈ 13.2 / 31.9 ≈ 0.41` — below the
½-LSB round-up threshold, which is
why survival is a per-frame coin-flip.

![SFDR vs bits, per stage](fixedpoint_sfdr_vs_bits.png)

**SFDR** is the dominant (strong) return's peak minus the largest spur outside the
target gates. **The flat ~27.5 dB floor is the strong target's own range leakage**,
confirmed two ways rather than asserted: (i) the limiting spur is spatially *locked to
the strong target* — same Doppler row, exactly ±2 range bins from the peak — and tracks
it as it walks in range frame-to-frame, so it is not noise, not the weak target, and not
a bit-dependent quantization spur; (ii) simulating the strong target *alone* reproduces
the same −27 dBc skirt at +2 bins, so the floor is the strong return's own range
response. That −27 dBc sits ~4 dB *above* the ideal Hann first sidelobe (−31.5 dBc): the
target is off-bin and the pre-FFT DC/clutter high-pass lifts the near skirt, so the level
is windowing **plus** off-bin straddle **plus** filtering — genuine range leakage. The spur
shows at ±2 bins, not ±1: the ±1 skirt is stronger (≈ −4 dBc) but falls inside the ±1
exclusion gate and is counted as the target. Because this floor is set by the signal, not the
quantizer, SFDR is **bit-independent** across *every* swept stage:
quantization spurs stay below it across the usable bit range, so adding bits does not move it.

Taken together, the four figures say Pd is what binds, Pfa is bit-robust, SNR
loss mirrors Pd, and SFDR is leakage-limited. The one remaining oddity the metrics don't explain — the
**cube** Pd curve scoring *worse* at 8 bits than 6 — is run down next.

### 3.1 Strong-return spur flooding (the cube-stage bit-driver)

The cube Pd curve dips from 6 to 8 bits — scoring *worse* with more bits. That looks
backwards, but no real detection is lost: at 6 bits the weak target is already buried,
and its "Pd" is entirely spurious padding. At 6 bits the coarsest quantization (largest
step → tallest signal-correlated spurs) generates the most strong-return spurs, some of
which land in the weak-target gate and inflate its "Pd"; going to 8 bits strips those
spurs away, so the padded count falls before the weak target is genuinely detected
(~10 bits) and Pd recovers for real. The 6→8 dip is that padding being removed, not
detections being lost. Measuring the false-alarm rate directly confirms it.

The false-alarm cost that *does* vary with bit width comes from the **signal**, not
the noise: the strong return's quantization spurs cross the CFAR threshold and flood
the map with spurious detections. It exists only with the strong return present, so it is
measured as the **spurious-detection rate** in target-free cells of the
dynamic-range scene (excluding a guard around both moving targets) — a distinct
quantity from the classical (noise-only) Pfa above. Sweeping `cube_bits` with the
other stages held wide (Monte-Carlo over seeds) gives the spurious-detection rate:

| `cube_bits` | spurious-detection rate | vs float |
|---|---|---|
| float | 2.3e-5 | 1× |
| 6 | 8.6e-2 | ~3800× |
| 8 | 5.5e-3 | ~240× |
| 10 | 2.0e-5 | ~1× |
| 12 / 16 | ~2e-5 | ~1× |

Reproduce: `strong_return_spur_rate(cfg, targets, fp)` in
`evaluation/fixedpoint_sweep.py`.

So the **cube** word length is bound at **≥ 10 bits by spur flooding**, even though
its Pd knee alone would allow fewer. Note that **SFDR does not surface this**: SFDR
(peak spur vs the strong target) stays ~27 dB while flooding, because what matters
operationally is the spur level vs the *detection threshold* (~60 dB below the
strong target), not vs the strong target — so the spurious-detection *rate* is the
right measure, not the peak-spur ratio.

### 3.2 Bit widths chosen

Reading each stage's knee off the sweeps — the fewest bits that meet the criterion.
The §1 Pfa gate is bit-robust (satisfied everywhere), so the operative gates are
**Pd** and the **spurious-detection rate** (§3.1) — `cube` is set by the latter.

| stage | bits | bound by |
|---|---|---|
| `adc` | ~8 | Pd (weak-target loss) |
| `cube` | ~10 | spurious-detection rate (spur flooding, §3.1) |
| `rd` | ~10 | Pd |
| `acc` | ~24–28 | Pd (bit-growth: 2·rd_bits + accumulation headroom) |

The **accumulator** dominates the cost — it is the one place *power* is quantized,
so it pays both the ~2× power-domain penalty and the ~4-bit accumulation headroom,
needing roughly 2.5–3× the bits of the voltage stages (`cube`/`rd`, ~10) — while the
**cube** word is set by false alarms, not detection. The voltage stages (`adc`,
`cube`, `rd`) all knee at ~8–10 bits.

## 4. Memory benefit (analytic)

The radar cube for this scenario is `n_virtual × n_chirps × (n_range/2)` =
12 × 64 × 128 = 98 304 complex samples. Storage at the swept `cube_bits` vs the
float prototype (computed by `processing.fixed_point.cube_bytes`, 2 components/sample):

| Cube word (per I/Q component) | Cube size | Saving vs float64 | Saving vs float32 |
|---|---|---|---|
| 64-bit (complex128 prototype) | 1536 KiB | 1× | — |
| 32-bit (complex64) | 768 KiB | 2× | 1× |
| 16-bit | 384 KiB | 4× | 2× |
| 12-bit | 288 KiB | 5.3× | 2.7× |
| 10-bit | 240 KiB | 6.4× | 3.2× |

This is an **analytic word-length → storage** translation, it quantifies the
radar-cube SRAM saving the chosen word lengths imply.

## 5. Scope & limitations

The methodology (fixed/float split, quantize-in-float, sweep-against-golden,
Pd-primary knee) is standard first-pass practice, but boundary quantization
makes the recommended bit widths **lower bounds** and the savings **upper bounds**,
and a few further choices bound the scope:

1. **The scaling policy is assumed, not compared.** All sweeps use one number-format
   choice — fixed-scale ADC plus per-frame BFP on the internal stages
   (`scaling='block'`). The code also supports a static global scale
   (`scaling='fixed'`, plain Q-format), but it is not compared, so the recommended
   widths are **conditional on BFP**; whether a static Q-format or BFP better fits
   this pipeline is a question this study hasn't answered yet.
2. **The FFT is computed in float, not fixed-point.** A bit-true fixed-point FFT is
   a substantially larger, more involved implementation — per-stage bit growth and
   rescaling, quantized twiddle factors, quantized window coefficients (a separable
   pre-FFT multiply, minor effect) — so for this first-pass study it is deliberately
   left in float and deferred to a bit-true sign-off. Consequence: the
   `cube_bits`/`rd_bits` knees assume ideal FFT arithmetic and are optimistic by
   perhaps a bit or two.
3. **The signal model is clean** (no PA nonlinearity / phase noise), so the only
   added spur *sources* are quantization — but the **SFDR floor here is set by the
   strong target's own range leakage**, not quantization (§3). So SFDR mainly
   confirms that quantization stays below that leakage floor; read it that way, not
   as an absolute spur level.

## 6. Artifact index

| Artifact | Source |
|---|---|
| `fixedpoint_pd_vs_bits.png` | `evaluation/fixedpoint_sweep.py` (`main`) |
| `fixedpoint_pfa_vs_bits.png` | `evaluation/fixedpoint_sweep.py` (`main`) |
| `fixedpoint_snrloss_vs_bits.png` | `evaluation/fixedpoint_sweep.py` (`main`) |
| `fixedpoint_sfdr_vs_bits.png` | `evaluation/fixedpoint_sweep.py` (`main`) |
| `scenarios/scenario_mimo_fixedpoint_dynrange.json` | the dynamic-range scene |
| fixed-point mode | `processing/fixed_point.py`, `FixedPointConfig`, `fp=` on `range_doppler`/`integrate`/`process_mimo_frame` |
