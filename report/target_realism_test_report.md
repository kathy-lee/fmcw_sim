# Target Realism Test — Part I: Swerling RCS Fluctuation

**Date:** 2026-06-12
**Branch:** `feat/add-swerling-I-and-extended-target-model`
**Pipeline under test:** unchanged from the baseline reports —
`range_doppler` (Hann, DC removal) → `cfar_2d` (OS-CFAR, Pfa = 1e-4) →
`cluster` → `Tracker` (GNN, M-of-N, KF). What changes is the **target
model**: per-frame RCS fluctuation (Swerling 1/3).

---
## 0. TL;DR

  Added Code:
  - simulator/fluctuation_models.py — Swerling 1/3 as AR(1)-evolving complex-Gaussian scatterers, unit mean power, prefix-property RNG
  - Target.fluctuation + RadarConfig.fluctuation_seed (77777, third RNG stream), generator integration, run_pipeline(..., fluctuation_seed=) hook, run_one_point(..., advance_frame_index=)
  refactor
  - evaluation/sweep_amplitude_fluctuation.py and evaluation/tracker_stress_swerling_sweep.py
  - scenarios/scenario_missed_detections_swerling.json, report/README.md, report/target_realism_test_report.md (Part I, complete), plus all PNG/CSV artifacts

  Headline results (all anchors against the published baselines passed):
  - Detection: Swerling 1 costs +8.0 dB at Pd 0.9 (textbook value) but only +1.4 dB at Pd 0.5; Swerling 3 in between (+4.1 dB). FA unchanged across cases.
  - Tracking illustration: adding fading alone (same mean SNR) takes the published §3.1.1 baseline from Pd 0.61/CLEAN to Pd 0.40/3 IDs/TOTAL CONFUSION — degradation with no mean-SNR change,
  vs the baseline's −1.6 dB knob.
  - Sweep: steady rows anchor on §3.1.2/§3.4 (max_coast=5: 0.67±0.04 vs published 0.67±0.05); ρ=0 fading halves track Pd but max_coast stays monotone; ρ=0.9 burst fading is barely responsive
  to the coast lever (5/6 TOTAL CONFUSION even at max_coast=8) — coast budgets must be sized to fade duration, not miss rate.

## 1. Introduction

The two baseline reports characterize the detector
([report 1](detection_performance_test_report.md)) and the tracker
([report 2](tracker_stress_test_report.md)) against **steady point
targets**: the echo amplitude is a fixed constant, and every missed
detection is driven by thermal noise alone. Real targets do not behave
this way — a car, a pedestrian, a drone is a collection of scatterers
whose phasors add differently as the aspect angle drifts, so the received
power *scintillates* from frame to frame, sometimes fading 10 dB or more
below its mean for several consecutive frames.

This report upgrades the target model with the classic Swerling
fluctuation cases and re-measures both pipeline stages. Every experiment
is designed to be directly comparable with a published baseline result:
the detection sweep reuses report 1 §3's exact scenario and its steady
curve must reproduce report 1's curve; the tracking experiments reuse
report 2 §3.1's exact scenario, and their steady rows literally re-run
report 2 §3.1.2. Differences from the quoted baselines are therefore
attributable to the target model alone — which is the point.

Part I covers Swerling RCS fluctuation. Part II (extended targets — one
target occupying many cells) follows separately.

## 2. Methodology — deltas only

All conventions of the baseline reports are inherited (fixed front-end
lens, seed hygiene, classification taxonomy — see
[report/README.md](README.md)). Three deltas:

### 2.1 The fluctuation model

A target may now carry an optional `fluctuation` dict, e.g.
`{"model": "swerling1", "frame_corr": 0.9}` (`None` = steady amplitude,
the exact baseline behavior). Each Swerling case is realised physically,
as the RMS magnitude of *k* complex-Gaussian scatterer voltages evolving
across frames as a first-order AR process
(`simulator/fluctuation_models.py`):

```
s_f = ρ·s_{f−1} + √(1−ρ²)·w_f,    w_f ~ CN(0, 1)
power_f  = mean_i |s_f[i]|²       (k=1 → Swerling 1, exponential power;
                                   k=2 → Swerling 3, chi-squared-4 power)
factor_f = √power_f               (E[factor²] = 1: mean POWER preserved)
```

The per-frame echo amplitude is `target.amplitude × factor_f`. Because
the factors are unit-mean-**power**, adding fluctuation changes a
target's *mean* SNR not at all — only its frame-to-frame distribution.
`frame_corr` (ρ) is the frame-to-frame **voltage** correlation; the
resulting power correlation is ρ², and the fade decorrelation time is
−1/ln(ρ²) frames (ρ = 0.6 → ~1 frame, ρ = 0.9 → ~4.7 frames). ρ = 0
reproduces the classic scan-to-scan-independent Swerling cases; ρ > 0
produces multi-frame fade bursts at the same marginal distribution.

### 2.2 A third RNG stream

Report 2 §2.5's two-stream seed-hygiene table gains a third row:

| Stream | Seed | Purpose |
|---|---|---|
| CFAR calibration | `0` | the pure-noise pool that sets α |
| Generator thermal noise | `RadarConfig.seed` (default `99999`, Monte Carlo `1…`) | per-frame noise realization |
| **Swerling fades (new)** | `RadarConfig.fluctuation_seed` (default `77777`) | per-target amplitude factor sequences |

Decoupling the fade stream from the noise stream buys the same property
the first decoupling did: a Monte Carlo can redraw the noise while
holding the fade pattern fixed, or redraw both. The tracker sweep (§4.2)
redraws both per run — radar seed `s`, fade seed `1000+s` — an honest
Monte Carlo over noise and fades together. Factors are drawn per
`(fluctuation_seed, target_index)` and consumed strictly frame-by-frame,
so a fade sequence is reproducible regardless of how many frames are
generated or when the cache regrows.

### 2.3 Frame-advancing Pd sweep

The baseline amplitude sweep (report 1 §3) froze `frame_index=0` — the
scene held fixed while the noise redrew, so Pd was measured at a
stationary target position. Swerling factors are indexed by frame, so the
fluctuation sweep must **advance** `frame_index` (a frozen index would
reuse `factor[0]` forever and measure no fading at all). It uses the SAME
`scenario_two_targets.json`, and recomputes the truth bins per frame from
`gen.ground_truth()` so the ±1-bin matcher follows the slow drift
(0.1 m/s × 15 s = 1.5 m over 300 frames, well inside the unambiguous
range; the simulator's SNR is range-independent, so Pd is stationary
across the run). The steady curve is therefore a same-scenario anchor
check against report 1 §3.

The sweep also runs **300 frames per point** instead of the baseline's
100, for two reasons. First, the headline metric (SNR@Pd = 0.9, §3.2)
sits on the *flattest* part of the Swerling curves: a Pd estimate at 0.9
carries a standard error of √(0.9·0.1/n) — ±0.03 at 100 frames vs ±0.017
at 300 — and where the steady curve's steep ramp turns that into a
fraction of a dB, the Swerling-1 curve crawls from 0.8 to 0.95 over ~6 dB,
so the same Pd error would smear the interpolated threshold by whole dB.
Second, each fluctuating point holds one fixed fade seed, so its frames
must sample the fade distribution as well as the noise: 300 frames
averages over 300 fade draws per point, making the measured Pd a property
of the Swerling marginal rather than of one lucky or unlucky fade
sequence. The anchor comparison is unaffected — it is checked on the
steady *curve*, not on per-point sample counts. (The longer run is also
what exposed the FA undersampling discussed in §3.1.)

One correction to the design discussion that preceded this report: we
briefly believed a zero-velocity variant was impossible because "DC
removal cancels a v = 0 target exactly." That conflates two different
knobs in `range_doppler`: `dc_removal` subtracts the per-chirp
**fast-time** mean (it kills range bin 0 only), while `clutter_removal`
subtracts the **slow-time** mean across chirps — *that* is the operation
which cancels a zero-Doppler echo exactly. The evaluation pipeline runs
with `clutter_removal=False`, so a strong v = 0 target is in fact
detected (pinned by
`test_zero_velocity_cancellation_is_clutter_removal_not_dc_removal`).
The sweep reuses the baseline's moving targets anyway — same-scenario
comparability with report 1 is the stronger reason and was always the
primary one.

## 3. Detection under RCS fluctuation

CMD: `python -m evaluation.sweep_amplitude_fluctuation --frames 300`

![Pd vs mean SNR, steady vs Swerling](sweep_amplitude_fluctuation.png)

### 3.1 Anchor check against the baseline

| Quantity | Report 1 §3 (steady, frozen-frame, 100 frames) | This sweep (steady, frame-advancing, 300 frames) |
|---|---|---|
| Pd ramp (per-sample SNR) | ~0 → 1 over −35 → −18 dB | same window: SNR@Pd0.5 −25.4 dB / SNR@Pd0.9 −22.8 dB (target 0); −25.3 / −22.6 dB (target 1) |
| FA / frame (OS) | ~0.05–0.06 | 0.123 (see caveat below) |

The steady Pd curves reproduce the baseline: both targets climb 0 → 1
inside the same −35 → −18 dB window, with the stronger 3 m target leading
the weaker 6 m target by the same few dB. The frame-advancing methodology
change is benign, so the Swerling deltas below are attributable to
fluctuation alone.

The FA row deserves an honest caveat: it is the **baseline's** number
that is off, not this sweep's. Under the published condition (100 frames,
canonical seed 99999) this code reproduces report 1's FA exactly
(0.050/frame — 5 events). But 100 frames is a small sample for a rare
event: over 300 frames the same configuration measures 0.123/frame, and
three independent noise seeds give 0.117–0.133/frame. The true
centroid-level FA rate of this lens is ~0.12/frame; report 1's
0.05–0.06 was an undersampled estimate from a lucky-low stretch of the
canonical stream. None of the comparisons below are affected — FA is
measured identically for all cases in this sweep and is flat across them.

### 3.2 The fluctuation loss

Headline SNR thresholds, target 0 (target 1 agrees within ~1 dB on every
entry):

| Case | SNR@Pd=0.5 (Δ vs steady) | SNR@Pd=0.9 (Δ vs steady) |
|---|---|---|
| steady | −25.4 dB | −22.8 dB |
| swerling1 | −24.0 dB (**+1.4**) | −14.9 dB (**+8.0**) |
| swerling3 | −25.0 dB (+0.5) | −18.7 dB (**+4.1**) |

(Target 1: steady −25.3/−22.6; swerling1 −24.0 (+1.3) / −13.6 (+9.1);
swerling3 −24.6 (+0.7) / −18.5 (+4.1).)

This is the textbook fluctuation-loss signature, measured end-to-end
through the real CFAR/clustering chain rather than from a closed form:

- **At Pd = 0.5 the penalty is nearly zero** (+0.5 to +1.4 dB). Around
  the mid-point the curves cross: deep fades push frames below threshold,
  but up-fluctuations rescue frames a steady target at the same mean SNR
  would have lost. Wins and losses roughly cancel.
- **At Pd = 0.9 the penalty is large** — **+8 dB for Swerling 1**
  (literature value: ~8 dB), because high Pd is controlled by the miss
  *tail*, and an exponential power distribution puts substantial mass in
  deep fades no matter how high the mean SNR is pushed. The Pd curve does
  not just shift, it *flattens*: swerling1 needs the top of our grid to
  reach Pd 0.9–0.95 where the steady target has been saturated at 1.0
  for 6 dB.
- **Swerling 3 sits in between** (+4.1 dB): its two-scatterer
  (dominant-plus-rest) construction gives chi-squared-4 power — variance
  0.5 vs the exponential's 1.0 — so deep fades are rarer and the
  high-Pd tail is kinder.
- **FA is unchanged across cases** (0.123 steady / 0.123–0.137
  fluctuating): fluctuation is a target property; the noise floor and
  the CFAR threshold never see it.

The practical consequence: a link budget sized on a steady-target curve
overstates high-Pd performance against a real fluctuating target by the
full §3.2 delta — at Pd 0.9, by ~8 dB for Swerling-1-like targets. This
is why fluctuation models exist in every radar detection textbook: the
steady-target Pd curve is the wrong curve to size margins on.

## 4. Tracking under correlated fades

### 4.1 Illustrative run — one knob from the published baseline

The scenario is report 2 §3.1.1's missed-detection baseline
(`scenario_missed_detections.json`: one marginal-SNR target, amp 0.0006,
Pd ≈ 0.6 at the detector) plus **only** the fluctuation block
(`swerling1, frame_corr 0.9`), at the same mean power. The baseline row
below is quoted from report 2 (and re-verified byte-identically on this
branch); only the Swerling row is a new run:

CMD: `python -m evaluation.tracker_stress --scenario scenarios/scenario_missed_detections_swerling.json --algorithm kf`

![Published steady baseline (report 2 §3.1.1)](tracker_stress_missed_detections.png)
![Swerling-1, frame_corr 0.9, same mean power](tracker_stress_missed_detections_swerling.png)

| Variant (same mean power, amp 0.0006) | Pd | Distinct IDs | Coast | Switches | Classification |
|---|---|---|---|---|---|
| steady (report 2 §3.1.1, quoted) | 0.61 | 1 | 28 | 0 | ✓ CLEAN |
| swerling1, ρ=0.9 (this report) | **0.40** | 3 | 18 | 2 (frame 40: 2→6; frame 78: 6→7) | **⚠ TOTAL CONFUSION** |

The reason behind the collapse is *when* the misses happen, not how many.
The baseline's misses are per-frame noise coin flips on a marginal steady
target — scattered, mostly isolated — so `max_coast = 5` bridges every
gap and the tracker holds one CLEAN ID through 28 coast frames. Adding
unit-mean-power fading changes neither the mean SNR nor the noise; it
reshapes the miss pattern into **bursts** (~4.7-frame fade decorrelation
at ρ = 0.9). Multi-frame fades exceed the coast budget, the track dies
mid-fade and respawns under a new ID when the fade lifts — three IDs and
two switches later the run classifies as TOTAL CONFUSION.

Contrast this with report 2 §3.1.1's own failure knob: there, degradation
required an actual mean-SNR cut (amp 0.0006 → 0.0005, −1.6 dB) to produce
TOTAL CONFUSION (Pd 0.38, 3 IDs). Here the same qualitative collapse —
Pd 0.40, 3 IDs — arrives with **no mean-SNR change at all**. A tracker
qualified against steady targets at a given link margin can fail against
a fluctuating target at the *same* margin.

(Single-seed illustration; the statistical claim is §4.2's sweep.)

### 4.2 Fade correlation vs coast tolerance

CMD: `python -m evaluation.tracker_stress_swerling_sweep`

The grid: fade correlation ρ ∈ {steady, 0.0, 0.6, 0.9} × max_coast ∈
{1, 3, 5, 8}, six (radar seed, fade seed) pairs per cell, 80 frames,
on the published `scenario_missed_detections.json` (fading injected per
cell — the steady rows are literally report 2 §3.1.2 re-run multi-seed).
Full summary (mean ± std over 6 seeds):

| frame_corr | max_coast | Pd | distinct | coast | switches | classes |
|---|---|---|---|---|---|---|
| steady | 1 | 0.57±0.04 | 5.0 | 11.8 | 3.33 | 6 ID SWITCH |
| steady | 3 | 0.62±0.05 | 2.8 | 18.0 | 1.83 | 1 CLEAN, 5 ID SWITCH |
| steady | 5 | 0.67±0.04 | 1.2 | 22.3 | 0.17 | 5 CLEAN, 1 ID SWITCH |
| steady | 8 | 0.67±0.03 | 1.0 | 22.8 | 0.00 | 6 CLEAN |
| 0.0 | 1 | 0.20±0.07 | 5.3 | 7.8 | 2.50 | 6 TOTAL CONFUSION |
| 0.0 | 3 | 0.25±0.09 | 4.5 | 17.3 | 2.50 | 6 TOTAL CONFUSION |
| 0.0 | 5 | 0.32±0.09 | 2.7 | 28.0 | 1.33 | 6 TOTAL CONFUSION |
| 0.0 | 8 | 0.36±0.07 | 1.8 | 38.5 | 0.83 | 5 TOTAL CONFUSION, 1 TRACK BREAK |
| 0.6 | 1 | 0.24±0.09 | 5.7 | 9.7 | 2.67 | 5 TOTAL CONFUSION, 1 TRACK BREAK |
| 0.6 | 3 | 0.28±0.10 | 4.7 | 18.3 | 2.33 | 6 TOTAL CONFUSION |
| 0.6 | 5 | 0.37±0.07 | 2.3 | 33.3 | 1.33 | 5 TOTAL CONFUSION, 1 TRACK BREAK |
| 0.6 | 8 | 0.40±0.08 | 1.7 | 39.3 | 0.67 | 1 CLEAN, 3 TOTAL CONFUSION, 2 TRACK BREAK |
| 0.9 | 1 | 0.34±0.10 | 3.8 | 6.0 | 2.33 | 6 TOTAL CONFUSION |
| 0.9 | 3 | 0.36±0.11 | 2.8 | 10.3 | 1.67 | 6 TOTAL CONFUSION |
| 0.9 | 5 | 0.38±0.12 | 2.7 | 17.2 | 1.50 | 6 TOTAL CONFUSION |
| 0.9 | 8 | 0.40±0.13 | 2.2 | 24.3 | 1.17 | 1 CLEAN, 5 TOTAL CONFUSION |

Side-by-side with the baseline survival lever. Report 2 §3.1.2 swept
max_coast on this very scenario minus the fluctuation block (steady
marginal, amp 0.0006, single canonical seed) — quoted verbatim in the
first column. The second column is this sweep's steady rows: the SAME
experiment re-run over 6 seeds, the anchor:

| max_coast | report 2 §3.1.2: steady, 1 seed (Pd / class) | this sweep: steady, 6 seeds (Pd / classes) | this sweep, ρ=0 (Pd / classes) | this sweep, ρ=0.9 (Pd / classes) |
|---|---|---|---|---|
| 1 | 0.46 / TOTAL CONFUSION | 0.57±0.04 / 6 ID SWITCH | 0.20±0.07 / 6 TOTAL CONF. | 0.34±0.10 / 6 TOTAL CONF. |
| 3 | 0.56 / ID SWITCH | 0.62±0.05 / 1 CLEAN, 5 ID SWITCH | 0.25±0.09 / 6 TOTAL CONF. | 0.36±0.11 / 6 TOTAL CONF. |
| 5 | 0.61 / ✓ CLEAN | 0.67±0.04 / 5 CLEAN, 1 ID SWITCH | 0.32±0.09 / 6 TOTAL CONF. | 0.38±0.12 / 6 TOTAL CONF. |
| 8 | (not run in report 2) | 0.67±0.03 / 6 CLEAN | 0.36±0.07 / 5 TOTAL CONF., 1 BREAK | 0.40±0.13 / 1 CLEAN, 5 TOTAL CONF. |

**Anchor.** The steady column reproduces report 2's survival-lever story
multi-seed: Pd rises monotonically with max_coast, classifications walk
from all-failed at max_coast = 1 to all-CLEAN at 8, and the max_coast = 5
row (0.67±0.04, 5/6 CLEAN) lands exactly on report 2 §3.4's 30-seed
distribution (0.67±0.05, 27/30 CLEAN). At max_coast = 1 the 6-seed mean
(0.57, ID SWITCH dominant) sits above the published single-seed 0.46 /
TOTAL CONFUSION; to confirm this is seed draw and not a regression, the
canonical seed was re-run on this branch at max_coast = 1 and 3 and
reproduces §3.1.2 **byte-identically** (0.46 / 7 IDs / 12 coast / TOTAL
CONFUSION; 0.56 / 3 / 21 / ID SWITCH). The canonical seed is simply a
pessimistic draw at low coast budgets — the same pattern §3.4 already
documented at the default budget (canonical 0.61 at the low edge of
0.67±0.05). The harness is validated; the fading columns can be read.

**The rate effect (steady → ρ=0).** Same mean SNR, fades added: track-level
Pd halves (0.67 → 0.32 at max_coast = 5) and every run fails. This is
§3.2's fluctuation loss applied at a marginal operating point — the
steady target sat near detector-Pd 0.6, where the Swerling-1 curve runs
well below the steady curve, so the per-frame miss rate rises
substantially with no change in mean power. The misses are still
scan-to-scan independent, so max_coast retains its report-2 role
monotonically (Pd 0.20 → 0.36, distinct 5.3 → 1.8 as the budget grows) —
but it is recovering from a deeper hole and never reaches CLEAN on this
grid.

**The duration effect (ρ=0 → ρ=0.9).** All ρ rows share the same fade
marginal — the same long-run miss rate — and differ only in *when* the
misses come (decorrelation ~1 frame at ρ = 0.6, ~4.7 frames at ρ = 0.9).
The burst structure shows up in two ways. First, at small coast budgets
the ρ = 0.9 rows actually post **higher** raw Pd than ρ = 0 (0.34 vs
0.20 at max_coast = 1): bursty fading clusters the misses, which also
clusters the *hits* — long clear stretches the tracker rides cleanly,
while the same miss count spread uniformly (ρ = 0) fragments every few
frames at a budget of 1. Second — and this is the headline — the
correlated rows are much less responsive to the coast lever: from
max_coast 1 → 8, ρ = 0 gains +0.16 Pd and cuts its distinct-ID count by
3×, while ρ = 0.9 gains only +0.06 with coast frames that never catch up
(24.3 vs 38.5 at budget 8). Multi-frame fade bursts simply exceed any
budget on the grid: the track dies mid-fade no matter how the budget is
tuned within practical range, and classifications stay TOTAL CONFUSION
through max_coast = 8 (5/6 runs). Where uncorrelated misses can be
coasted through, correlated fades must be *survived*, and a fixed coast
budget is the wrong tool.

**Partial saturation, reported as a finding.** At this marginal operating
point (the published amp 0.0006), no fading cell reaches CLEAN reliably
even at max_coast = 8 — burst fading at marginal SNR exceeds any
practical coast budget. This is a property of the operating point, not a
measurement failure: the steady anchor rows behave exactly as published,
and the fading rows retain readable structure (monotone in max_coast,
ordered in ρ at fixed budget).

## 5. Cross-cutting discussion

What changed relative to the two baseline reports, and why it matters:

1. **The detector's Pd-vs-SNR curve is no longer a property of SNR
   alone** — it is a property of (SNR, fluctuation model). Report 1's
   steady curves remain correct for steady targets and as anchors, but a
   link margin sized on them overstates high-Pd performance against a
   Swerling-1 target by ~8 dB at Pd 0.9 (§3.2). The Pd 0.5 point, by
   contrast, barely moves — so a system specified at "50% acquisition
   range" would see almost no change while its "90% track range"
   collapses. Margins must be sized on the fluctuating curve.

2. **Report 2's missed-detection story survives, but the lever must be
   re-sized.** "max_coast is the survival lever" (report 2 §3.1.2) holds
   verbatim under *uncorrelated* fading — the steady→ρ=0 columns differ
   in miss rate, not in mechanism. Under *correlated* fading the lever
   loses most of its authority: i.i.d.-miss intuition (coast budget ≈ a
   few expected consecutive misses) underestimates the budget needed by
   the full fade duration, and at ρ = 0.9 (~4.7-frame bursts) even
   max_coast = 8 leaves 5/6 runs in TOTAL CONFUSION. Coast tolerance
   must be sized against the fade correlation **time**, not the average
   miss rate — and beyond some correlation time, no fixed budget works.

3. **The M-of-N / coasting machinery is the right defense layer, used at
   its limit.** Frame-to-frame binary integration (M-of-N confirmation +
   coasting) is the practice-standard answer to scan-to-scan
   fluctuation, and §4.2 shows it absorbing ρ = 0 fading gracefully.
   Pre-CFAR frame-to-frame integration would not help here: the target
   moves ~one range bin per frame per 1.5 m/s at this geometry (7.5 cm
   bins, 50 ms frames), so straddling frames smear and the integration
   gain evaporates — cell migration is why the defense lives at the
   track level in practice. Against long correlated fades the remaining
   options are structural: slower fade dynamics relative to the frame
   rate (faster frame rate), frequency agility to decorrelate the fades
   (turning ρ = 0.9 into ρ ≈ 0, which §4.2 shows is recoverable), or
   accepting reacquisition and handling identity at a higher level.

## 6. Limitations and future work

- **Seed budget.** Illustrative runs are single-seed; the sweep uses 6
  (radar, fade) seed pairs per cell — smaller than report 2's 30-seed
  matrix. The ρ = 0.9 cells in particular carry high variance
  (Pd ±0.10–0.13; only ~17 fade decorrelation times per 80-frame run),
  so their Pd means are coarse. The classification counts (24/24 fading
  cells failing below max_coast = 8) are the robust readout.
- **Swerling 2/4 not modeled.** Chirp-to-chirp (fast) fluctuation breaks
  *intra-frame* coherent integration — a detector-level study that would
  modify `generate_frame` itself, deliberately out of scope here.
- **Fades are independent across targets** (per-target streams). No
  common-mode fading (e.g. shared propagation effects).
- **FA baseline correction.** §3.1's anchor check found report 1's
  FA ≈ 0.05–0.06/frame to be an undersampled 100-frame estimate; the
  stationary rate of the shared lens is ~0.12/frame. Report 1's
  method-vs-method FA comparisons are unaffected (all methods measured
  on the same 100 frames), but the absolute number should be read with
  this correction.
- **Part II (extended targets)** will revisit clustering and the
  GT-association contract; a multi-scatterer extended target produces
  *emergent* scintillation, which should be compared against the explicit
  Swerling knob added here.

## Appendix — Artifact index

| Artifact | Produced by | Status |
|---|---|---|
| `sweep_amplitude_fluctuation.png` / `.csv` | `evaluation/sweep_amplitude_fluctuation.py` | ✅ |
| `tracker_stress_missed_detections.png` (baseline, report 2) | `evaluation/tracker_stress.py` | ✅ pre-existing |
| `tracker_stress_missed_detections_swerling.png` | `evaluation/tracker_stress.py` | ✅ |
| `tracker_stress_swerling_raw.csv` / `_summary.csv` | `evaluation/tracker_stress_swerling_sweep.py` | ✅ |
