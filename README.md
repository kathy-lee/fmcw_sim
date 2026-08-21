# fmcw-sim

A from-scratch FMCW radar simulation and signal-processing pipeline: it
synthesizes raw beat signals from moving targets and carries them all the way to
tracked objects — for both single-channel (SISO) and TDM-MIMO front ends. It
runs **offline** (batch evaluation, dataset export) and **online** (real-time
ZeroMQ pub/sub), and ships with report studys of failure analysis and evaluation tests.

---

## 1. What it does: the processing pipeline

The core of the project is an end-to-end chain from a synthetic FMCW beat-signal
frame to confirmed tracks. Pure NumPy; each stage is an independent, testable
module.

SISO (single channel):

```
generate ─► range-Doppler ─► CFAR ─► clustering ─► tracking
 (frame)      (RD map)      (cells) (detections)   (tracks)
```

MIMO (virtual array):

```
generate ─► range-Doppler ─► NCI ─► CFAR ─► DoA ─► point cloud ─► clustering ─► tracking
 (frame)    (per channel)  (power) (cells)  (az)  (x,y,v,snr)     (objects)    (tracks)
```

The two paths differ after CFAR: SISO clusters detected *cells* directly on the
RD grid, while MIMO first turns each detected cell into Cartesian points (via
per-cell DoA), then clusters the *point cloud*.

| Stage | Module | Entry point | What it does |
|-------|--------|-------------|--------------|
| Beat-signal generation | `simulator/beat_signal_generator.py` | `FMCWBeatSignalGenerator.generate_frame[_mimo]()` | Synthesize complex frames from target trajectories (+ noise, clutter, RCS fluctuation, extent) |
| Range-Doppler | `processing/range_doppler.py` | `range_doppler()` | DC removal → Hann range FFT → clutter (slow-time mean) removal → Doppler FFT |
| Non-coherent integration *(MIMO)* | `processing/point_cloud.py` | `integrate()` | Σ\|RD_v\|² over virtual elements; noise → `Gamma(n_virtual)`, CFAR α recalibrated |
| CFAR detection | `processing/cfar.py` | `cfar_2d()` | CA / OS / GO / SO noise estimation, α empirically calibrated to a design Pfa |
| DoA / angle *(MIMO)* | `processing/doa.py` | `estimate_doa()` | Bartlett / Capon / MUSIC / ESPRIT on the per-cell virtual-array snapshot |
| Clustering | `processing/clustering.py` | `cluster()` | Connected-components / DBSCAN / peak-picking → discrete detections |
| Point cloud *(MIMO)* | `processing/point_cloud.py` | `mimo_point_cloud()` | Per cell: TDM Doppler compensation → DoA → Cartesian `(x, y, v_radial, snr)` |
| Tracking | `processing/tracking.py`, `tracking_cartesian.py`, `imm.py` | `Tracker` / `CartesianTracker` | GNN association, M-of-N confirmation, coasting; KF (CV) / EKF (drag) / IMM |


### Features

- **SISO and MIMO.** SISO tracks in the 1-D range-Doppler plane; TDM-MIMO
  (12-element virtual ULA = 3 TX × 4 RX in default) adds non-coherent integration, per-cell
  angle estimation, and 2-D Cartesian CTRV-EKF tracking.
- **Extended-target model** (`simulator/extended_target.py`) — targets with
  physical range/azimuth extent (multiple scatterers), not just point returns.
- **Fluctuating-target model** (`simulator/fluctuation_models.py`) — Swerling I/III
  RCS scintillation with AR(1) frame-to-frame correlation.
- **Fixed-point implementation** (`processing/fixed_point.py`) — boundary quantization
  (ADC, range-FFT, Doppler-FFT, magnitude, accumulator) with configurable bit
  widths, rounding, and overflow, threaded through the whole front end.

---

## 2. Streaming / online pipeline

The pipeline can run live over **ZeroMQ PUB/SUB**: a publisher emits frames as
they are generated; one or more subscribers connect and run detection/tracking
in real time.

**Architecture.** `streaming_publisher` binds a TCP socket and sends each frame
as a multi-part message `[topic, header(JSON), payload(binary)]`. The wire
format (`simulator/wire.py`) is shape-polymorphic — the header carries `shape`,
`dtype`, per-frame `metadata`, and `ground_truth`, so the same transport serves
both SISO (2-D) and MIMO (3-D) frames. Subscribers deserialize with
`wire.deserialize_frame(header, payload)` and process.

| Subscriber | Module | Pipeline | Output |
|------------|--------|----------|--------|
| RD-only | `streaming_subscriber_RD.py` | range-Doppler | top-k `(range, Doppler, power_dB)` per frame |
| SISO tracked | `streaming_subscriber.py` | RD → CFAR → cluster → track (1-D) | layered RD view + track plot |
| MIMO point cloud | `streaming_subscriber_mimo_pointcloud.py` | RD → NCI → CFAR → DoA → cloud | RD+CFAR / DoA spectra / BEV cloud |
| MIMO tracked | `streaming_subscriber_mimo.py` | + Cartesian clustering & CTRV-EKF | BEV 2-D tracks (id-labelled trails) |

### Usage

Start a publisher (terminal 1), then a subscriber (terminal 2). Default socket
`tcp://127.0.0.1:5555`, topic `fmcw`.

```bash
# SISO: publisher + tracked subscriber
python -m simulator.streaming_publisher --scenario scenarios/scenario_two_targets.json
python -m simulator.streaming_subscriber --connect tcp://127.0.0.1:5555 --topic fmcw --stats-window 20

# MIMO: publisher + tracked subscriber (with optional fixed-point front end)
python -m simulator.streaming_publisher --scenario scenarios/scenario_mimo_turn.json
python -m simulator.streaming_subscriber_mimo --doa music --cfar ca --fixed-point adc=12,cube=14

# MIMO detect-then-angle, no tracking (point cloud + DoA spectra)
python -m simulator.streaming_subscriber_mimo_pointcloud --doa esprit --cfar os
```

**Common flags.** Publisher: `--bind`, `--topic`, `--scenario`, `--frames`
(0 = run forever), `--startup-delay`. Subscribers: `--connect`, `--topic`,
`--max-frames` (0 = infinite), `--stats-window`; MIMO adds
`--doa {bartlett,capon,music,esprit}`, `--cfar {ca,os}`, and `--fixed-point`.

**Live reporting.** Every `--stats-window` frames each subscriber prints
per-stage timings (rd / cfar / cluster / track ms), visualization skip counts,
and overload flags (`over_stream`, `over_realtime`, `drops`, `lag`) so you can
see whether processing keeps up with the frame rate.

---

## 3. Offline radar-cube export

For dataset generation and inspection, the pipeline can simulate a scenario and
export its intermediate **radar cubes** to `.npz`, with quick-look plots. This could be used as input data for modern object detection / classification networks as well. Currently provided the following levels:
`adc` (raw), `range_cube`, `rd_cube`, `power` (integrated RD), and `rad`
(dense range-azimuth-Doppler tensor via Bartlett beamforming). With `--fp` the
saved `adc` is the quantized `Q_adc` frame the downstream cubes derive from, so
each file is internally reproducible.

The levels form a chain that branches at the complex range-Doppler cube; any
subset can be selected with `--levels`:

```text
raw beat frame
   │  adc = raw   (or Q_adc when --fp)
   ▼
 [adc] ── range FFT ──▶ [range_cube] ── Doppler FFT ──▶ [rd_cube]  (complex)
                                                            │
                          ┌─────────────── integrate ───────┤
                          ▼                                  ▼  Bartlett beamform
                       [power]                             [rad]
                     RD power map                  range-azimuth-Doppler tensor

--levels picks any subset:  adc, range_cube, rd_cube, power, rad (default: all 5)
```

The following table shows the data dimension (if exported) and the default plot config. Only three
levels have a quick-look plot; `range_cube` and `rd_cube` are export-only. Dimensions are `V` = virtual channels, `C` = chirps (slow-time),
`S` = ADC samples per chirp (fast-time), `R` = range bins (`= S/2`), `D` = Doppler bins (`= C`), `A` = azimuth grid points. 

| Level | Dimensions | Dtype | Preview plot |
|---|---|---|---|
| `adc` | `V × C × S` (virtual × chirp × sample) | complex | ADC magnitude (channel 0) |
| `range_cube` | `V × C × R` (virtual × chirp × range) | complex | — (export only) |
| `rd_cube` | `V × D × R` (virtual × Doppler × range) | complex | — (export only) |
| `power` | `D × R` (Doppler × range) | real | RD power map [dB] |
| `rad` | `A × D × R` (azimuth × Doppler × range) | real | RAD slice at given Doppler bin (`--doppler`, default: 0) |

```bash
# Export 15 frames of a scenario to .npz (one file per frame)
python -m simulator.simulate_cubes \
    --scenario scenarios/scenario_mimo_fixedpoint_dynrange.json \
    --out data/cubes/dynrange --frames 15

# Fixed-point export (adc = Q_adc), then preview the first frame
python -m simulator.simulate_cubes --scenario scenarios/scenario_mimo.json \
    --out data/cubes/fp --frames 15 --fp --preview

# Re-open a saved cube and view a range-azimuth slice at a Doppler bin
python -m simulator.simulate_cubes --view data/cubes/dynrange/frame_0000.npz --doppler 32
```

**Common flags.** `--scenario`, `--out`, `--frames`, `--levels` (subset of the levels
above), `--fp` (apply fixed-point), `--preview` (plot the first exported frame),
`--view <file.npz>` (open a saved cube), `--doppler` (Doppler bin for the RAD
slice). The reusable logic lives in `processing/cube_export.py` (`extract_cubes`
/ `save_cubes`), `processing/rad.py`, and `processing/cube_viz.py`.

---

## 4. Failure analysis & extensions: the report studys

The following reports probe where each stage of the chain fails and what the governing
lever is. They **build on one another**: later reports state only their
methodology deltas and quote earlier results as baselines. Report 1&2 are failure analysis, report 3&4&5 are extensions, report 6 is an implementation study.

| # | Report | Stage under test | Serves as baseline for |
|---|---|---|---|
| 1 | [Detection Performance Test](report/detection_evaluation/detection_performance_test_report.md) | CFAR + clustering | steady point-target Pd/Pfa curves |
| 2 | [Tracker Stress Test](report/tracker_stress/tracker_stress_test_report.md) | Tracker (association, M-of-N, coasting, KF/EKF/IMM) | track survival & association on steady targets |
| 3 | [Target Realism Test](report/target_realism/target_realism_test_report.md) | Target models (generator) → effects across the whole chain | realistic-target (Swerling fluctuation + extended multi-scatterer) detection & track penalties |
| 4 | [MIMO Evaluation](report/mimo_evaluation/mimo_evaluation_report.md) | TDM-MIMO front end (NCI, DoA, Cartesian clustering + tracking) | float MIMO detection/angle/tracking baseline |
| 5 | [Fixed-Point MIMO Front-End](report/fixedpoint_evaluation/fixedpoint_evaluation_report.md) | Front-end finite precision (ADC→FFT→NCI→CFAR input) | bit-width knees + radar-cube memory benefit |
| 6 | [C Implementation](report/c_implementation/c_implementation.md) | C11 port of the MIMO front end (`c_frontend/`) | stage-validated C baseline + C-vs-Python timing |

---

## 5. Install

Requires Python ≥ 3.10. The project uses [`uv`](https://docs.astral.sh/uv/)
(lockfile committed); runtime deps are NumPy and PyZMQ, with Matplotlib, SciPy,
and scikit-learn for evaluation/plots.

```bash
# Install dependencies into a managed virtualenv
uv sync

# Run the test suite (~429 tests)
uv run pytest tests/
# or, using the venv directly:
.venv/bin/python -m pytest tests/
```

---

## 6. Note on Seed hygiene in implementation 
The simulator draws from **five independent RNG streams**, so noise, fades,
scatterers, and per-element MIMO noise stay decoupled — each effect is
attributable and every run is reproducible. Base seeds are distinct (no
`SeedSequence` collisions); streams that vary per target or per frame are keyed
accordingly.

| # | Stream | Config field (default) | Keyed by |
|---|--------|------------------------|----------|
| 1 | CFAR calibration (pure-noise α) | CFAR `seed` (`0`) | scalar |
| 2 | Generator thermal noise + clutter | `RadarConfig.seed` (`99999`) | one stream per generator |
| 3 | Swerling RCS fluctuation | `RadarConfig.fluctuation_seed` (`77777`) | `[seed, target_index]` |
| 4 | Extended-target scatterers | `RadarConfig.extent_seed` (`555`) | `[seed, target_index]` |
| 5 | Per-element MIMO noise | `RadarConfig.element_noise_seed` (`33333`) | `[seed, frame_index, element_index]` |

The CFAR-calibration stream (1) is deliberately decoupled from the generator
noise (2) so the measured Pfa/Pd is honestly out-of-sample. Monte-Carlo runs vary
the generator noise seed (e.g. `1…30`); the ground-truth trajectory is
noise-independent, so only the detector input changes across seeds.
