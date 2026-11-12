# Faster one-pass detection front end on the NUCLEO-F446RE

The one-pass Q15 front end of `embedded/nucleo_f446re_frontend/` (range-Doppler,
NCI, CA-CFAR, Bartlett DoA), made faster. That folder is the certified build and
is not changed; this folder started as a copy of it.

A frame takes **1.78 s** here, against **5.6 s** for the certified build, and every
frame's detections are still checked bit for bit against a laptop reference.

| Change | Frame time | What it removed |
|---|---:|---|
| Certified build, as copied | 5.64 s | |
| Shorter listen for the next frame | 3.78 s | 1.9 s of waiting for the board's ready signal |
| Single-precision DoA scan | 2.73 s | 1.0 s of software floating point |
| CMSIS-DSP FFT | 2.73 s | nothing yet: it makes the next step possible |
| UART at 1.875 Mbaud | 1.78 s | 0.94 s of transmission |

The wire protocol, the memory layout and the kernels other than the FFT and the
DoA scan are those of the certified build; its README describes them.

## What was changed, and why

### 1. Time every part of a frame

Before changing anything, the board and the subscriber were made to report where
the time goes. The board's `TIME` line gives the cycles of each finish stage
(`power`, `cells`, `det`, `doa`, `emit`) and of the slowest single row
(`plane_max_cyc`). The subscriber times its own phases: the wait for the board's
`RDY`, sending, waiting for the result, and computing the reference.

For this the kernel's finish was split into separately callable stages
(`c_frontend/src/onepass/onepass_stages_q15.c`), with the same arithmetic.

The first run showed three costs, which the next steps remove in order:

| Cost | Per frame |
|---|---:|
| Waiting for `RDY` | 1.99 s |
| DoA scans | 180.5 Mcyc, 1.0 s |
| Sending 384 rows at 921600 baud | 2.5 s |

### 2. Shorter listen for the next frame

Between frames the board listens for a frame header and, if none comes, prints
its banner and `RDY` again. The listen was 2 s, so after a result the subscriber
waited up to 2 s for the next `RDY`. It is now 100 ms (`HDR_LISTEN_MS` in
`Core/Src/frontend.c`).

One side effect: the board now prints its banner about ten times a second while
idle, and the ST-LINK buffers that output. The subscriber therefore discards
what has accumulated before it reads the banner; otherwise the first banner
after a flash can be one from the previous firmware.

### 3. Single-precision DoA scan (`DOA=f32`, the default)

The DoA scan evaluates a 181-point spectrum per detection. The kernel's scan is
in double precision, which this FPU does not have, so it ran in software. The
new scan (`c_frontend/src/onepass/doa_cell_f32.c`) is in single precision and
steps the steering vector by one complex multiplication per element, with the
181 step factors in a table generated from the array geometry.

The scan fell from 180.5 Mcyc to 0.87 Mcyc per frame. The azimuth is a grid
index, and it stayed identical to the laptop's double-precision scan in every
frame. `DOA=f64` builds the original scan.

### 4. CMSIS-DSP FFT (`FFT=cmsis`, the default)

Rows are received in two alternating buffers: while one row is processed, the
next arrives. A row must therefore be finished before the next one has arrived
completely. The last row of each plane also runs that plane's 64 Doppler FFTs
and is the slowest. With the project's FFT it takes 716 264 cycles, and the time
one row takes on the wire is:

| Baud | Cycles for one row |
|---:|---:|
| 921 600 | 1 003 906 |
| 1 500 000 | 616 800 |
| 1 875 000 | 493 440 |
| 2 000 000 | 462 600 |

So the project's FFT cannot go above 921600. CMSIS-DSP's `arm_cfft_q15`
(`Core/Src/fft_q15_cmsis.c`, behind the kernel's `fft_q15.h` interface) brings
the slowest row to about 466 000 cycles and the row phase to 13.2 Mcyc per
frame (25.6 Mcyc in the certified run). `FFT=own` builds the project's FFT.

Two things had to be solved to use it:

- **Small inputs.** `arm_cfft_q15` drops the lowest bit of every input and
  truncates at every stage. That suits inputs near full scale, but the Doppler
  FFT's inputs are small, and the errors were as large as the values: the first
  run had 5389 detections the reference did not have. Each block is now shifted
  up as far as its largest value allows, transformed, and shifted back with
  rounding.
- **A reference to compare with.** The CMSIS-DSP FFT rounds differently from the
  project's, so its results are close to the old laptop reference but not equal:
  of 110 frames, 28 matched exactly. The laptop reference for this build is
  therefore the same batch chain built on the same FFT, `libfmcw_cmsis.so`.
  CMSIS-DSP has one version of this FFT for cores with DSP instructions and a
  plain C version, and the two round differently.
  `c_frontend/src/onepass/cfft_q15_board_path.c` compiles the board's version
  on the laptop, running each DSP instruction as the C function CMSIS-DSP
  supplies for it. With that reference every frame matches bit for bit.

Also tried and dropped: CMSIS-DSP's 32-bit FFT for the range transform. It was
slower and no closer to the project's FFT.

### 5. UART at 1.875 Mbaud (`BAUD`, `OVER8`)

The rate and the oversampling are build options. `Core/Src/main.c` sets them
after the generated initialisation, so the generated code and the `.ioc` keep
921600. The banner reports the rate the hardware really runs (`# baud`), and the
subscriber refuses to start when that differs from `--baud` by more than 3 %.

The board derives the rate from its 45 MHz bus clock, so only some rates are
exact:

| Requested | Rate on the board | Used |
|---:|---:|---|
| 921 600 | 918 367 (0.35 % slow) | default for `FFT=own` |
| 1 500 000 | exact | not needed |
| 1 875 000 | exact | default for `FFT=cmsis` |
| 2 000 000 | 2 045 454 (2.3 % fast) | not run: the slowest row does not fit |

The attempts at 1.875 Mbaud with the CMSIS-DSP FFT:

| Run | Result |
|---|---|
| 5 frames with a 20 ms pause after each row | 5 MATCH, no row error: the link works at this rate |
| 110 frames at full speed | 110 MATCH, 1.78 s per frame |
| 500 frames at full speed | 500 MATCH, no board error |

## Build and run

Prerequisites are those of the certified build (toolchain, `openocd`,
`Drivers/` generated from the `.ioc`), plus the CMSIS-DSP submodule:

```bash
git submodule update --init embedded/third_party/CMSIS-DSP
```

From the repository root, export the parameters and build both laptop
references:

```bash
python3 scripts/export_for_c.py \
    --scenario scenarios/scenario_mimo_stream_128.json \
    --out data/c_export_stream128 --frames 5
make -C c_frontend lib lib-cmsis
```

From this folder, build and flash:

```bash
make flash PYTHON=python3                    # CMSIS-DSP FFT, 1.875 Mbaud
make flash FFT=own PYTHON=python3            # project FFT, 921600 baud
```

| Option | Values | Default |
|---|---|---|
| `DOA` | `f32`, `f64` | `f32` |
| `FFT` | `cmsis`, `own` | `cmsis` |
| `BAUD` | a rate | 1875000 for `cmsis`, 921600 for `own` |
| `OVER8` | `1` (8 samples per bit), `0` (16) | 1 for `cmsis`, 0 for `own` |

Each combination builds in its own directory, `build/<FFT>-<DOA>-<BAUD>-o<OVER8>`.

Then, from the repository root, in two terminals:

```bash
python3 -m simulator.streaming_publisher \
    --scenario scenarios/scenario_mimo_stream_128.json
python3 -m simulator.streaming_subscriber_nucleo_tolerant \
    --params data/c_export_stream128/params.cfg \
    --baud 1875000 --max-frames 110
```

`--baud` must be the rate the board was built for.

## The subscriber

`simulator/streaming_subscriber_nucleo_tolerant.py` is used with this build. The
certified `simulator/streaming_subscriber_nucleo.py` is unchanged, and the new
script imports from it the serial conversation, the frame and row packing, the
result parsing and the exact comparison. What it adds:

| Addition | What it does |
|---|---|
| Full timing | Reads every field of the board's `TIME` line. Each summary shows the finish stages and the slowest row. |
| Row deadline | Compares the slowest row with the time one row takes on the wire, warns when it is exceeded, and lists the deadline for each candidate rate at the end. |
| Laptop phases | Times the wait for `RDY`, sending, the wait for the result and the reference, per frame. |
| Banner fields | Shows the DoA arithmetic, the FFT and the UART rate the board reports, and refuses to start on a wrong rate. |
| Fresh banner | Discards buffered output before reading the banner (see step 2). |
| Reference by FFT | `--reference auto` (default) compares with the batch chain on the FFT the board reports: `libfmcw.so` or `libfmcw_cmsis.so`. `own` and `cmsis` force one. |
| Tolerant comparison | `--compare tolerant` gives MATCH, CLOSE or MISMATCH with stated limits, for a reference whose FFT differs from the board's. The default is `exact`. |
| True targets | Judges the detections against the true targets in each published frame (see below). |
| Board errors | Counts the frames the board answers with `ERR`. They are not compared and were in no total before. |

`processing/c_frontend.py` gained one optional argument for this,
`CFrontend(params_cfg, lib_path=None)`, to load `libfmcw_cmsis.so` beside
`libfmcw.so`.

### The true-target report

Every published frame carries the true range, velocity and azimuth of its
targets. The final block reports, for the board and for the laptop chain on the
project's FFT:

- in how many frames each target was found, meaning a detection within one bin
  of it in range and in Doppler;
- the largest angle error of the strongest such detection;
- how many detections lie near a target and how many away from every target.
  "Near" is the CFAR guard region from the parameters, here 4 range bins and
  2 Doppler bins.

It answers a question the bit-exact comparison does not: whether changing the
FFT costs detections.

## Results

All runs: NUCLEO-F446RE at 180 MHz, scenario `scenario_mimo_stream_128.json`,
publisher at 20 frames per second, the subscriber always taking the newest
frame.

### Timing

Seconds per frame on the laptop, millions of cycles (Mcyc) on the board.

| | As copied | Listen 100 ms | + `DOA=f32` | + `FFT=cmsis` | + 1.875 Mbaud |
|---|---:|---:|---:|---:|---:|
| Frame time | 5.64 s | 3.78 s | 2.73 s | 2.73 s | 1.78 s |
| Wait for `RDY` | 1.99 s | 0.10 s | 0.09 s | 0.09 s | 0.09 s |
| Sending | | | 2.48 s | 2.48 s | 1.54 s |
| Wait for the result | | | | 0.12 s | 0.12 s |
| DoA scans | 180.5 Mcyc | | 0.87 Mcyc | 0.94 Mcyc | 1.00 Mcyc |
| Slowest row | 716 264 | | | 465 926 | 481 506 |
| Row deadline | 1 003 906 | 1 003 906 | 1 003 906 | 1 003 906 | 493 440 |

Empty cells were not recorded for that run. The remaining finish stages do not
depend on these options: power 0.48 Mcyc, cells 16.08 Mcyc, detection 0.6 Mcyc,
and printing 3.1 Mcyc at 921600 or 1.9 Mcyc at 1.875 Mbaud. The whole finish is
about 18 Mcyc, 101 ms.

### Correctness

| Build | Frames | Verdict against the reference with the same FFT |
|---|---:|---|
| `FFT=own`, 921600 | 110 | 110 MATCH, 0 tie flips |
| `FFT=cmsis`, 921600 | 110 | 110 MATCH, 0 tie flips |
| `FFT=cmsis`, 1.875 Mbaud | 500 | 500 MATCH, 0 tie flips, 0 board errors |

The CMSIS-DSP build against the true targets, beside the laptop chain on the
project's FFT (500 frames at 1.875 Mbaud):

| | Board, CMSIS-DSP FFT | Laptop, project FFT |
|---|---:|---:|
| Target 0 found, of 220 frames in the map | 220 | 220 |
| Target 1 found, of 222 frames in the map | 193 | 194 |
| Largest angle error | 0.6° | 0.6° |

Target 1 is weak in part of the run: its strongest cell came within 0.2 dB of
the CFAR threshold. At that distance one count of rounding decides, and in one
frame of 222 the two FFTs decided differently.

### Memory

Static memory is unchanged by any option: `data` 476 B and `bss` 124 480 B of
131 072 B. The CMSIS-DSP build adds 1 932 B of flash. Heap use peaked at 336 B
of 2 048 and the stack at 992 B of 4 096.

### What limits the frame time now

Of the 1.78 s, sending takes 1.54 s.

- **The slowest row.** At 1.875 Mbaud it takes 481 506 cycles of the 493 440
  available, a margin of 2.5 %. A missed deadline is reported by the board as a
  row error; it cannot produce a wrong result. A faster rate needs a shorter
  slowest row, for example by spreading a plane's Doppler FFTs over the rows of
  the next plane.
- **The path through the ST-LINK.** The rows need 1.05 s on the wire at this
  rate, but sending takes 1.54 s. Where the other 0.5 s goes has not been
  measured.
- **The finish** is 101 ms, of which the CFAR cell stage is 89 ms.

## Known properties of the detections

These belong to the processing chain and are the same on the laptop and the
board, with either FFT.

- **A strong target is detected several bins wide.** The first sidelobe of the
  Hann window is 31 dB below the peak, and the strong target here is up to 52 dB
  above the threshold.
- **A slow target has a weaker copy at zero velocity.** After the range FFT the
  chain subtracts from each range cell its mean over the chirps, to remove
  stationary clutter. For a target within about three Doppler bins of zero
  (2.4 m/s here) that mean is not zero, and subtracting it leaves a copy at
  zero velocity, 15 to 18 dB below the target. For the second target this copy
  is detected about 2.4 Doppler bins from the target, and it accounts for most
  of the detections the report counts as away from every target (64 in 110
  frames, 63 for the project's FFT).

## Tests

```bash
make -C c_frontend test PYTHON=python3
python3 -m pytest tests/test_c_frontend_cmsis.py \
    tests/test_streaming_subscriber_nucleo_tolerant.py \
    tests/test_frontend_speedup_params.py -q
```

`tests/test_c_frontend_cmsis.py` builds both laptop libraries and checks that
they load side by side, that each runs its own FFT, and that the CMSIS-DSP
reference finds both targets. That the laptop reference equals the board bit
for bit cannot be tested on the laptop; the board runs above are that test.
