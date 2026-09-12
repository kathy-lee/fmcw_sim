# One-Pass Streaming Front End on the NUCLEO-F446RE

**Date:** 2026-09-06
**Branch:** `feat/c-implementation-front-end`
**Pipeline under test:** one-pass streaming restructure of the Q15
detection front end (row-at-a-time range FFT → Doppler FFT per column →
NCI → CA-CFAR → Bartlett DoA), running on a NUCLEO-F446RE fed live
simulator frames over the ST-LINK VCP, verified frame-by-frame against
the laptop's frozen reference chain.

---
## 0. Motivation

Having implemented the FMCW MIMO frontend pipeline in C with fixed-point(Q15), consider one more step forward to a board: Can we run the frontend pipeline on a board(STM32 Nucleo F446RE), output detection results from the streaming publisher? 

Goal. The Q15 detection front end — range FFT, clutter removal, Doppler FFT, NCI, CA-CFAR, TDM compensation, Bartlett DoA — runs on the NUCLEO-F446RE(512 Kbytes flash memory, 128 Kbytes SRAM, 180 MHz CPU), fed live from streaming_publisher.py through a laptop subscriber over UART, emitting a detection list per processed frame.

Success criteria:
1. Correctness. For every frame, the board should produce the same detections as the straightforward pipeline in laptop.
2. Memory Fit on the board. Can consider shrink the scenario `scenarios/scenario_mimo.json`(V = n_tx*n_rx = 12, C = n_chirps = 64, S = n_range = 256).
3. No real-time claim.

## 1. Theoretical Feasibility

1. restructure the code to One-pass architecture 
- Row-streaming range-Doppler
- Zero-scratch capped CA-CFAR cell walk
- Tableless per-snapshot Bartlett DoA


2. lifetime/overlay memory model(V=12 fixed, varying C and S)
`USART2.BaudRate=921600`
`STM32F446xx_FLASH.ld`: `_Min_Heap_Size = 0x800;` and `_Min_Stack_Size = 0x1000;`.


## 2. Architecture Design

### 2.1 The processes and the links between 
As previous experiments, the publisher and the subscriber are
two Python processes on the same laptop, joined by ZMQ over TCP
loopback. The subscriber subscribes
to the simulator's ZMQ stream, feeds each frame to the board over the
serial wire, the board sends the detection results back over the same UART, the subscriber reports frames received vs processed vs dropped and end-to-end latency. The board hangs off the subscriber alone, over the ST-LINK's
virtual serial port, ZMQ never reaches the board. 
And the subscriber is made to play one more role as the referee: runs the identical frame through the laptop's frozen Q15
chain, and diffs the two detection lists live.

```
┌─ laptop ─────────────────────────────────────────────────────┐
│                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐   │
│  │ streaming_publisher  │   ZMQ   │ streaming_           │   │
│  │ simulate + publish   │ PUB─▶SUB│ subscriber_nucleo    │   │
│  │ frames at 20 fps     │────────▶│ quantize, feed the   │   │
│  └──────────────────────┘ tcp://  │ board, run the       │   │
│              127.0.0.1:5555       │ reference chain      │   │
│                                   │ (libfmcw.so), diff   │   │
│                                   └──────┬──▲────────────┘   │
│                                          │  │                │
└──────────────────────────────────────────┼──┼────────────────┘
        frames: header + 384 rows,         │  │ RDY + result blocks
        each 512 B + a checksum,           │  │ (DET/D, TIME, MEM,
        paced by the board's RDY           │  │  SAT, END)
                                           │  │
        UART over USB (ST-LINK VCP),       │  │
        /dev/ttyACM0, 921600 baud          │  │
                              ┌────────────▼──┴───────────┐
                              │ NUCLEO-F446RE             │
                              │ one-pass Q15 detection    │
                              │ front end (frontend.c +   │
                              │ synced c_frontend kernels)│
                              └───────────────────────────┘
```

### 2.2 One source, two compilations

The kernels under `c_frontend/src/` are the single source of truth, and
they are compiled twice — once for each processor that has to run them.
The board never receives `libfmcw.so`: that file holds x86-64 machine
code, and it is a *shared* library, which would need a dynamic loader and
an operating system. Bare metal has neither. What reaches the board is
the same C source, cross-compiled:

```
                        c_frontend/src/*.c
                     (the kernels, one copy)
                          │           │
      make sync           │           │      make -C c_frontend lib
      + arm-none-eabi-gcc │           │      gcc -fPIC -shared
      (Cortex-M4)         │           │      (x86-64)
                          ▼           ▼
                Core/Src/*.c      libfmcw.so
                      │                 │
                      ▼                 ▼
            build/q15/*.elf       ctypes wrappers
                      │           (CFrontend, CFrontendOnepass)
              openocd │                 │
                      ▼                 ▼
              NUCLEO-F446RE      streaming_subscriber_nucleo.py
              (runs one-pass)    (runs the batch reference)
```

Two compilers, two machine codes, two processors, one set of sources.
That is what gives the live diff its meaning: when the board's detections
match the library's, the only variables that differed were the compiler
and the CPU, so agreement is evidence that the arithmetic survived the
port intact.

It also fixes a maintenance rule. Editing a kernel means redoing *both*
compilations — `make -C c_frontend lib` for the laptop and
`make -C embedded/nucleo_f446re_frontend flash` for the board — or the
referee silently compares two different versions of the same code. The
banner handshake catches mismatched parameters, but nothing detects
mismatched code.

## 3. Experiment Design


1. Implementation and test on memory-lean kernals
2. Implementation and test on one-pass pipeline
3. Differential test of pipeline through ctypes wrapper
4. Implementation the subscriber and dry-run verification
5. Flash and monitor: Banner-only first, then full fun
6. One-shot run without publisher

     In `--one-shot` mode the left half of the laptop box is absent — no
     publisher, no ZMQ; the subscriber generates frame 0 of the scenario
     itself. The serial half is identical in both modes.

7. Streamed run with publisher


<!-- why one pass: the 196 608 B frame never exists on the board;
     planes / power / overlay memory structure; caller-provided storage;
     the row/finish split -->

## 4. Implementation

<!-- export, CubeMX generation, make flash, one-shot, streaming session;
     the deaf-window discovery and the --row-pace experiment -->

### 4.1 Flow in the subscriber
The steps marked ▸ are the two call sites of
`_process_one`, the per-frame path both modes share; the streaming loop
and `_process_one` are each expanded in their own chart below.

```
                    ┌─────────────────────────────┐
                    │ parse CLI args              │
                    │ read_params(--params)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ open NucleoLink             │
                    │ (serial open + discard      │
                    │  stale input backlog)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ wait_ready → parse_banner   │
                    └──────────────┬──────────────┘
                                   │
                 dims/alpha  ┌─────▼─────┐
              ┌──────────────│ board cfg │
              │   mismatch   │ == params?│
              ▼              └─────┬─────┘
     "refusing: ..."               │ match
      SystemExit(1)                ▼
                    ┌─────────────────────────────┐
                    │ build CFrontend (reference) │
                    │ + CFrontendOnepass (quant)  │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │   --one-shot?   │
                          └───┬─────────┬───┘
                        yes   │         │   no
              ┌───────────────▼──┐   ┌──▼───────────────────┐
              │ load scenario,   │   │ ZMQ SUB connect      │
              │ generate frame 0 │   │ + subscribe to topic │
              │ locally          │   └──┬───────────────────┘
              └───────┬──────────┘      │
                      │              ┌──▼───────────────────┐
              ┌───────▼─────────┐    │ streaming loop       │
              │ ▸ _process_one  │    │ (expanded in the     │
              └───────┬─────────┘    │  next chart)         │
                      │              └──┬───────────────────┘
              ┌───────▼─────────┐       │ Ctrl-C, or
              │ print verdict + │       │ --max-frames done
              │ TIME/MEM/SAT    │    ┌──▼───────────────────┐
              │ close link      │    │ "bridge stopped."    │
              │ exit 0 if MATCH │    │ (Ctrl-C); finally:   │
              │ else exit 1     │    │ close link + socket  │
              └─────────────────┘    └──────────────────────┘
```

The streaming loop, expanded. Its three back-edges — an empty poll, a
board-rejected frame, and the normal end of an iteration — each return
separately to the top-of-loop test:

```
           │ enter, from the ZMQ setup
           │      ┌◀── end of iteration ────────────────────────┐
           │      │  ┌◀── board ERR ────────────────────────┐   │
           │      │  │  ┌◀── nothing on the wire ────────┐  │   │
           │      │  │  │                                │  │   │
        ┌──▼──────▼──▼──▼─────────────┐                  │  │   │
  no ◀──│ processed < --max-frames?   │                  │  │   │
        └──────────────┬──────────────┘                  │  │   │
                       │ yes                             │  │   │
        ┌──────────────▼──────────────┐                  │  │   │
        │ poll socket, 1 s            │──nothing─────────┘  │   │
        └──────────────┬──────────────┘                     │   │
                       │ frame                              │   │
        ┌──────────────▼──────────────┐                     │   │
        │ drain queue: keep NEWEST,   │                     │   │
        │ drops += frame-id gap;      │                     │   │
        │ deserialize frame           │                     │   │
        └──────────────┬──────────────┘                     │   │
        ┌──────────────▼──────────────┐                     │   │
        │ ▸ _process_one              │                     │   │
        └──────────────┬──────────────┘                     │   │
        ┌──────────────▼──────────────┐                     │   │
        │ print verdict line          │──board ERR──────────┘   │
        └──────────────┬──────────────┘                         │
                       │ ok                                     │
        ┌──────────────▼──────────────┐                         │
        │ processed++; update         │                         │
        │ mismatch/tie-flip counts,   │                         │
        │ latency/row/finish sums,    │                         │
        │ heap/stack watermarks       │                         │
        └──────────────┬──────────────┘                         │
        ┌──────────────▼──────────────┐                         │
        │ every --summary-every:      │                         │
        │ print "-- N processed ..."  │                         │
        │ summary, reset window sums  │─────────────────────────┘
        └─────────────────────────────┘
```

Two labels the chart compresses: the `no` exit of the top test fires
when `--max-frames` is reached (an open-ended run, `--max-frames 0`,
leaves the loop only by Ctrl-C); and Ctrl-C itself can fire anywhere in
the loop — it is drawn as no arrow because no single box owns it — and
lands in the shutdown box of the `main()` chart. Note the board-ERR
back-edge bypasses `processed++`: an error frame does not count toward
`--max-frames`.

`_process_one`, called at the two ▸ marked steps above — one frame
through the board and the referee. A board error returns early (the
loop chart's `board ERR` back-edge is the caller reacting to exactly
that); otherwise the verdict line is returned at the bottom:

```
        ┌─────────────────────────────┐
        │ quantize frame to Q15       │
        │ (warn if ADC clips)         │
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │ wait_ready: drain backlog,  │
        │ then require a LIVE RDY     │
        │ (fresh permission from the  │
        │ board)                      │
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │ send_frame: 14 B header,    │
        │ then 12×32 rows, 512 B +    │
        │ checksum each (sleep        │
        │ --row-pace after each row,  │
        │ if set)                     │
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │ read_result: DET/D/TIME/    │
        │ MEM/SAT lines, ends at END  │
        │ or ERR, skips banner blocks │
        └──────────────┬──────────────┘
                ┌──────▼──────┐
                │ board ERR?  │──yes──▶ ┌───────────────────────────┐
                └──────┬──────┘         │ wait_ready: re-sync on    │
                       │ no             │ the recovery RDY; return  │
        ┌──────────────▼──────────────┐ │ "[frame N] BOARD ERR ..." │
        │ run laptop reference:       │ └───────────────────────────┘
        │ q15_process_frame           │
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │ compare_detections          │
        │ (board vs ref)              │
        └──────────────┬──────────────┘
        ┌──────────────▼──────────────┐
        │ return "[frame N] MATCH/    │
        │ MISMATCH (dets, tie flips)  │
        │ latency ..." verdict line   │
        └─────────────────────────────┘
```


### 4.2 Wire protocol

- Laptop → board, binary, little-endian:
  - Frame header, 14 bytes: magic `b"OPF1"`, `uint32` frame_id, `uint16` V, C, S. Board answers nothing on success; on a dimension mismatch it prints `ERR cfg <V> <C> <S>` (the dims it received) and returns to `RDY`.
  - Then V·C rows in v-major order (channel 0 chirps 0..C−1, then channel 1, ...): 512 payload bytes (S=128 `int16` I/Q pairs, i.e. the `(S, 2)` int16 row) followed by a 2-byte checksum = sum of the 512 payload bytes mod 65536, `uint16` LE.
- Board → laptop, text lines (`\n`-terminated):
  - Idle/banner block, repeated every ~2 s while waiting and once at startup:
    ```
    # onepass-frontend 1
    # clk 180000000
    # fpu 1
    # flash_ws 5
    # flash_acr 0x00000705
    # dims 12 32 128
    # cfar 2 4 4 8 <alpha %.17g> 6
    # doa 181 1
    # maxdet 64
    RDY
    ```
  - Per-frame result block:
    ```
    DET <n> <overflow>
    D <range_bin> <doppler_bin> <az_idx> <x %.9e> <y %.9e> <v %.9e> <snr %.9e> <power %.17g> <thresh %.17g>
    ... (n lines)
    TIME row_cyc <unsigned> finish_cyc <unsigned>
    MEM heap <unsigned> stack_peak <unsigned> free_start <unsigned>
    SAT <ld>
    END <frame_id>
    RDY
    ```
  - Errors: `ERR cfg ...`, `ERR row <v> <c>` (checksum), `ERR timeout <v> <c>` — each followed by a return to the idle/banner block. The bridge logs them and moves on; no retry.

### 4.3 Ping-pong buffer
  USART2 interrupt

## 5. Results


One certified session (2026-09-06): publisher at 20 fps, 110 frames processed end to end against the live reference diff. Reproduction commands(two terminals):

```bash
python -m simulator.streaming_publisher \
    --scenario scenarios/scenario_mimo_stream_128.json
python -m simulator.streaming_subscriber_nucleo \
    --params data/c_export_stream128/params.cfg \
    --port /dev/ttyACM0 --baud 921600 --max-frames 110 --summary-every 10
```

Verdicts: 110/110 MATCH, 0 mismatches, 0 tie flips. Every count, cell, azimuth (grid-exact), coordinate and SNR agreed with the laptop's frozen chain on the identical quantized input; SAT 0 throughout.
Cadence ~0.18 frames/s (~5.5 s per processed frame): 11 959 publisher frames dropped ≈ 109 per processed, by design — the subscriber always takes the newest frame. Latency publish→verdict 4.6–7.2 s, scaling with detection count.
Compute: row phase 25.6 Mcyc/frame (142 ms at 180 MHz), hidden entirely inside the ~2.1 s transfer by ping-pong interrupt reception. Finish phase 0.86–1.24 s summary means (~2.3 s at 31 detections): one 181-point Bartlett scan per detection in double precision, which this FPU (single-precision) emulates in software.
Memory: static data+bss 124 956 of 131 072 B (margin 6 116 B) — planes 98 304, the power/snapshot/spectrum overlay 16 384, cell list 1 536, rx ping-pong pair 2×516. Heap high-water 1 328 of the 2 048 reserved; stack peak 1 328 of 4 096 — both constant across all 110 frames (board-measured, heap<=/ stack_peak<= watermarks in the subscriber summaries).

The memory model confirmed, not assumed. The linker map shows peak static ≈ 116 KiB with ≥ 4 KiB margin against the budget, and the measured stack watermark stays under 2 KiB of the 4 KiB budget.
<!-- 110/110 MATCH session numbers, timing, memory, saturation;
     see embedded/nucleo_f446re_frontend/README.md "Measured results" -->

## 6. Limitations

<!-- software-double DoA bottleneck, DMA reception, no real-time claim -->
