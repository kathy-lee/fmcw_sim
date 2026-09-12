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

### 2.3 Wire protocol

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

### 4.1 Static storage and the overlay

Section 1 argued the budget; this is how the budget became declarations.

**Every lifetime is visible to the linker.** All the large buffers are
`static` at file scope in `frontend.c` — no `malloc`, no large locals.
The consequence is the one that matters for a memory claim: the linker
map becomes the proof, and an object that no longer fits does not crash
at 3 a.m., it fails the build. Measured with `arm-none-eabi-nm` on the
certified binary:

| object | bytes | what it holds |
|---|---:|---|
| `planes` | 98 304 | V·C·(S/2) range-Doppler bins, Q15 complex |
| `shared` | 16 384 | the overlay union (below) |
| `cells_list` | 1 536 | up to `ONEPASS_MAX_DET` = 64 CFAR cells |
| `rx_buf` | 1 032 | the receive ping-pong pair, 2 × 516 |
| `stdout_buf` | 512 | so `printf` never reaches the heap |
| `plan` | 104 | the kernel's plan struct |
| heap + stack reservation | 6 144 | `_Min_Heap_Size` 0x800, `_Min_Stack_Size` 0x1000 |

Two things are deliberately *not* in SRAM: `FRONTEND_PARAMS` (8 288 B,
including the velocity axis and the element positions) is `const` and so
lives in flash, and the FFT twiddle tables are the plan's only heap
allocation, built once at init.

**The kernels allocate nothing.** `onepass_q15_init` takes every buffer
as a pointer:

```c
int onepass_q15_init(OnepassPlanQ15 *p, const FrontendParams *prm,
                     cplx_q15 *planes, int64_t *power, CfarCellQ15 *cells,
                     cplx_q15 *snapshots, double *spectrum, cplx_q15 *rowbuf);
```

That is what lets the same kernels serve a laptop (where the wrapper
mallocs) and a 128 KiB part (where the board decides placement), and it
is what makes the overlay possible at all — the kernel cannot object to
two of its arguments pointing into the same memory.

**The overlay.** The header states the lifetime contract the caller may
exploit: *power is never read after the CFAR walk, and all snapshots are
gathered before spectrum is first written — so snapshots and spectrum may
overlay power's storage (snapshots first, disjoint ranges).* In
`frontend.c` that becomes a union:

```c
static union {
    int64_t power[FRONTEND_C * R_BINS];          /* 16 384 B */
    struct {
        cplx_q15 snapshots[ONEPASS_MAX_DET * FRONTEND_V];   /*  3 072 B */
        double spectrum[181];                               /*  1 448 B */
    } post;
} shared;
```

A union is sized by its largest member, so `shared` costs 16 384 B —
exactly what `power` alone would have cost. The post-CFAR stage is
therefore housed for free, and the alternative (separate arrays) would
have added 4 520 B, cutting the final margin from 6 116 B to under
1 600 B. At the shrunk scenario the overlay is what makes the fit
comfortable; at the full scenario it is the difference between fitting
and not.

**The instruments.** Two small routines produce the runtime numbers in
section 5. `paint_stack` fills the gap between the heap end and the
current stack pointer with a known byte at startup, and `stack_peak`
later counts how far the untouched pattern was eaten into — the stack's
high-water mark, which no static analysis can give. The heap figure is a
simple `sbrk(0)` probe taken right after init, before any `printf` could
have allocated — which it never does, because `stdout` is bound to the
static `stdout_buf` above.

### 4.2 Flow in the subscriber
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


### 4.3 Receiving while computing: a USART with no receive FIFO

**The hardware constraint.** The STM32F446's USART has no receive FIFO:
one data register plus the shift register, two bytes of slack. A third
byte arriving with none read overwrites an unread one and raises the
overrun flag. At 921600 baud in 8N1 framing — 1 start + 8 data + 1 stop,
10 bits per byte, 10.85 µs each — those two bytes are a **21.7 µs
deadline on every byte**, not a throughput allowance.

**Why blocking reception with a single buffer cannot work.** With a
blocking `HAL_UART_Receive`, the call returns when a row's last byte
lands, and nothing is listening again until the checksum walk and
`onepass_q15_row` have finished. The host transmits rows back-to-back,
so the next row's bytes start arriving immediately:

```
per-byte arrival        10.85 us
hardware slack           21.7 us   (2 bytes)
deaf window per row     ~365 us    (measured: 25.6 Mcyc / 384 rows)
                        ───────
bytes arriving in the deaf window   ~34
bytes the hardware can hold            2      ->  ~32 lost, every row
```

The margin is not thin, it is absent: the deaf window exceeds the
hardware's tolerance by a factor of ~17. Even against a pre-measurement
estimate of 190 µs per row — derivable from the FFT harness's measured
76.3 cycles per Q15 butterfly — the ratio is 17 : 2. Nothing in this
calculation required hardware: the baud rate, the frame format, the
absence of a FIFO (RM0390) and a cycle estimate were all available in
advance.

The trap is arithmetic at the wrong granularity. Per row, 365 µs of
compute against 5.58 ms of transfer is a 6.5 % duty cycle, which looks
like ample headroom — but a two-byte buffer does not impose a throughput
requirement, it imposes a deadline, and no amount of average headroom
satisfies a deadline. Worst-case response time against the peripheral's
buffer depth is the quantity that decides whether a receiver works.

**Confirmed on hardware, by the protocol.** Blocking single-buffer
reception was tried first, and the first full-speed run failed as
`ERR row 0 4` — a row whose 2-byte additive checksum did not match its
payload. This is the wire protocol earning its keep: corrupt samples were
refused at the door instead of silently becoming wrong detections, and
the failure surfaced as a named error with coordinates rather than an
inexplicable mismatch. The index is informative too — the error struck
row 4, not row 1, because buffering upstream in the ST-LINK's USB-CDC
bridge absorbed the first few deaf windows, and bytes were dropped at the
chip only once that buffering saturated.

**The controlled experiment.** Analysis names a cause; it does not
exclude rivals — wire noise, a host-side timing bug or a row-indexing
error would also produce checksum failures. So the feeder was paced
rather than the board changed: `--row-pace 0.009` sleeps after each row,
leaving the wire silent exactly while the board computes. The *same
binary* then processed all 384 rows to a MATCH (31 detections, latency
8.53 s). One run, one variable, and the only surviving explanation is
that the board cannot listen while it computes.

**The fix.** Reception became interrupt-driven ping-pong. Two 516-byte
buffers; `HAL_UART_Receive_IT` arms the *next* row before the current one
is checksummed, so the interrupt machinery lands bytes while the kernel
works; `plan.rowbuf` is re-pointed at the full buffer each row (the
kernel clobbers it in place, which is safe until that buffer is
re-armed); two `volatile` flags, written in `HAL_UART_RxCpltCallback` /
`HAL_UART_ErrorCallback` and polled in the main loop, synchronize the two
sides — single writer, single reader, no further locking required. Error
paths call `HAL_UART_AbortReceive` before draining, since blocking reads
and an in-flight interrupt transfer cannot both own the peripheral. The
vector-table entry is a hand-written `USART2_IRQHandler` in
`stm32f4xx_it.c`'s user fences delegating to `HAL_UART_IRQHandler`; the
startup file's weak default is an infinite loop, so the name must match
exactly.

**Cost and result.** +528 B of `bss` — the second buffer, the two flags
and alignment padding — leaving a 6 116 B margin. Full speed then MATCHed
on the first attempt, and the row phase's 25.6 Mcyc (142 ms per frame)
now does hide inside the ~2.1 s transfer.

## 5. Results

One certified session (2026-09-06): publisher at 20 fps, 110 frames
processed end to end against the live reference diff. Reproduction
commands (two terminals):

```bash
python -m simulator.streaming_publisher \
    --scenario scenarios/scenario_mimo_stream_128.json
python -m simulator.streaming_subscriber_nucleo \
    --params data/c_export_stream128/params.cfg \
    --port /dev/ttyACM0 --baud 921600 --max-frames 110 --summary-every 10
```

The three subsections below answer the three success criteria of
section 0, in order.

### 5.1 Correctness

**110/110 MATCH, 0 mismatches, 0 tie flips.** For every processed frame,
every detection agreed with the laptop's frozen batch Q15 chain on the
identical quantized input: count, range bin, Doppler bin, azimuth
(grid-exact, no tolerance), x, y, radial velocity and SNR. Not one
azimuth even landed one grid step away, though the comparison would have
tolerated that as a tie flip.

`SAT 0` throughout: the Q15 arithmetic never saturated on real frame
data, so the fixed-point headroom budgeted for the row FFTs and the
non-coherent integration was sufficient in practice, not only on the
exported test vectors.

The verdict is therefore stronger than "the port works": since the
reference is the *batch* chain while the board runs the *one-pass*
composition, each MATCH simultaneously confirms that the restructure
preserves detections and that cross-compilation to the Cortex-M4 changed
nothing.

### 5.2 Memory fit

Flash and SRAM both fit, with the SRAM figure the binding one:

| region | used | available | margin |
|---|---:|---:|---:|
| flash (`text` + `data`) | 51 652 B | 524 288 B | 90 % free |
| SRAM (`data` + `bss`) | 124 956 B | 131 072 B | 6 116 B |

The SRAM total decomposes as `planes` 98 304, the power/snapshot/spectrum
overlay 16 384, the cell list 1 536, the receive ping-pong pair 1 032,
`stdout` buffer 512, the plan struct 104, and the 6 144 B heap + stack
reservation. Because every one of those is a `static` object inside a
linker-visible section, the fit is not an estimate: an object that grew
past the budget would fail the link.

Runtime watermarks, measured on the board and reported per frame:

- **heap 1 328 B of the 2 048 reserved** — the FFT tables and windows
  only; `printf` never allocates, because `stdout` is bound to the static
  buffer.
- **stack peak 1 328 B of the 4 096 reserved** — by canary painting, so
  this is the deepest the stack ever actually went.

Both were *constant across all 110 frames*, which is the evidence that
matters: no growth with detection count, no slow leak, no path that
allocates more on a busier frame.

One correction against the feasibility estimate of section 1: static use
came out at 122.0 KiB rather than the predicted ≈ 116 KiB. The margin
criterion still held comfortably (6 116 B against the 4 KiB required),
but the prediction was ~6 KiB optimistic — and the ping-pong pair, which
the model had not anticipated, accounts for only 0.5 KiB of that gap.

### 5.3 Cadence — recorded, not claimed

**~0.18 frames/s, about 5.5 s per processed frame.** Of 12 069 published
frames, 110 were processed and 11 959 dropped — roughly 109 per processed
frame. The drops are by design: the subscriber always discards the queue
backlog and takes the newest frame, so it stays current instead of
falling ever further behind. End-to-end latency from publish to verdict
was 4.6–7.2 s, scaling with detection count.

Where the time goes:

| phase | cost | note |
|---|---:|---|
| row phase | 25.6 Mcyc = 142 ms/frame | hidden inside the ~2.1 s transfer |
| UART transfer | ~2.1 s | 197 KB at 921600 baud |
| finish phase | 0.86–1.24 s typical, ~2.3 s at 31 detections | dominant |

The Q15 row pipeline performs exactly as designed: 142 ms of compute,
entirely concealed within the transfer it overlaps. The unbudgeted cost
is the finish phase, which runs one 181-point Bartlett scan per detection
in **double** precision — and this FPU is single-precision, so those
scans are emulated in software. Cadence is therefore UART-and-finish
bound, and no real-time claim is made.

## 6. Limitations

<!-- software-double DoA bottleneck, DMA reception, no real-time claim -->
