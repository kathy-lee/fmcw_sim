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

Before any code, one question decides everything: what must be resident
in 128 KiB of SRAM while a frame is processed, and at what scenario size
does that fit?

### 1.1 Why the pipeline has to be one-pass

An ADC frame is V·C·S complex samples, 4 bytes each as int16 I/Q:

| scenario | V | C | S | frame |
|---|---:|---:|---:|---:|
| `scenario_mimo.json` | 12 | 64 | 256 | 786 432 B |
| shrunk (`..._stream_128`) | 12 | 32 | 128 | 196 608 B |

Both exceed the 131 072 B of SRAM — the shrunk frame by 50 %, the full
one by a factor of six. No amount of parameter shrinking rescues a batch
pipeline, because a batch pipeline needs the cube resident before it can
start. The board must therefore consume the frame *as it arrives* and
keep only what genuinely outlives a row. That is the whole argument for
the one-pass restructure: **the raw frame never exists on the board.**

### 1.2 What must persist

With R = S/2 range bins kept from a real-input FFT, the terms are:

| term | formula | at V=12, C=32, S=128 |
|---|---|---:|
| range-Doppler planes | V·C·R·4 = 2·V·C·S | 98 304 B |
| integrated power map | C·R·8 | 16 384 B |
| CFAR cell list | 64 · 24 | 1 536 B |
| DoA snapshots | 64·V·4 | 3 072 B |
| DoA spectrum | 181·8 | 1 448 B |
| one row buffer | S·4 + 2 = 514, padded to a multiple of 4 | 516 B |

The range-Doppler planes dominate, at 2·V·C·S bytes, which makes the scenario choice in section 1.5 a memory decision rather than a signal-processing one.

### 1.3 Restructures that remove batch-pipeline buffers

The one-pass form is not merely the batch chain fed differently: three
stages were rebuilt so that the buffers the batch pipeline needed — the
raw frame, the CFAR scratch arrays and the DoA steering table — are
never allocated. That is why none of them appears in the memory table of
1.2, which lists only what remains. What each restructure removes:

| restructure | what it removes | at V=12, C=32, S=128 |
|---|---|---:|
| row-streaming range-Doppler | the raw frame, and the FFT's private scratch copy | 196 608 B |
| zero-scratch CA-CFAR cell walk | four map-sized sliding-sum arrays and the full mask | 88 320 B |
| tableless Bartlett DoA | the 181 × V steering table | 17 376 B |

Each one keeps the frozen arithmetic exactly — same operations, same
order, same casts — so its output is bit-identical to the frozen stage
it replaces (tested in 3.1). What changes is only *where values live and
for how long*.

#### 1.3.1 Row-streaming range-Doppler

```c
int  rd_stream_q15_plan_init(RdStreamPlanQ15 *p, int n_chirps, int n_samples);
void rd_stream_q15_row(RdStreamPlanQ15 *p, cplx_q15 *row, int c,
                       cplx_q15 *plane, long long *sat);
void rd_stream_q15_plane_finish(RdStreamPlanQ15 *p, cplx_q15 *plane,
                                cplx_q15 *col, long long *sat);
```

This stage streams its *input*: rows are consumed the moment they arrive
off the wire, so the raw frame — 196 608 B, one and a half times the
board's entire RAM — never exists anywhere. It is the frozen
`rd_q15_process_channel` re-cut at the row boundary: the same operations
in the same order, split into "what can happen per row" and "what needs
the whole channel".

![Each arriving row is processed in place — DC removal, Hann, range FFT —
and its kept half lands in one row of the resident plane; when a channel's
last row arrives, the plane finish runs the slow-time work; the raw frame
never exists](figures/rd-stream-row-path.svg)

The split follows the data dependencies exactly:

- **Per row (fast time)** — DC removal, range Hann, range FFT, keep the
  S/2 positive-beat bins. Each of these reads only the one row, so it can
  run the moment the row lands. The arriving row buffer doubles as the FFT
  workspace; the frozen stage used a private scratch copy, and in-place
  produces the same values in the same order, so the outputs stay
  bit-identical.
- **Per plane (slow time)** — the clutter (slow-time mean) removal and the
  Doppler pass each read a whole *column* of chirps, so they cannot run
  until the channel's last row has arrived. `rd_stream_q15_row` at
  `c == C−1` is the trigger; the plane finish then walks the range bins,
  reusing the row buffer as its column workspace (C·4 = 128 B of its
  512 B).

`RdStreamPlanQ15` holds only what is frame-independent — two FFT plans
and two Hann windows, built once on the heap at start-up (~1.4 KiB).
Every byte of frame storage (the planes, the row buffer) belongs to the
caller: static arrays on the board, ordinary allocations in host tests.
One plan serves all 12 channels, because nothing per-channel lives in it.

#### 1.3.2 Zero-scratch capped CA-CFAR cell walk

```c
int cfar_cells_q15(const int64_t *power, const FrontendParams *p,
                   CfarCellQ15 *cells, int max_cells, int *overflow);
```

Where the range stage streams its input, the cell walk streams its
*output*. The frozen CFAR answers "is this cell a detection?" with a full
0/1 mask the size of the map. The cell walk asks the same question with
the same arithmetic, but writes only the answers that matter — a short
list of passing cells — so it needs no scratch memory at all.

![Frozen mask CFAR vs the cell walk: same power map in, identical threshold
decisions; ~84 KiB of sliding-sum scratch and a full mask versus zero scratch
and a 1.5 KiB ranked list](figures/cfar-frozen-vs-walk.svg)

Both sides implement the identical CA-CFAR decision — geometry, edge
clamping, Q16 threshold — on the same power map. The frozen path buys
O(1)-per-cell speed with four map-scale int64 scratch arrays and emits a
full mask. At this geometry (C=32, R=64, guard (2, 4), training (4, 8),
so window half-extents 6 × 12) they cost:

```
padded copy                 44 × 88 × 8     30 976
row sums                    44 × 64 × 8     22 528
full- and guard-window sums 2 × 32 × 64 × 8 32 768
mask                        32 × 64         2 048
                                           ───────
frozen working memory                      88 320 B  (86.3 KiB)
cell list (the walk's only storage)  64 × 24  1 536 B
```

The walk instead recomputes each window directly, which is O(window) per
cell — slower in theory, irrelevant here — and stores nothing between
cells. That ≈86 KiB is, together with the raw frame, one of the two
things that could never fit on the board.

![Decision flow of the cell walk: min-range gate, direct window sums, the
threshold test, append or ranked replacement into the capped list, final sort
by scan order](figures/cfar-cell-walk-flow.svg)

Most cells exit at the threshold and cost nothing. A passing cell is
appended while there is room; once the list is full, a newcomer must beat
the weakest kept cell's margin — `margin = (power << 16) − thresh_q16` —
to enter, and the spill is reported through `overflow`. A plain "stop at
64" cap would keep whichever detections came first in scan order; the
margin ranking keeps the strongest, so the cost is bounded by the cap,
not by the scene. A final sort restores the scan order a mask walk would
have produced, so downstream comparisons are element-by-element whenever
nothing overflowed.

Each kept cell also stores its `power` and `thresh_q16` at decision time.
That is what lets the power map die the moment the walk returns, which
the overlay of 1.4 depends on.

#### 1.3.3 Tableless per-snapshot Bartlett DoA

```c
int doa_bartlett_cell(const cplx *snap, const double *pos_wl, int n_elem,
                      double *spectrum);
```

```
one detected (doppler, range) cell
        │  read the cell from all V channel planes
        ▼
snapshot[V]
        │  Bartlett over 181 angles
        ▼
spectrum[181]
        │  spectral-maxima selection, strongest wins
        ▼
one azimuth
```

One detection gives a snapshot `x[e]`. The angular grid is
θ_g = (g − 90)° for g = 0…180, and the array element positions `p_e` are
in wavelengths. The **steering vector** is the phase pattern a plane wave
from θ_g imprints on the array:

```
a_g[e] = exp( j · 2π · p_e · sin θ_g ),      e = 0 … V−1
```

and the **Bartlett spectrum** correlates the snapshot with each candidate
angle's steering vector:

```
              | V−1                     |²
P(θ_g)  =     | Σ   conj(a_g[e]) · x[e] |
              | e=0                     |
```

If the wave truly comes from θ_g, then x[e] ≈ s·a_g[e]: the conjugate
cancels every phase, all V terms add in phase, and P(θ_g) ≈ V²·|s|² — a
peak. At wrong angles the terms partially cancel. The estimate is the
strongest strict local maximum of P (one-sided rule at the edges,
whole-grid argmax as fallback on a flat spectrum, ties to the lower
index); the azimuth is the winning index minus 90°.

The frozen DoA precomputes every steering vector at init — a 181 × V
table of complex floats, 181·V·8 = 17 376 B on the heap — and reads it
back on every use. The tableless form evaluates the *identical
expression* at the point of use, so the table never exists.

![Frozen DoA vs the tableless cell: the frozen path fills a 17 KiB steering
table at init and reads it per use; the tableless path computes the identical
expression at the point of use and stores nothing; both feed the same
accumulation, so the spectra are bit-identical](figures/doa-cell-tableless.svg)

The equivalence rests on one line. The frozen init writes

```c
p->steer[(size_t)g * n_elem + e] = (cplx){(float)cos(ph), (float)sin(ph)};
```

and the tableless form computes, at the moment of use,

```c
cplx a = { (float)cos(ph), (float)sin(ph) };
```

with `ph` built from the same doubles. Same libm calls, same float casts,
same order — every steering element is bit-identical, so the table only
ever cached values this line can regenerate. The cost moves from RAM to
compute: 181 × V sin/cos pairs per detection, the right trade where
detections are few and RAM is the wall.

Two further choices keep the function small. There is no plan and no
init/free pair, because nothing frame-independent is worth keeping; the
caller passes positions and scratch. And since the streaming pipeline
runs with `doa_n_sources = 1`, the cell form bakes in top-1 and returns
the grid index directly. The 1 448 B spectrum is caller scratch, so on
the board it can occupy storage the power map no longer needs (1.4).

### 1.4 Lifetimes and the overlay

The composition that joins the three restructures adds no signal
processing. Its design content is scheduling — what runs when (2.3) —
and storage — what lives where, and for how long, which is the subject
of this section.
The terms of 1.2 do not all live at once:

![Frame storage across one one-pass frame: planes live throughout, the row
buffer only while rows arrive, the cell list from the CFAR walk on; the
power map is live only from NCI to the CFAR walk, and the snapshots and
spectrum reuse its storage after](figures/onepass-frame-memory.svg)

Every byte of frame storage is caller-provided; the plans own only FFT
tables and windows (~1.4 KiB, heap, built once). The lifetimes are part
of the contract, not an accident:

- **`planes` (98 304 B)** — the irreducible core: resident all frame,
  because the DoA snapshots must read any (doppler, range) cell across
  all 12 channels after detection.
- **`rowbuf` (512 B)** — one buffer, three jobs in sequence: the
  arriving row, the range FFT's in-place workspace, and the Doppler
  pass's column scratch. No separate FFT or column buffer exists.
- **`power` (16 384 B)** — born at NCI, dead the moment the CFAR walk
  returns; each kept cell carries its own `power` and `thresh_q16`
  copies precisely so nothing needs the map afterwards.
- **`snapshots` (3 072 B) and `spectrum` (1 448 B)** — needed only after
  `power` dies. Every snapshot is gathered before the spectrum is first
  written, so both may occupy the power map's storage, snapshots first,
  in disjoint ranges, and their 4 520 B cost nothing extra.
- **No result array** — detections leave through a callback, one at a
  time, in the order a mask walk would emit them, so the caller holds
  one detection rather than 64.

### 1.5 Choosing the scenario size

V = 12 is fixed: the 3 TX × 4 RX virtual array is the point of the
experiment, and shrinking it would remove the MIMO content rather than
the memory pressure. That leaves C and S, and the planes term 2·V·C·S
decides:

| C | S | planes | verdict |
|---:|---:|---:|---|
| 64 | 256 | 393 216 B | 3× the whole SRAM |
| 32 | 256 | 196 608 B | over budget |
| 64 | 128 | 196 608 B | over budget |
| **32** | **128** | **98 304 B** | fits, 75 % of SRAM |
| 16 | 128 | 49 152 B | fits, but halves Doppler resolution |

C = 32, S = 128 is the largest pair whose planes leave room for
everything else. Summing the main terms of 1.2 at that point, with the
overlay applied and one row buffer, gives an estimate that confirms it:

```
planes                                98 304
overlay (power / snapshots+spectrum)  16 384
cell list                              1 536
row buffer                                516
stdout buffer                             512
                                     ───────
data footprint                       117 252 B  (114.5 KiB)
heap + stack reservation               6 144 B  (0x800 + 0x1000)
                                     ───────
total against 131 072 B              123 396 B  (120.5 KiB)
```

The reservation belongs in this total because the linker places
`._user_heap_stack` inside `bss`: an object that outgrows the budget
therefore fails at link time rather than corrupting the run. Flash is never
the constraint: the code and the `const` parameters (section 2.5) occupy
a small fraction of 512 KiB.

The prediction is thus "it fits, with roughly 7 KiB spare", and section
5.2 reports what the linker actually produced.

The shrink costs resolution — 32 Doppler bins instead of 64, 64 range
bins instead of 128 — but preserves the full virtual array, so the
detection chain under test is the real one.

### 1.6 Choosing the UART baud rate

How many bytes is a frame on the wire? A 14-byte header of our own
protocol (section 2.6), then 384 rows of 512 bytes each plus a 2-byte
checksum: 14 + V·C·(4S + 2) = 197 390 bytes. With 8N1 framing (1 start
bit, 8 data bits, 1 stop bit) every byte costs 10 bits. At CubeMX's
default of 115200 baud one frame would take about 17 s to send; at
921600 baud it takes about 2.14 s. The UART link was therefore set to
921600 baud, eight times the default and the highest of the usual
standard rates.

## 2. Architecture Design

This experiment joins two earlier ones: the FFT harness put a kernel on
this board but fed it no live data, and the hybrid subscriber ran the
whole front end on live frames but inside the laptop's own process. Here
the full pipeline runs on the board, on the live stream:

| | FFT harness (`nucleo_f446re_fft`) | hybrid subscriber (`streaming_subscriber_mimo_hybrid.py`) | this experiment |
|---|---|---|---|
| what runs | one FFT kernel, benchmarked | the full front end | the full Q15 detection front end |
| where | Nucleo | laptop, in the subscriber's process (ctypes) | Nucleo |
| input | no stream | live ZMQ frames | live ZMQ frames, forwarded over UART |
| arithmetic | Q15 and float, compared | float, per stage | Q15 |
| pipeline form | a single kernel | batch: whole cube in memory | one-pass: rows consumed on arrival |
| storage | static | `malloc`'d by `io.c` | static, caller-provided, overlaid |
| scenario | one FFT size, N = 256 test vectors | any exported scenario | shrunk to 12·32·128 by the memory model |
| subscriber's role | none | front end, then clustering, CTRV-EKF tracking, live view | quantizer, feeder and referee; stops at detections |
| verification | cycle counts beside the host; Q15 bit-exact | live A/B against the Python chain | per-frame diff against the laptop's batch Q15 chain |


### 2.1 System overview
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

**Two links, opposite rules for dropping frames.** The ZMQ link and the UART link treat frames in opposite ways, on purpose:

| link | may a frame be dropped? | who sets the pace |
|---|---|---|
| ZMQ, publisher → subscriber | yes — the subscriber keeps only the newest frame | the publisher, 20 frames/s |
| UART, subscriber → board | not silently — every row is checksummed, and a corrupt frame is refused with `ERR` and skipped | the board, with `RDY` |

The board cannot keep up with the publisher, and that is known before
any board is run: section 1.6 shows a frame takes about 2.14 s just to
send, so even with no computing at all the board could accept fewer
than 0.5 frames per second, while the publisher sends 20 — at least 40
times more. A queue that kept every frame would grow without limit, and each
verdict would concern an ever older frame. So whenever the subscriber is
ready for the next frame, it empties its queue, keeps only the newest
one, and counts the rest as dropped. Section 5.3 reports the rate and the
number of dropped frames actually measured. On the serial link the opposite holds: once a frame is under way,
it has to arrive intact, because a lost byte corrupts a row the board is
already computing on. The board therefore sets the pace — the subscriber
transmits only after a fresh `RDY` — and the checksums catch the corruption an overrun causes. The one place frames are thrown away on purpose is inside the subscriber, where discarding a whole frame is cheap and harmless.

### 2.2 The board program: layers and run cycle

**Three layers, dependencies pointing down only.** Each box names its
files and the functions they define; each arrow names who calls down
into the next layer.

```
┌────────────────────────────────────────────────────────────────────┐
│ generated by CubeMX                                                │
├────────────────────────────────────────────────────────────────────┤
│ main.c        defines main(): hardware init, frontend_init(),      │
│               then frontend_run() on every pass of while (1)       │
│ stm32f4xx_it.c · stm32f4xx_hal_msp.c · system_stm32f4xx.c ·        │
│ syscalls.c · sysmem.c · Drivers/ (HAL)                             │
├────────────────────────────────────────────────────────────────────┤
│ does: clock, GPIO and USART2 setup; interrupt handlers; libc stubs │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │ main() calls
┌──────────────────────────────────▼─────────────────────────────────┐
│ apparatus, hand-written                                            │
├────────────────────────────────────────────────────────────────────┤
│ frontend.c    defines frontend_init(), frontend_run()              │
│ retarget.c    defines _write(): printf goes to the UART            │
├────────────────────────────────────────────────────────────────────┤
│ does: static storage, UART conversation, reporting, instruments    │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │ frontend_init() and frontend_run() call
┌──────────────────────────────────▼─────────────────────────────────┐
│ kernels, synced from c_frontend and compiled unmodified            │
├────────────────────────────────────────────────────────────────────┤
│ onepass_q15.c defines onepass_q15_init(), _row(), _finish()        │
│ plus the kernels it builds on:                                     │
│ rd_stream_q15 · cfar_cells_q15 · doa_cell · fft_q15 ·              │
│ rd_q15 · convert_q15 · doa · pointcloud                            │
├────────────────────────────────────────────────────────────────────┤
│ does: range-Doppler, NCI, CFAR, DoA, detections                    │
│ rule: portable C, no HAL, no I/O, no board knowledge               │
└────────────────────────────────────────────────────────────────────┘
```

The bottom layer is the one under test, and the rule protecting it is
strict: no kernel includes `main.h` or any HAL header. That is what lets
the identical files compile for x86-64 as well (section 2.4), and it is
why a detection printed by the board is a detection the host-tested code
computed.

The middle layer is everything board-specific that was written by hand.
`frontend.c` owns the static storage (section 4.1), receives and checks
frames, drives the kernels, and prints results; `retarget.c` routes
`printf` to the UART by overriding newlib's weak `_write`, so every
result line leaves the board through `HAL_UART_Transmit`.

The top layer is regenerable, and hand-written code enters it only
through CubeMX's `USER CODE` fences, which survive regeneration: the two
calls into the apparatus in `main.c`, and the `USART2_IRQHandler` in
`stm32f4xx_it.c` that reception needs (section 4.4).

**The run cycle.** `main()` calls `frontend_run()` over and over, and each
call is either an idle window or one frame: print the banner and `RDY`,
wait up to two seconds for a frame header, and if one arrives, receive the
rows, run the pipeline and print the result. Every call begins with a new
banner, and its `RDY` is what paces the subscriber. Section 4.3 draws the
cycle in full.

### 2.3 The one-pass pipeline: row phase and finish phase

The one-pass pipeline is set up once, by `onepass_q15_init()`, which
`frontend_init()` calls at start-up. After that, every frame drives it
with two calls in a fixed pattern: 384 calls to `onepass_q15_row()` as
the rows arrive, then one call to `onepass_q15_finish()`.

```
frontend_run()
│
├─ 384 × onepass_q15_row(v, c)            as each row arrives, channel by channel
│    chirp c of channel v:
│      DC removal → range Hann → range FFT → keep S/2 bins
│      → written into row c of plane v
│    if c is the channel's last chirp, plane v is now complete:
│      slow-time mean removal → Doppler Hann → Doppler FFT → fftshift
│      (in place, inside plane v)
│
└─ 1 × onepass_q15_finish(emit)           after the last row
     NCI: sum of |plane|² over the 12 planes → power map
     CA-CFAR cell walk → up to 64 cells
     gather each cell's 12-element snapshot from the planes
     for each cell:
       TDM compensation → Bartlett DoA on the 181-point grid
       → range, azimuth, x, y, radial velocity, SNR
       → emit(): one line per detection, sent to the UART at once
```

Every stage of the front end appears exactly once. The row phase is the
batch `rd_q15` chain reordered for rows that arrive one at a time, in the
same stage order and bit-exact against it.

What each step reads and writes, which is where the memory model of
section 1 comes from:

| step | reads | writes |
|---|---|---|
| row phase | the row just received | its row of one plane |
| NCI | all 12 planes | the power map |
| CFAR | the power map | the cell list |
| snapshots | planes, cell list | snapshots, in the power map's storage |
| DoA | snapshots | spectrum (also in that storage), detections |

Splitting into a row phase and a finish phase has the following benefits.

**Makes streaming possible.** A row call needs only the row it
was handed and the plane that row belongs to. Nothing in it waits for a
later row, so each row can be processed while the next one is still on
the wire (section 4.4).

**Shapes the memory layout.** The planes must survive until the very
end, because the DoA snapshots are gathered from them after detection. The power map, in
contrast, is dead as soon as CFAR has produced its cell list, which is
exactly the window the overlay of section 1.4 exploits.

**Keeps the scene-dependent cost out of the streaming path.** The row
phase does the same work for every frame, whatever the scene. The finish
phase has a fixed part too: NCI and the CFAR walk over every cell. But it
also has a part that grows with the number of detections: each detection
gets its own 181-point Bartlett scan, computed in double precision, which
the board's single-precision FPU cannot execute, so it runs in software.
Since the row phase overlaps the transfer, as described above, it adds no
time to the frame; the finish phase is where a busy scene costs time.
Section 5.3 reports the measured cost of both.


### 2.4 One source, two compilations

The kernels under `c_frontend/src/` are the single source of truth for
both pipelines, and they are compiled twice — once for each processor
that has to run them.
The board never receives `libfmcw.so`: that file holds x86-64 machine
code, and it is a *shared* library, which would need a dynamic loader and
an operating system. Bare metal has neither. What reaches the board is
the same C source, cross-compiled:

```
                            c_frontend/src/
                        (the kernels, single source)
                      ┌────────────┴────────────┐
             one-pass pipeline          all of src/: batch Q15 chain,
              + shared kernels          one-pass pipeline, shared kernels
                      │                         │
            make sync │                         │ make -C c_frontend lib
  + arm-none-eabi-gcc │                         │ gcc -fPIC -shared
          (Cortex-M4) │                         │ (x86-64)
                      ▼                         ▼
                Core/Src/*.c               libfmcw.so
                      │                         │
                      ▼                         ▼
               build/q15/*.elf           ctypes wrappers
                      │           (CFrontend, CFrontendOnepass)
              openocd │                         │
                      ▼                         ▼
                NUCLEO-F446RE     streaming_subscriber_nucleo.py
               (runs one-pass)     (runs the batch chain as reference)
```

Two compilers, two processors, one source tree, but two different
pipelines from it: the board runs the one-pass pipeline, the laptop the
batch Q15 chain, and both share the lower-level kernels such as the Q15
FFT. That is what gives the live diff its meaning: when the board's
detections match the library's, both the restructure (batch → one-pass)
and the port (x86-64 → Cortex-M4) preserved the detections.

**Keeping the two builds in step.** One source tree does not mean both
machines run the same version of it. Each build is a snapshot of
`c_frontend/src` taken when its command runs, and the two commands run
independently:

```
make -C embedded/nucleo_f446re_frontend flash   → the board, as of that moment
make -C c_frontend lib                          → the laptop, as of that moment
```

If a kernel is edited between the two builds, the two machines run
different code. In practice it is the board that falls behind, because
the tests rebuild the library automatically, while nothing reflashes the
board when a kernel changes. After editing a kernel, the board must
therefore be reflashed by hand; otherwise the stale board shows up as
MISMATCH verdicts that look like a porting bug, or, if the change does
not affect detections on these frames, does not show up at all: every
frame still MATCHes, against code the board is not running.

### 2.5 One parameter file, two bindings

The C front end needs the
settings the Python simulator processes with: the frame dimensions, the
CFAR settings, the virtual antenna positions and the scales that turn
bins into metres and m/s. C cannot import them from Python, so
`scripts/export_for_c.py` writes them to a flat `key=value` file,
`params.cfg`, alongside a few exported test frames. It holds 17 keys:

| group | keys |
|---|---|
| frame and array dimensions | `n_tx`, `n_rx`, `n_virtual`, `n_chirps`, `n_samples` |
| CFAR | `cfar_alpha`, `cfar_guard_r`, `cfar_guard_d`, `cfar_training_r`, `cfar_training_d`, `pfa`, `min_range_bins` |
| DoA | `pos_wl` (virtual antenna positions in wavelengths), `doa_n_sources` |
| bins to physical units | `range_per_bin`, `velocity_axis` |
| quantization | `adc_full_scale` |

That file reaches the two machines by different routes, and at different
times:

```
                       scripts/export_for_c.py
                                  │
                                  ▼
                              params.cfg
                                  │
                  ┌───────────────┴───────────────┐
         laptop: at run time           board: at compile time
                  │                               │ make sync runs
                  │                               │ gen_frontend_params.py
                  ▼                               ▼
    CFrontend / CFrontendOnepass          frontend_params.h
    pass the path to the library,                 │ compiled in
     which parses the file when                   ▼
        the object is created         FRONTEND_V, _C, _S macros
                  │                  + FRONTEND_PARAMS, a const
                  │                    struct stored in flash
                  │                               │
                  └─────── banner check ──────────┘
               at start-up the subscriber compares the
              board's reported dims and CFAR alpha with
            params.cfg, and refuses to run if they differ
```

On the laptop, the file is parsed each time a front-end object is
created, so a re-export takes effect on the next run. On the board, the
same file is turned into a header file 'frontend_params.h', and compiled in. For the board this is
not a choice but a requirement: V, C and S set the sizes of the static
arrays of section 4.1, and C needs an array's size when it compiles it.
There is also no file system on the board to read a configuration from.
`FRONTEND_PARAMS`, 8 288 B, therefore sits in flash as `const` data,
costing no SRAM. The parameters never travel over the UART: the board
has them in flash from the moment it is flashed, and the frame header
carries only V, C and S, which the board checks against its own and
refuses on a mismatch.

**Parameter drift is caught; code drift is not.** Parameters can drift
apart the same way as the code (section 2.4): re-export them and the
laptop follows on its next run, while the board keeps the values it was
last flashed with. Here, though, there is a guard. At start-up the
subscriber reads the same `params.cfg` itself, compares it with the
board's banner, and refuses to start unless the dimensions (V, C, S)
match exactly and CFAR alpha matches to within 1e-9 relative. The banner
also shows the CFAR guard and training sizes and the minimum range bin,
but those are displayed, not checked. There is no equivalent check for
code: the banner reports the board's parameters, not which version of
the sources it was built from.

### 2.6 Wire protocol

The laptop and the board talk over the UART link, in a
different format each way. Frames go to the board in binary, because the
same samples written out as text would take several times longer to
send. Results come back as text lines, which the subscriber parses and a
person can read in any serial monitor. The board sets the pace: the
subscriber sends a frame only after the board has printed `RDY`.

- Laptop → board, binary, little-endian:
  - Frame header, 14 bytes: magic `b"OPF1"`, `uint32` frame_id, `uint16` V, C, S. Board answers nothing on success. If the magic is not `OPF1` or the dimensions do not match, it prints `ERR cfg <V> <C> <S>` (the dimension fields as received), discards the rest of the incoming frame, and starts its next cycle with a banner.
  - Then V·C rows in v-major order (channel 0 chirps 0..C−1, then channel 1, ...): 512 payload bytes (S=128 `int16` I/Q pairs, i.e. the `(S, 2)` int16 row) followed by a 2-byte checksum = sum of the 512 payload bytes mod 65536, `uint16` LE.
- Board → laptop, text lines (`\n`-terminated):
  - Banner block, printed at the start of every cycle: once after power-on, every ~2 s while idle, and again after every result block or error:
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

    If `onepass_q15_init()` failed at start-up, the banner carries one extra line, `# ERROR plan init failed`, just before `RDY`, and the board accepts no frames.

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
  - Errors: `ERR cfg ...`, `ERR row <v> <c>` (checksum), `ERR timeout <v> <c>` — each followed by a return to the idle/banner block. The subscriber logs them and moves on; no retry.

## 3. Experiment Design

### 3.1 Implementation and test of the memory-lean kernels

The three restructures of section 1.3 were each built as a
separate kernel. Each kernel's test runs the new kernel and the frozen kernel on the same input, and requires them to agree exactly:

- `test_cfar_cells_q15` — the CFAR cell walk against the frozen
  reference mask `cfar_q15_ca_mask_ref`, on a 16 × 16 noise map with
  two strong cells added. The list must contain exactly the cells the
  mask marks, in the same scan order, with the same power, and with a
  stored threshold that reproduces the mask's decision. Exactly two
  cells must pass, none below the minimum range, with no overflow. A
  second case sets alpha to 0, so all 224 cells beyond the minimum range
  pass, and caps the list at 8: the walk must keep exactly 8, raise the
  overflow flag, keep them in scan order, and keep the strongest — the
  weakest kept cell is at least as strong as the 9th-strongest
  candidate.
- `test_rd_stream_q15` — row-streaming range-Doppler against the batch
  `rd_q15` chain, on 3 random channels of 32 chirps × 128 samples,
  bounded like the quantizer's output. Rows are fed in one at a time and
  the plane is then finished; the result must be bit-identical to the
  batch plane, with the same saturation count.
- `test_doa_cell` — the tableless Bartlett DoA against the table-based
  frozen `doa_bartlett`, on 50 random 12-element snapshots. The azimuth
  index must match, and the whole 181-point spectrum must be
  bit-identical.

### 3.2 Implementation and test of the one-pass pipeline

The three kernels were then composed into the one-pass pipeline,
`onepass_q15_init/_row/_finish` (section 2.3). Its test runs the whole
pipeline and the frozen stage-wise chain — batch range-Doppler per
channel, NCI, the reference CFAR mask, point cloud — on the same input,
and requires them to agree exactly:

- `test_onepass_q15` — 3 synthetic frames of 12 channels × 32 chirps ×
  128 samples, each noise plus one strong target. The target is needed:
  integrating 12 channels concentrates the noise so tightly that noise
  alone never crosses the threshold, and a test with no detections would
  compare nothing. The pipeline receives the rows one at a time, channel
  by channel, then finishes. Its planes must be bit-identical to the
  frozen chain's range-Doppler cube, with the same saturation count and
  no overflow. It must report the same number of detections, at least
  one, and each detection must match exactly: range bin, Doppler bin,
  azimuth, x, y, radial velocity, SNR, and a power equal to the frozen
  power map at that cell. On the host every buffer is a separate
  allocation, so the overlay of section 1.4 is not used here.

### 3.3 Differential test of the pipeline through the ctypes wrapper

The one-pass pipeline was exposed to Python through a flat C API,
`fmcw_onepass_*`, built into `libfmcw.so`, and the wrapper class
`CFrontendOnepass` — the same path the subscriber later uses to quantize
frames. This step also introduces the reduced scenario,
`scenario_mimo_stream_128.json`: V = 12, C = 32, S = 128, with the ramp,
the chirp period and the bandwidth halved together with the chirp and
sample counts, so the range window stays about 9.6 m and both of its
targets, near 3 m and 7 m, stay in view. The test exports 5 frames of
that scenario and rebuilds the library, then checks:

- `test_onepass_matches_frozen_chain` — each frame is quantized, with no
  sample clipped, and processed twice: by the one-pass pipeline, and by
  the frozen batch Q15 chain through `CFrontend`. Both must find at least
  one detection, and the two detection arrays must be exactly equal.
- `test_detections_land_near_ground_truth` — a loose physical check, so
  that the exact comparison cannot pass on a broken scenario: detections
  must appear within 0.5 m of both targets, and the far target's
  azimuth must lie within 3° of its true 25°.
- `test_dims` — the library reads the dimensions 12 × 32 × 128 from the
  exported parameters.

Unlike section 3.2, the input here is real simulator frames rather than
synthetic data. Like section 3.2, the library allocates every buffer
separately, so the overlay is not used.

### 3.4 Implementation of the subscriber and dry-run verification

`streaming_subscriber_nucleo.py` implements the laptop side of the wire
protocol (section 2.6), the banner check, the result parser and the
detection comparison. All of it is tested without a board — a dry run, with no serial port, no board and no publisher.
Its 10 tests fall into three groups:

- **Wire format.** The frame header must be exactly 14 bytes: `OPF1`,
  the frame id as a 32-bit little-endian integer, then V, C and S as
  16-bit integers. A row must be 514 bytes: the int16 row itself, then
  a checksum equal to the sum of those 512 bytes modulo 65536.
- **Parsing and the comparison rules.** A sample banner must yield the
  dimensions 12 × 32 × 128, `maxdet` 64, and CFAR alpha to full
  precision. A software loopback renders two reference detections as a
  board result block, parses it back, and compares it with the original:
  it must MATCH with no tie flips, while a copy with one x coordinate
  moved by 1e-3 — far beyond the 1e-6 tolerance — must fail. An azimuth
  one grid step off must still MATCH but count as one tie flip, two
  steps off must fail, and a missing detection must fail.
- **The serial conversation, against a fake board.** A fake serial port
  stands in for the board: it holds stale banners in its buffer and then
  plays scripted board output. Opening the link must clear the stale
  ones; a frame must be sent only after a live `RDY`, which then yields a
  MATCH with both detections and exactly 14 + 384 × 514 bytes written; a
  banner arriving just before the result must be skipped rather than read
  as an empty result; and a silent board must raise an error instead of
  waiting forever.

A dry run checks the logic of encoding, parsing and comparing. It
cannot check timing on a real wire: the receive overrun of section 4.4
was invisible here and appeared only on the board.

### 3.5 Flash and monitor: banner-only first, then the full program

The board program was brought up in two stages, so that each stage adds
only a few new things that can go wrong.

- **Banner-only program.** The board project was cloned from the FFT
  harness, set to 921600 baud with the heap and stack reservations of
  section 1.5, and given a wrapper Makefile whose `make sync` copies the
  kernel sources in and generates the parameter header from
  `params.cfg`. This first program does nothing but print the banner and
  `RDY` every two seconds. Flashing it and reading the serial port checks
  the toolchain, the flashing, the clock, the baud rate and the parameter
  header before any pipeline code runs: the banner must show the 180 MHz
  clock, the FPU switched on, 5 flash wait states, the dimensions
  12 × 32 × 128, and CFAR alpha with all 17 digits equal to `params.cfg`.
- **Full program.** `frontend.c` then gained the static storage and the
  complete run cycle of section 4.3, and is checked twice. Before
  flashing, the build must link: because all large buffers are static and
  the heap and stack reservations sit inside `bss`, a program that does
  not fit fails right here with `region RAM overflowed`, and
  `arm-none-eabi-size` and `make map` show the static regions against the
  131 072 B of SRAM. After flashing, the banner must appear without
  `# ERROR plan init failed`, which shows that `onepass_q15_init()` could
  set up its FFT tables within the board's heap.

### 3.6 One-shot run without the publisher

The first run through the whole system uses a single frame and no
stream. With `--one-shot`, the left half of the laptop box in section
2.1 is absent — no publisher, no ZMQ: the subscriber generates frame 0
of the scenario itself. The serial half is identical to the streamed
run: the frame is quantized, sent after a live `RDY`, processed on the
board, and compared with the laptop's batch Q15 chain, exactly as every
streamed frame will be.

One fixed frame makes a failure easy to interpret. Sections 3.1–3.4 have
already shown that the laptop side is correct, so a disagreement here
can only come from the wire or the board build, and the result block
shows which:

- a non-zero `SAT`, where the laptop saw none, means the board computed
  on different samples than the laptop sent;
- an azimuth that differs points at the math library, since the
  board's and the laptop's C libraries may round `sin` and `cos`
  differently;
- a different number of detections means the CFAR input differs; the
  power and threshold printed with each detection show where, because
  they are printed to full precision and either equal the laptop's
  values at those cells or not.

**Passes when** the banner is accepted, the verdict is MATCH, `SAT` is
0, the stack peak stays under 2 048 B, and the heap is about
1.2–1.5 KiB. A one-shot MATCH is a gate to the streamed run, not a
result.

### 3.7 Streamed run with the publisher

The acceptance run puts the whole system of section 2.1 to work: the
publisher streams the scenario at 20 frames per second, and the
subscriber, keeping only the newest frame, feeds the board one frame at
a time and compares every result with the laptop's batch Q15 chain. The
commands are listed in section 5.

Compared with the one-shot run, this adds what a single frame cannot
show: many different frames, since the targets move from frame to
frame; the stream's own behavior — dropped frames and end-to-end
latency; and the board's memory use over time. Every processed frame
prints its verdict, and every 10 processed frames the subscriber prints
a summary: frames processed and dropped, mismatches and tie flips so
far, mean latency, the cost of the row and finish phases, and the
highest heap and stack use the board has reported so far.

**Passes when** at least 100 frames are processed with zero mismatches,
and the stack peak stays under 2 048 B throughout. Tie flips do not fail
a frame, but each one is listed rather than hidden. The run stops after
110 processed frames; its results are reported in section 5.

## 4. Implementation

### 4.1 Static storage and the overlay

Section 1 argued the budget; this is how the budget became declarations.

**Every lifetime is visible to the linker.** All the large buffers are
`static` at file scope in `frontend.c` — no `malloc`, no large locals.
The consequence is the one that matters for a memory claim: the linker
map becomes the proof, and an object that no longer fits does not crash
at 3 a.m., it fails the build. Sizes from `arm-none-eabi-nm` on the certified binary, except the last
row, which the linker script sets:

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
lives in flash, and the FFT tables and Hann windows are the plan's only heap
allocation, built once at init.

**The kernels allocate no frame storage.** `onepass_q15_init` takes every buffer
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
comfortable.

**The instruments.** Two small routines produce the runtime numbers in
section 5. `paint_stack` fills the gap between the heap end and the
current stack pointer with a known byte at startup, and `stack_peak`
later counts how far the untouched pattern was eaten into — the stack's
high-water mark, which no static analysis can give. The heap figure is a
simple `sbrk(0)` probe taken right after init, before any `printf` could
have allocated — which it never does, because `stdout` is bound to the
static `stdout_buf` above.

### 4.2 Flow in the subscriber

The first chart is the control flow of the subscriber's `main()`: the
shared setup, then the fork into its two modes. The steps marked ▸ are
the two call sites of
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
              │ close link      │    │ print stop message   │
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
`--max-frames`. The chart also leaves out one skip: after draining the
queue, a frame whose header does not describe a 3-D array is ignored,
and the loop starts over before the frame is counted. This publisher
sends only 3-D frames, so the skip never fires here.

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


### 4.3 Flow on the board

One call to `frontend_run()` is one idle window or one frame. The chart
follows a single call through every way it can end:

```
            ┌──────────────────────────────┐
            │ banner: # config lines, RDY  │
            └──────────────┬───────────────┘
                ┌──────────▼──────────┐
                │ onepass_q15_init()  │──no──▶ wait 2 s and return; each banner
                │ succeeded at        │        says # ERROR plan init failed
                │ start-up?           │
                └──────────┬──────────┘
                           │ yes
            ┌──────────────▼───────────────┐
            │ read the 14-byte header,     │──nothing in 2 s──▶ return (idle)
            │ blocking, 2 s timeout        │
            └──────────────┬───────────────┘
                   ┌───────▼───────┐
                   │ OPF1 and dims │──no──▶ ERR cfg, drain, return
                   │ match?        │
                   └───────┬───────┘
                           │ yes
            ┌──────────────▼───────────────┐
            │ 384 rows, received by        │──bad checksum───▶ ERR row, drain, return
            │ interrupt: checksum each,    │──no row in 1 s──▶ ERR timeout, drain, return
            │ onepass_q15_row()            │
            └──────────────┬───────────────┘
            ┌──────────────▼───────────────┐
            │ onepass_q15_finish(): a DET  │
            │ count line, then one line    │
            │ per detection, printed as    │
            │ each is computed             │
            └──────────────┬───────────────┘
            ┌──────────────▼───────────────┐
            │ TIME, MEM, SAT, END, RDY     │
            └──────────────┬───────────────┘
                           ▼
                        return
```

Two properties follow from this shape. First, results leave the board
*during* the call, not at its end: there is one line per detection, each
transmitted as soon as that detection is finished, so by the time `frontend_run()` returns, the
subscriber already holds the whole result block. Second, every exit —
idle, error or success — leads to the same next event: the loop calls
`frontend_run()` again, which prints the banner again. An idle board
therefore banners every two seconds whether or not anyone is listening,
which is the fact behind the subscriber's rule of trusting only a freshly
arrived `RDY` (section 4.2).

The row-reception box is the one step where the board must receive and
compute at the same time; section 4.4 explains how that is done.

### 4.4 Receiving while computing: a USART with no receive FIFO

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
average deaf window     ~365 us    (25.6 Mcyc / 384 rows)
                        ───────
bytes arriving in the deaf window   ~34
bytes the hardware can hold            2      ->  ~32 lost per row on average
```

The margin is not thin, it is absent: the deaf window exceeds the
hardware's tolerance by a factor of ~17 — and that is only the average: the last row of
each channel also runs the Doppler step for its plane, so its deaf
window is longer still. Even against a pre-measurement
estimate of 190 µs per row — derivable from the FFT harness's measured
76.3 cycles per Q15 butterfly — the ratio is 17 : 2. Nothing in this
calculation required hardware: the baud rate, the frame format, the
absence of a FIFO (RM0390) and a cycle estimate were all available in
advance.

The trap is arithmetic at the wrong granularity. On average, 365 µs of compute per row against 5.58 ms of transfer is a 6.5 % duty cycle, which looks
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
inexplicable mismatch. The index is informative too: with a continuous byte stream the first
deaf window would already have corrupted row 1, so the stream reaching
the board must have had gaps at first that happened to fall on those
windows — most likely while buffering in the ST-LINK's USB bridge was
still filling — and bytes were lost only once it ran continuously.

**The controlled experiment.** Analysis names a cause; it does not
exclude rivals — wire noise, a host-side timing bug or a row-indexing
error would also produce checksum failures. So instead of changing the board, the subscriber was slowed down:
`--row-pace 0.009` makes it sleep after each row,
leaving the wire silent exactly while the board computes. The *same
binary* then processed all 384 rows to a MATCH (31 detections, latency
8.53 s). One run, one variable, and the only surviving explanation is
that the board cannot listen while it computes.

**The fix: ping-pong reception.** The board now receives by interrupt into
two 516-byte buffers, A and B, which take turns. For each row:

1. Row *n* has finished arriving in buffer A.
2. The board immediately starts receiving row *n + 1* into buffer B. From
   then on the UART interrupt stores each arriving byte in B by itself,
   without the main loop's help.
3. Meanwhile the main loop checks row *n*'s checksum and runs
   `onepass_q15_row` on buffer A.
4. When B is full, the roles swap: B is processed while A receives.

Reception therefore never pauses while the board computes. Three details
make this work:

- **The row pointer follows the full buffer.** The kernel reads its row
  through `plan.rowbuf`, so that pointer is moved to whichever buffer just
  filled. The kernel overwrites the buffer while processing it, which is
  harmless: the buffer is not used for reception again until that
  processing has finished.
- **Two flags carry the news.** HAL's receive callbacks set `rx_done` when
  a row is complete and `rx_err` on a receive error, and the main loop
  waits on them. The main loop clears `rx_done` just before starting each
  reception and `rx_err` at the start of each frame, both at moments when
  the interrupt cannot be writing them. They are `volatile`, so the
  compiler reads them from memory every time.
- **Errors cancel reception first.** Before draining the wire after an
  `ERR`, the board stops the reception in progress with
  `HAL_UART_AbortReceive`; otherwise the interrupt and the blocking drain
  would both try to take the same bytes.

The interrupt reaches HAL through a hand-written `USART2_IRQHandler` in
`stm32f4xx_it.c`, inside CubeMX's `USER CODE` fences, which calls
`HAL_UART_IRQHandler`. The name must match exactly: otherwise the startup
file's default handler, an endless loop, runs instead.

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

Both builds came from the same kernel sources: the kernels last changed
on 2026-09-04, and every board build, as well as the session itself,
took place on 2026-09-06.

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

The SRAM total consists mainly of `planes` 98 304, the power/snapshot/spectrum
overlay 16 384, the cell list 1 536, the receive ping-pong pair 1 032,
`stdout` buffer 512, the plan struct 104, and the 6 144 B heap + stack
reservation — 124 016 B together; the remaining 940 B are the initialized
data section and HAL and C-library variables. Because every one of those is a `static` object inside a
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

Against the prediction of section 1.5, the linker's total of 124 956 B
came out 1 560 B higher than the predicted 123 396 B. The difference is
items the estimate did not list: the second receive buffer that
ping-pong reception added, with its two flags (528 B), the initialized
data section (476 B), about 450 B of HAL and C-library variables, and
the kernel's 104 B of state (pointers and counters). The margin of 6 116 B left the fit comfortable.

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

**Speed.** The board processes about one frame every 5.5 s, while the
publisher sends 20 frames per second, so this experiment makes no
real-time claim. Two costs set that pace, and neither is the Q15 row
pipeline itself:

- **The serial link.** Moving one frame's samples takes about 2.14 s
  (section 1.6). A real radar would not pay this cost, because its samples
  come from an ADC on the same board; here the UART stands in for that
  connection.
- **Double-precision DoA.** Each detection's 181-point Bartlett scan runs
  in double precision, which this single-precision FPU can only compute in
  software. Moving the scan to single precision should shorten the finish
  phase, but it was not tried, and its results could then differ slightly
  from the double-precision laptop chain, so the comparison rules would
  need revisiting.

**The reduced scenario.** The experiment runs at V = 12, C = 32, S = 128,
not the full 12 × 64 × 256 scenario, which halves the number of Doppler
bins (32 instead of 64) and coarsens the range bins from 0.075 m to
0.15 m. The one-pass design cannot reach the full size on this board: the
range-Doppler planes alone would need 393 216 B, three times all of its
SRAM (section 1.5). The full scenario needs a different design, which is
outside this experiment.

**The detection cap.** The CFAR cell list holds at most 64 cells. When
more cells pass, the weakest are dropped and the overflow flag is set
(section 1.3). In this session every frame matched the uncapped laptop
chain, so the cap never removed a detection, but a crowded scene would
lose its weakest ones.

**What the verification covers.**

- The board is compared with the laptop's chain, not with the scene. A
  MATCH shows that the board computes the same detections; it does not
  show that those detections are good. Detection quality — how reliably
  targets are found, how many false alarms appear — is outside this
  experiment, and the only check against the actual targets is the loose
  one in section 3.3.
- The evidence is one session: 110 frames of one scenario, with two
  targets moving at constant speed.
- The overlay is used only on the board (sections 3.2 and 3.3), so the
  board's MATCH results are the only evidence that its lifetime rule
  holds.
- Nothing detects a board built from older kernel sources than the
  library (section 2.4).

**Reception and reproducibility.**

- Row reception takes one interrupt per received byte. Receiving by DMA,
  which moves bytes into memory without the CPU, was not tried.
- The enlarged heap and stack reservations were edited directly in the
  linker script and are not recorded in the CubeMX project file.
  Regenerating the project would silently restore CubeMX's defaults,
  0x200 for the heap and 0x400 for the stack — a stack reservation below
  the measured peak of 1 328 B — and the link would no longer catch a
  stack that outgrows its reservation.
