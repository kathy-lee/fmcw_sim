# 1.The CFAR cell walk

```c
int cfar_cells_q15(const int64_t *power, const FrontendParams *p,
                   CfarCellQ15 *cells, int max_cells, int *overflow);
```

The frozen CFAR answers "is this cell a detection?" with a full 0/1 mask the
size of the map. The cell walk asks the same question with the same
arithmetic, but writes only the answers that matter — a short list of passing
cells — so it needs *zero* scratch memory. Two pictures: what actually
changes against the frozen path, and how each passing cell fights for a place
in the capped list.

## What actually changes

![Frozen mask CFAR vs the cell walk: same power map in, identical threshold
decisions; ~84 KiB of sliding-sum scratch and a full mask versus zero scratch
and a 1.5 KiB ranked list](figures/cfar-frozen-vs-walk.svg)

Both sides implement the identical CA-CFAR decision — geometry, edge
clamping, Q16 threshold — on the same power map. The frozen path buys
O(1)-per-cell speed with four map-scale scratch arrays and emits a full mask;
the cell walk recomputes each window in place, stores nothing between cells,
and emits only the passing cells with their evidence. ≈86 KiB of working
memory becomes 1.5 KiB — the difference that fits the pipeline onto the
board.

Sizes in the figure are at the shrunk scenario's geometry (C=32, R=64, guard
(2, 4), training (4, 8) → window half-extents 6 × 12): padded copy
44 × 88 × 8 = 30 976 B, row sums 44 × 64 × 8 = 22 528 B, full-window and
guard-window sums 2 × 32 × 64 × 8 = 32 768 B.

## The walk and its ranked cap

![Decision flow of the cell walk: min-range gate, direct window sums, the
threshold test, append or ranked replacement into the capped list, final sort
by scan order](figures/cfar-cell-walk-flow.svg)

Most cells exit at the threshold and cost nothing. A passing cell is appended
while there is room; once the list is full, a newcomer must beat the weakest
kept cell's margin — `margin = (power << 16) − thresh_q16` — to enter, and
the spill is reported through `overflow` instead of silently losing the
strongest targets. The final sort restores the scan order a mask walk would
have produced, so downstream comparisons are element-by-element whenever
nothing overflowed.

## Why it looks this way

- **Zero scratch:** the frozen production CFAR keeps four map-sized int64
  arrays of sliding sums; the walk recomputes each window directly, which is
  O(window) per cell — slower in theory, irrelevant here, and the memory it
  saves is what lets the pipeline fit the board.
- **Captured values:** each kept cell stores its `power` and `thresh_q16` at
  decision time, because the power map's storage is reused (overlaid) later
  in the frame — the list must survive on its own.
- **Ranked, not truncated:** a plain "stop at 64" cap would keep whichever
  detections came first in scan order; the margin ranking keeps the
  strongest, and `overflow` tells the host the frame spilled.

Equivalence with the frozen path is pinned by
`c_frontend/tests/test_cfar_cells_q15.c`: the cell set equals the reference
mask's 1-set exactly (both directions), the stored threshold reproduces each
decision, and the ranked cap provably keeps the top-N by margin.

---

# 2.The row-streaming range stage

```c
int  rd_stream_q15_plan_init(RdStreamPlanQ15 *p, int n_chirps, int n_samples);
void rd_stream_q15_row(RdStreamPlanQ15 *p, cplx_q15 *row, int c,
                       cplx_q15 *plane, long long *sat);
void rd_stream_q15_plane_finish(RdStreamPlanQ15 *p, cplx_q15 *plane,
                                cplx_q15 *col, long long *sat);
```

Where the cell walk streams its *output* (a list instead of a mask, nothing
materialized in between), this stage streams its *input*: rows are consumed
the moment they arrive off the wire, so the raw frame — 196 608 B at the
shrunk scenario, one and a half times the board's entire RAM — never exists
anywhere. It is the frozen `rd_q15_process_channel` re-cut at the row
boundary: the same operations in the same order, split into "what can happen
per row" and "what needs the whole channel".

## The path of one row

![Each arriving row is processed in place — DC removal, Hann, range FFT —
and its kept half lands in one row of the resident plane; when a channel's
last row arrives, the plane finish runs the slow-time work; the raw frame
never exists](figures/rd-stream-row-path.svg)

The split follows the data dependencies exactly:

- **Per row (fast time)** — DC removal, range Hann, range FFT, keep the
  S/2 positive-beat bins. Each of these reads only the one row, so it can
  run the moment the row lands. The arriving row buffer doubles as the FFT
  workspace (the frozen stage used a private scratch copy; in-place produces
  the same values in the same order, so the outputs stay bit-identical).
- **Per plane (slow time)** — the clutter (slow-time mean) removal and the
  Doppler pass each read a whole *column* of chirps, so they cannot run
  until the channel's last row has arrived. `rd_stream_q15_row` at
  `c == C−1` is the trigger; the plane finish then walks the range bins,
  reusing the row buffer as its column workspace (C·4 = 128 B of its 512 B).

On the board this timing is what hides the compute: one row's fast-time work
costs roughly 190 µs while the next row needs ~5.6 ms to cross the UART, so
the range stage rides inside the transfer time for free.

## Plan owns tables, caller owns data

`RdStreamPlanQ15` holds only what is frame-independent — two FFT plans and
two Hann windows, built once on the heap at startup (~1.4 KiB). Every byte
of frame storage (the planes, the row buffer) belongs to the caller: static
arrays on the board, ordinary allocations in host tests. One plan serves all
12 channels, because nothing per-channel lives in it.

Equivalence with the frozen path is pinned by
`c_frontend/tests/test_rd_stream_q15.c`: the same random channel is pushed
through `rd_q15_process_channel` whole and through the streamed pair row by
row, and the two output planes must be `memcmp`-identical — including the
saturation counters — across multiple trials. Anything but bit-equality
would mean the re-cut changed the operation order, not just the storage.

---

# 3.The tableless single-snapshot Bartlett DoA
某个 detected RD cell
        │
        │ 取所有 V 个 virtual channels
        ▼
snapshot[V]
        │
        ▼
对 181 个角度做 Bartlett
        │
        ▼
spectrum[181]
        │
        ▼
找 local peaks
        │
        ▼
选最强 peak
        │
        ▼
返回 1 个 DOA

```c
int doa_bartlett_cell(const cplx *snap, const double *pos_wl, int n_elem,
                      double *spectrum);
```

The frozen DoA precomputes every steering vector at init — a 181 × V table
of complex floats, 17 376 B on the heap at V = 12 — and reads it back on
every use. The tableless form evaluates the *identical expression* at the
point of use instead, so the table never exists. Its cost moves from RAM to
compute: 181 × V sin/cos pairs per call, which is the right trade where
detections are few per frame and RAM is the wall.

## The estimator

One detection gives a snapshot `x[e]` — the complex value of that
(doppler, range) cell read across the V channel planes. The angular grid is
θ_g = (g − 90)° for g = 0…180, and the array element positions `p_e` are in
wavelengths.

**Steering vector** — the phase pattern a plane wave from θ_g imprints on
the array:

```
a_g[e] = exp( j · 2π · p_e · sin θ_g ),      e = 0 … V−1
```

**Bartlett spectrum** — correlate the snapshot with each candidate angle's
steering vector, take the squared magnitude:

```
              | V−1                     |²
P(θ_g)  =     | Σ   conj(a_g[e]) · x[e] |
              | e=0                     |
```

If the wave truly comes from θ_g, then x[e] ≈ s·a_g[e]: the conjugate
cancels every phase, all V terms add in phase, and P(θ_g) ≈ V²·|s|² — a
peak. At wrong angles the terms point in different directions in the
complex plane and partially cancel.

**Estimate** — spectral-maxima selection: strict local maxima of P over the
grid (one-sided rule at the edges, whole-grid argmax as fallback on a flat
spectrum), the strongest maximum wins, ties to the lower index. The azimuth
is the winning index minus 90 degrees.

## Where the table went

![Frozen DoA vs the tableless cell: the frozen path fills a 17 KiB steering
table at init and reads it per use; the tableless path computes the identical
expression at the point of use and stores nothing; both feed the same
accumulation, so the spectra are bit-identical](figures/doa-cell-tableless.svg)

The whole equivalence rests on one expression. The frozen init writes

```c
p->steer[(size_t)g * n_elem + e] = (cplx){(float)cos(ph), (float)sin(ph)};
```

and the tableless form computes, at the moment of use,

```c
cplx a = { (float)cos(ph), (float)sin(ph) };
```

with `ph` built from the same doubles. Same libm calls, same float casts,
same order — every steering element comes out bit-identical, which means
the table only ever cached values this line can regenerate. The
accumulation is shared arithmetic from there: `conj(a)·x` summed in
doubles, so both paths square and compare the exact same bits.

## Why it looks this way

- **No plan at all:** unlike the other two primitives there is nothing
  frame-independent worth keeping — no FFT twiddles, no windows — so there
  is no init/free pair. The caller passes positions and scratch; the
  function is pure.
- **Caller-owned spectrum:** the 181-double spectrum (1 448 B) is caller
  scratch, not a static — on the board it overlays storage the power map no
  longer needs by the time DoA runs.
- **Fixed at one source:** the frozen estimator can return several maxima;
  the streaming pipeline runs with `doa_n_sources = 1`, so the cell form
  bakes in top-1 and returns the grid index directly — no output array, no
  count.

Equivalence with the frozen path is pinned by
`c_frontend/tests/test_doa_cell.c`: 50 random 12-element snapshots through
`doa_plan_init` + `doa_bartlett` and through `doa_bartlett_cell`, asserting
the selected index matches exactly and the two 181-point spectra are
`memcmp`-identical — bit-equality on doubles, not a tolerance.

---

# 4.The one-pass composition

```c
int  onepass_q15_init(OnepassPlanQ15 *p, const FrontendParams *prm,
                      cplx_q15 *planes, int64_t *power, CfarCellQ15 *cells,
                      cplx_q15 *snapshots, double *spectrum, cplx_q15 *rowbuf);
void onepass_q15_row(OnepassPlanQ15 *p, int v, int c);
int  onepass_q15_finish(OnepassPlanQ15 *p, OnepassEmit emit, void *ctx);
```

The three primitives meet here, and the frozen four-stage chain becomes two
calls: `onepass_q15_row`, made once per arriving row, and
`onepass_q15_finish`, made once per frame. The composition adds no new
signal processing — its design content is *scheduling* (what runs when) and
*storage* (what lives where, for how long).

## Algorithm structure: split by data dependency

Each piece of the pipeline runs at the earliest moment its inputs exist:

- **Per row** — the range FFT needs only the one row in `rowbuf`, so it
  runs on arrival; the 64 kept bins land in row `c` of channel `v`'s
  resident plane. The RD map is never computed "at once": it accumulates,
  one range spectrum per call.
- **Per plane** — the Doppler pass needs a channel's full slow-time axis,
  which exists exactly when that channel's last chirp lands; the
  `c == C−1` call runs it. After 384 row calls, all 12 RD maps are done.
- **Per frame** — NCI needs all 12 planes, so it and everything after it
  (CFAR walk, snapshot gather, per-detection TDM + Bartlett + Cartesian)
  wait for frame end, inside `finish`.

On the board this split is what buys the throughput: a row's range work
(~190 µs) hides inside the ~5.6 ms the UART needs to deliver the next row,
so when the last row lands, the FFT-heavy majority of the frame's compute
has already happened. Only the detection tail appears as frame-end latency.

Detections leave through a callback, one at a time, in the same
(doppler, range) order a mask walk would emit — there is no result array.
`n_cells` is set before the first emit, so a caller whose wire format
prints a count first can. The emit shape means a caller needs storage for
*one* detection, not `MAX_DET` of them.

## Memory structure: lifetimes and the overlay

![Frame storage across one one-pass frame: planes live throughout, the row
buffer only while rows arrive, the cell list from the CFAR walk on; the
power map is live only from NCI to the CFAR walk, and the snapshots and
spectrum reuse its storage after](figures/onepass-frame-memory.svg)

Every byte of frame storage is caller-provided; the plan owns only FFT
tables and windows (~1.4 KiB, heap, built once). The lifetimes are part of
the contract, not an accident:

- **`planes` (98 304 B)** — the irreducible core: resident all frame,
  because the DoA snapshots must read any (doppler, range) cell across all
  12 channels after detection.
- **`rowbuf` (512 B)** — one buffer, three jobs in sequence: the arriving
  row, the range FFT's in-place workspace, and the Doppler pass's column
  scratch.
- **`power` (16 384 B)** — born at NCI, dead the moment the CFAR walk
  returns; each kept cell carries its own `power` and `thresh_q16` copies
  precisely so nothing needs the map afterwards.
- **`snapshots` (3 072 B) and `spectrum` (1 448 B)** — needed only after
  `power` dies, so a tight caller may overlay both onto its storage
  (disjoint ranges, snapshots first). `finish` guarantees the ordering:
  every snapshot is gathered before the spectrum is first written.

With the overlay, static frame storage totals ≈114 KiB of the board's
128 KiB. The raw frame (196 608 B) and the frozen CFAR scratch (~86 KiB)
— the two things that could never fit — are exactly what the streaming
input and streaming output primitives removed.

Equivalence with the frozen chain is pinned by
`c_frontend/tests/test_onepass_q15.c`: noise plus a strong coherent target
runs through the frozen four stages and through the composition fed row by
row; the planes must be `memcmp`-equal to the frozen cube, saturation
counts equal, and every emitted detection field bit-equal — doubles
compared with `==`, because both sides run identical operations in
identical order.

---

# 5.The referee chain

How the board gets proven is itself a design. Every new primitive was
accepted only by exact equality against a frozen counterpart on the host —
and that discipline is what lets a running board be judged frame by frame.

![One complex64 frame feeds both the quantizer, whose bytes cross the UART
to the board, and the frozen Q15 chain on the host; the bridge compares
the two results every frame, while the one-pass host path was proven
bit-equal to the frozen chain once, offline](figures/onepass-referee-chain.svg)

Three ideas carry the structure:

- **Cast once, feed both.** The simulator's complex128 frame is cast to
  complex64 a single time; that one array goes into `fmcw_q15_quantize`
  (producing the bytes the wire carries) and into the frozen
  `q15_process_frame` (producing the reference detections). The two sides'
  inputs are identical by construction, so any output difference belongs
  to the board.

- **Referee against the frozen end, trust through the middle.** The live
  comparison targets the frozen chain — the oldest, most-tested artifact —
  not the one-pass host path. The one-pass host path earned its trust
  once, offline, through bit-exact parity tests; that proven link is what
  lets a live MATCH vouch transitively for the board's one-pass code.

- **Grade each field by its provenance.** Bit-exactness holds wherever
  both sides run the same operations on the same implementation — which
  ends at the UART, because the board links a different libm
  (newlib-nano) than the host (glibc), and sin/cos/log10 may differ in
  the last bit. So detection counts and the (doppler, range) bins, pure
  integer arithmetic, must match exactly; the azimuth must sit on the
  same grid point, with a one-step neighbor counted as a tie flip (a
  last-bit spectrum difference flipping a near-tie) rather than a
  failure; and x, y, snr_db — floats through libm — compare to 1e-6
  relative. A tolerance appears exactly where a shared-implementation
  assumption ends, and nowhere else.

Expectation of comparison with referee chain:
the board comparison is not bit-exact like every comparison so far, because one assumption finally breaks: same libm. All our host-side ==-on-doubles checks were legitimate because both sides called the same glibc sin/cos/log10. The board links newlib-nano's libm, a different implementation whose results can differ in the last bit. Everything downstream of a libm call inherits that. So compare_detections grades each field by its provenance:

- Counts and bins (doppler_bin, range_bin) — exact. These come from pure integer arithmetic (Q15 FFTs, int64 CFAR); no libm touches them, so any difference is a real bug.
- Azimuth — grid-exact, but a one-step neighbor is counted separately as a tie flip: a last-bit spectrum difference can flip which of two nearly-equal grid points wins. Flips are tallied and reported rather than failed.
- x, y, snr_db — relative tolerance 1e-6: floats through cos/sin/log10, so agreement to ~single-precision depth is the honest expectation.

---

# 6.The mirrored API layers

The one-pass code reaches Python through the same two layers the frozen
code does — a flat FFI file and a wrapper class — and each new layer is
written as a deliberate mirror of its frozen sibling:

```
processing/c_frontend.py   class CFrontend   ~   class CFrontendOnepass   processing/c_frontend_onepass.py
c_frontend/src/fmcw_api.c                    ~    c_frontend/src/onepass/fmcw_onepass_api.c
```

![Three mirrored layers — wrapper class, flat FFI file, C kernels — where
each pair shares its conventions and differs in exactly one thing: the
call granularity, whole-frame versus
row-fed](figures/onepass-mirrored-layers.svg)

## What is mirrored: the conventions

`fmcw_onepass_api.c` copies `fmcw_api.c`'s boundary rules one for one: an
opaque `void *` handle from a `create(params_cfg)` that returns NULL on
failure; `destroy` safe on NULL; `dims(h, int[3])`; caller-allocated
output buffers with a count-or-−1 return; no structs across the boundary;
even the same 7-column detection layout, so the two paths' outputs are
directly comparable.

`CFrontendOnepass` copies `CFrontend`'s wrapper idioms the same way: one
module-level `_load()` declaring every `argtypes`/`restype` once, the
`_ptr` helper, a class owning the handle, `close()` safe to call twice,
context-manager and `__del__` cleanup, validate-then-call with C's −1
turned into a Python exception.

The payoff is navigational: a reader who knows one file of a pair knows
the other's shape already, and anything that *doesn't* match signals a
real design difference, not a stylistic accident.

## What deliberately differs: the surface

- **Call granularity — the one difference that repeats at every layer.**
  The frozen surface is whole-frame (`q15_process_frame(adc)`) with
  per-stage entries (`rd`, `nci`, `cfar`, `pointcloud` and their Q15
  twins). The one-pass surface is the streaming pair — `row` × 384, then
  `finish` — because the row-fed shape is the design under test; a
  whole-frame entry would bypass exactly what needs exercising.
- **No per-stage functions on the one-pass side.** The frozen chain's
  intermediates (RD cube, power map, mask) are stable frame-level
  artifacts worth handing out. The one-pass intermediates are not: the
  planes mid-accumulation and the overlaid power map are transient by
  design.
- **The quantizer moves outside the handle.** In `fmcw_api.c`,
  quantization is the integer path's first stage, hidden inside
  `q15_process_frame`; the caller sees only saturation counters. The
  one-pass side exports the stateless `fmcw_q15_quantize` because its
  caller — the bridge — needs the quantized bytes *as data*: they are the
  payload the wire carries. Correspondingly, `CFrontendOnepass.quantize()`
  returns the int16 array and records `last_sat`, where `CFrontend`
  queries counters buried in C state.

Both `.c` files compile into the same `libfmcw.so` (membership in the
Makefile's `SRC` list is membership in the library), so the two wrapper
classes are two doors into one binary. `fmcw_onepass_api.c` is also the one
file under `src/onepass/` that is host-only glue: the board compiles the
composition and calls `onepass_q15_row`/`finish` directly on static
storage — it needs no FFI, no heap, and no handle.

# 7.the banner-only board program

scenario JSON 
──▶ export_for_c.py (you, once,tasks ago)
──▶ data/c_export_stream128/params.cfg (sits on disk,still valid)
──▶ gen_frontend_params.py (automatic, inside every `make sync`)
──▶ Core/Inc/frontend_params.h
                                                     
                                                         
1. Toolchain — arm-none-eabi-gcc builds the cloned project at all (link succeeds with the new heap/stack reservations).
2. Clock — # clk 180000000 shows the copied SystemClock_Config still yields 180 MHz.
3. FPU — # fpu 1 shows it's both compiled-for and enabled.
4. Flash wait states — # flash_ws 5 shows the latency matches 180 MHz (wrong latency = subtle random faults).
5. Params header — the # dims / # cfar lines are printed from the compiled FRONTEND_PARAMS struct, so gen_frontend_params.py → frontend_params.h → compiler round-trip demonstrably worked.
6. Float printf — %.17g output proves newlib-nano's float formatting is linked (-u _printf_float), which the D detection lines will depend on.
7. Baud 921600 — the banner arriving readable proves the ST-LINK VCP and your laptop's serial driver both actually sustain that rate (the harness only ever ran 115200).