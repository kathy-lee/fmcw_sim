# One-pass detection front end on the NUCLEO-F446RE

The full Q15 detection front end — range-Doppler, NCI, CA-CFAR, Bartlett
DoA — running on the Nucleo, fed real simulator frames over the ST-LINK
VCP at 921600 baud by `simulator/streaming_subscriber_nucleo.py`, which
also diffs every frame's detections live against the laptop's frozen
reference chain. Rows are processed as they arrive: the raw frame
(196 608 B, larger than the part's RAM) never exists on the board.

The project is a clone of the measured `nucleo_f446re_fft/` baseline.
Deliberate deltas from the harness: baud 921600 (a frame would take ~17 s
at 115200), linker reservations heap 0x800 / stack 0x1000 (fail the link,
not the night), and the target name.

## Prerequisites

- `arm-none-eabi-gcc`, `openocd` (same as the harness).
- STM32CubeMX: open `nucleo_f446re_frontend.ioc`, confirm USART2
  asynchronous at **921600**, Toolchain = Makefile, Generate Code; then
  `mv Makefile Makefile.cubemx`. This regenerates `Drivers/` and the
  build rules from the committed `.ioc` — neither is committed.
- The exported parameters and frames:

  ```bash
  python scripts/export_for_c.py \
      --scenario scenarios/scenario_mimo_stream_128.json \
      --out data/c_export_stream128 --frames 5
  ```

## Targets

```
make sync        # copy kernel sources/headers in, render frontend_params.h
make build       # sync + compile, prints arm-none-eabi-size
make flash       # build + openocd program/verify/reset
make map         # the static regions against the linker map, largest first
make distclean   # remove build output and every synced copy
```

The sync discipline is the harness's: the kernel sources under
`c_frontend/` are the single source of truth, copied in unconditionally
before every build so the copies cannot go stale — one source, two
compilers (host gcc for the referee tests, arm-none-eabi-gcc here). The
copies and the generated `frontend_params.h` are gitignored.

## Wire protocol

One authority: the framing constants and parsers at the top of
`simulator/streaming_subscriber_nucleo.py`, which the board program
implements exactly. In short: 14-byte frame header (`OPF1`, frame id,
V C S), then V·C rows v-major, each 512 B of int16 I/Q plus a 2-byte
additive checksum; the board answers with a text block per frame
(`DET`/`D` lines, `TIME`, `MEM`, `SAT`, `END`) and paces the link with
`RDY` — the feeder only transmits after reading one.

## Measured results

One certified session (2026-09-06): publisher at 20 fps, 110 frames
processed end to end against the live reference diff.

- **Verdicts: 110/110 MATCH, 0 mismatches, 0 tie flips.** Every count,
  cell, azimuth (grid-exact), coordinate and SNR agreed with the
  laptop's frozen chain on the identical quantized input; `SAT 0`
  throughout.
- **Cadence ~0.18 frames/s** (~5.5 s per processed frame): 11 959
  publisher frames dropped ≈ 109 per processed, by design — the
  subscriber always takes the newest frame. Latency publish→verdict
  4.6–7.2 s, scaling with detection count.
- **Compute:** row phase 25.6 Mcyc/frame (142 ms at 180 MHz), hidden
  entirely inside the ~2.1 s transfer by ping-pong interrupt reception.
  Finish phase 0.86–1.24 s summary means (~2.3 s at 31 detections):
  one 181-point Bartlett scan per detection in double precision, which
  this FPU (single-precision) emulates in software.
- **Memory:** static `data`+`bss` 124 956 of 131 072 B (margin
  6 116 B) — planes 98 304, the power/snapshot/spectrum overlay
  16 384, cell list 1 536, rx ping-pong pair 2×516. Heap high-water
  1 328 of the 2 048 reserved; stack peak 1 328 of 4 096 — both
  constant across all 110 frames (board-measured, `heap<=`/
  `stack_peak<=` watermarks in the subscriber summaries).

The receive design earned its shape on hardware: this UART has no FIFO
(two bytes of slack at ~11 us/byte), so blocking reads corrupted rows
within milliseconds at full wire speed. Reception is therefore
interrupt-driven into one buffer while the kernel processes the other;
`--row-pace` on the subscriber remains as the one-run diagnostic (a
paced wire is silent during the board's compute gaps — if that cures
checksum errors, the board is deaf between rows) and as a no-reflash
fallback.

Cadence is UART-and-finish bound and no real-time claim is made.
Natural next experiments on this baseline: DMA reception, and the DoA
scan in hardware single precision.

Reproduction (two terminals):

```bash
python -m simulator.streaming_publisher \
    --scenario scenarios/scenario_mimo_stream_128.json
python -m simulator.streaming_subscriber_nucleo \
    --params data/c_export_stream128/params.cfg \
    --port /dev/ttyACM0 --baud 921600 --max-frames 110 --summary-every 10
```
