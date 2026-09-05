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

Populated after the first streaming session.
