# Q15 FFT on an Arduino Uno

Runs the project's Q15 FFT, unmodified, on an ATmega328P, and measures what
it costs there beside the float transform.

Every other performance number in this project comes from one x86-64
laptop, where the float path vectorises, the fixed-point path does not, and
fixed point consequently *loses*. That is a real measurement and a
misleading one: a machine with a pipelined FPU is not the machine
fixed-point radar arithmetic exists for. The Uno is the opposite extreme —
8-bit registers, 16 MHz, no FPU, no barrel shifter, 2 KB of RAM. Measuring
the same kernel at both ends brackets the answer.

## Results

Measured at N = 64, both builds at `-O2`.

| | Q15 | float | ratio |
|---|---:|---:|---:|
| time per transform, Uno | 7974.4 µs | 17755.0 µs | **2.23x** |
| time per transform, host | 0.764 µs | 0.313 µs | **0.41x** |
| flash | 8366 B | 8064 B | |
| static RAM | 467 B | 725 B | |
| heap | 260 B | 388 B | |
| stack peak | 136 B | 139 B | |
| peak RAM of 2048 | 863 B | 1252 B | 0.69x |
| agreement with host | bit-exact | 2.09e-07 relative | |

**The sign flips.** Same source, same input, opposite verdict: Q15 wins by
2.23x on the Uno and loses by 2.4x on the laptop. The platform decides.

**2.23x is a floor, not the arithmetic's ceiling.** avr-gcc emits a libgcc
call, `__mulhisi3`, for every 16x16->32 signed multiply instead of inlining
it — four per butterfly. Around them, 32-bit work proceeds a byte at a time
and the `>>15` walks down one bit per instruction, the part having no barrel
shifter. Hand-written multiply assembly would move the number; by how much
is not worth guessing without measuring.

**Q15 is bit-exact across the two ISAs.** Twiddle table and both transforms
match the host count for count. This was the result most at risk: the
twiddle table is built in floating point, and avr-gcc makes `double` a
32-bit type, so a twiddle near a rounding boundary could have landed one
count off and propagated through every butterfly. At N = 64 it does not.
That exactness is what lets the fixed-point chain serve as a reference
model — develop on the laptop, trust the target. The float column cannot
make the same claim and does not: one ULP of float32 separates it from the
host, which is all IEEE 754 ever promised across compilers.

**Memory is the structural result.** 863 bytes against 1252, on a part with
2048. Extrapolating to N = 128, Q15 needs roughly 1375 bytes and still fits;
float needs roughly 2148 and does not. That is not a benchmark to be
trusted or doubted — it is arithmetic about where the wall sits.

## What this does not establish

- **Nothing about radar hardware.** An 8-bit AVR is not what FMCW front
  ends run on. The contested case is a Cortex-M4F, which has a
  single-cycle FPU *and* the `SMLAD` DSP instructions; nothing here speaks
  to it.
- **Nothing about the full pipeline.** The front end's working set is about
  1.2 MiB against 2 KB of RAM.
- **No real-time claim.** At 7974 µs per 64-point transform, one radar
  frame's transforms would take about 30 seconds against a frame period
  measured in tens of milliseconds.
- **No new precision figure.** Bit-exactness means the transform's
  precision here *is* what `evaluation/sqnr_fixed_point.py` measured on the
  laptop. This is a verification, not a measurement.

## Prerequisites

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=~/.local/bin sh
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli board list          # note the port; PORT defaults to /dev/ttyACM0
```

## Running it

```bash
make vectors                    # regenerate vectors.h from the simulator
make check-host ARITH=q15       # host reference, and its self-test
make flash  ARITH=q15           # compile and upload
make capture ARITH=q15          # read 20 s of serial into build/
make flash  ARITH=float         # repeat for the other column
make capture ARITH=float
make report                     # compare both against the host
```

`make check-host` asserts that the constant vector transforms to exactly
1000 in bin 0 and exactly zero in all 63 others. That pins the per-stage
halvings to exactly 1/n — one shift too many or too few and bin 0 comes
back at 500 or 2000 — and it catches a bad sync or a corrupt `vectors.h`
before anything is flashed.

## How it fits together

| file | runs on | role |
|---|---|---|
| `gen_vectors.py` | laptop | emits `vectors.h` from a scenario |
| `host/ref_host.c` | laptop | the same kernel, same protocol, to stdout |
| `uno_fft_q15.ino` | the Uno | loads, transforms, times, measures RAM, prints |
| `read_results.py` | laptop | parses both streams, compares, reports |
| `fft_q15.c` / `fft.c` | **both** | the kernel, compiled by gcc and by avr-gcc |

That last row is the design. One source file, two compilers, two machines,
two answer sheets — so a difference means something specific.

**The kernel is copied in, not referenced.** The Arduino toolchain compiles
only what is inside the sketch folder and its `src/`; it will not reach
across the tree. `make sync` copies the needed files from `c_frontend`
before every build and deletes the other arithmetic's, so the copies cannot
go stale and the unselected transform cannot be compiled in. `c_frontend`
remains the single source of truth, and nothing in it was modified for this
— which is what makes a number measured here a number about the shipping
code. The copies are gitignored.

**The host reference lives in `host/`** for the same reason in reverse: a
file in the sketch root would be handed to avr-gcc, and `clock_gettime`
does not exist there.

**The two arithmetics are separate builds**, selected by `-DARITH_Q15=1` or
`-DARITH_FLOAT=1`. Both working sets do not fit in 2 KB at once.

**Data is compiled into the image.** The Uno has no filesystem, so
`vectors.h` is the delivery format, marked `PROGMEM` — flash and RAM are
separate address spaces on AVR, and without it the tables would be copied
into RAM at startup, costing 512 bytes there is no room for.

**The sketch reports every three seconds** rather than once at startup.
Whether opening the serial port resets the board depends on the host tool,
and a report that has already scrolled past into a port nobody was reading
is unrecoverable. `read_results.py` keeps the last run that reached `END`
and discards the fragment at the front.

**Timing subtracts the reload.** `fft_q15_inplace` is destructive, so a
repeat has to reload its input first; re-running it on its own output would
be a different computation each pass. Both the copy-only and copy-plus-
transform figures are printed so the subtraction can be judged rather than
taken on trust.

**Peak RAM is measured, not computed.** Static RAM comes from the build,
but the heap and stack do not exist until run time. The sketch paints free
RAM with a canary byte at startup and afterwards scans for the first byte
still holding it, which gives the stack's high-water mark. The scan starts
at the heap's *current* end, not its start: by then `malloc` has overwritten
the painted bytes below that point, and counting them would report the heap
as stack.

## Notes for the next board

Most of this carries over unchanged. The sketch is the only real rewrite.

| file | reusable |
|---|---|
| `gen_vectors.py` | yes, unchanged |
| `vectors.h` | yes — the `__AVR__` shim already falls through to plain indexing, and on Cortex-M `const` lands in flash without an attribute |
| `host/ref_host.c` | yes, unchanged |
| `read_results.py` | yes, unchanged |
| the stream protocol | yes |
| `Makefile` | partly — `sync`, `host`, `check-host` and `report` carry; `build`, `flash` and `capture` are arduino-cli-specific |
| `uno_fft_q15.ino` | no |

The sketch's structure carries even so: load, transform, time with the reload subtracted, measure RAM, print the protocol. But four things inside the sketch are what do not travel: `micros()` becomes
`DWT->CYCCNT`, `Serial` becomes a UART or semihosting, the `PROGMEM`
accessors become plain indexing, and the stack painting keeps its idea but
changes its symbols — `__heap_start` and `__brkval` are avr-libc, where
newlib uses `end` and `_sbrk`. 

Two tooling problems cost time here and may recur:

- `arduino-cli monitor` emits nothing when its output is redirected to a
  file. `capture` reads the port with `stty` and `cat` instead.
- `Serial.println` sends CRLF, so a shell test for a final line must not
  anchor on `$` — the last line is `END\r`.

Sizing, if the scenario changes: at N = 64 the Q15 build uses 256 bytes for
the work buffer, 256 for the heap, and about 350 for Serial and object
overhead. N = 256 would want 1024 bytes of work buffer alone and does not
fit; N = 128 does, for Q15 only.
