# Q15 and float FFT on a NUCLEO-F446RE

Runs the project's `fft_q15.c` and `fft.c`, unmodified, on an STM32F446RE
— Cortex-M4F, 180 MHz, single-precision FPU, 128 KiB SRAM — and measures
what each costs there beside the host.

`embedded/uno_fft_q15/` measured the same kernel at the two extremes: a
laptop, where float vectorises and wins by 2.4x, and an 8-bit AVR, where
Q15 wins by 2.23x. Neither is what a radar front end ships on. This part
is: it has both a hardware FPU and the DSP instructions fixed-point
libraries are written around, so it is the one case where the answer was
genuinely open.

## Results

Measured at N = 256, both builds at `-O2`, 180 MHz from the ST-LINK's
8 MHz reference through the PLL, 5 flash wait states with prefetch and
both caches on. The board prints that configuration and the reader
refuses anything else.

| | Q15 | float | ratio |
|---|---:|---:|---:|
| cycles per transform | 78140 | 40870 | **0.52x** |
| cycles per butterfly | 76.3 | 39.9 | |
| time per transform, Nucleo | 434.11 µs | 227.06 µs | |
| time per transform, host | 3.773 µs | 1.773 µs | 0.47x |
| flash | 35824 B | 35604 B | |
| static RAM | 3500 B | 4524 B | |
| heap | 1552 B | 2064 B | |
| stack peak | 744 B | 832 B | |
| peak RAM of 131072 | 5796 B | 7420 B | 1.28x |
| agreement with host | bit-exact | 8.2e-08 relative | |

Cycles per butterfly, one kernel, three machines: Uno Q15 664, Nucleo Q15
76, Nucleo float 40; the host takes about 3.7 ns and 1.7 ns, at a clock
this project does not control.

**The sign flips back, as predicted.** The design note put on record,
before the run, that float would be 1.5–2.5x faster here; it is 1.9x. On a
Cortex-M4F, portable-C fixed point loses on time by nearly the same ratio
as on the laptop. Q15 keeps two things: it is bit-exact, and it needs
1.6 KiB less RAM at this size — half-width buffers, the same structural
result the Uno showed.

**Why: the compiler reached the FPU and did not reach the DSP unit.**
The float `fft_inplace` compiles to 87 lines with no calls, and its
butterfly is 18 VFP instructions: six `vldr`, four `vstr`, the complex
twiddle multiply as `vmul` + `vnmul` + two `vfma`, and two `vadd` and two
`vsub`. Multiply-and-add is one instruction with one rounding, and there
is no explicit rounding step at all. The Q15 `fft_q15_inplace` compiles to
207 lines, also with no library calls — the Cortex-M4 has a barrel shifter
and a single-cycle 32x32 multiplier, so none of the AVR's `__mulhisi3`
overhead survives — but the twiddle multiply is four plain `mul`/`mla`/`mls`
on `ldrsh`-sign-extended halfwords, and around them sit the six rounding
shifts per butterfly the format demands (`shr1_round` four times, twice
more inside the twiddle multiply). Not one `smulbb`, `smuad` or `smlad`
appears: the packed 16-bit multiply-accumulate that would do the whole
complex product in two instructions exists on this core, and `-O2` did not
use it. The fixed-point deficit here is a compiler-reach problem, the same
finding the x86 and AVR runs each made from their own direction, and
CMSIS-DSP's `arm_cfft_q15` — written around exactly those intrinsics — is
where the arithmetic's own ceiling would be measured.

**Q15 is bit-exact across the three ISAs.** All 128 twiddles and both
256-point transforms match the host count for count, as they did on the
Uno at 32 twiddles. The float column differs from the host by 8.2e-08
relative — about one float32 ULP, and traceable: the board's `vfma`
rounds once where the host's separate multiply and add round twice.

## What this does not establish

- **Nothing about CMSIS-DSP.** `arm_cfft_q15` is what fixed point looks
  like when written for this core; it is the deferred next column, and the
  only thing that would separate the arithmetic's cost from the
  compiler's.
- **Nothing about the pipeline.** One transform, one row. The front end's
  working set as written is ~1.57 MiB against 128 KiB; see the last
  section.
- **No real-time claim.** 434 µs per transform is a number about this
  kernel, not about a frame.
- **No new precision figure.** Bit-exactness means the transform's
  precision here is what `evaluation/sqnr_fixed_point.py` measured on the
  laptop. This is a verification, not a measurement.

## Prerequisites

```bash
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi openocd
sudo usermod -aG plugdev $USER      # then log in again, or `newgrp plugdev`
```

The three `arm-none-eabi` packages are one cross-toolchain split into
compiler, binutils and C library; newlib-nano is what CubeMX's link line
selects. openocd's udev rule (`60-openocd.rules`) ships installed on Ubuntu
and grants the ST-LINK to `plugdev` — a `LIBUSB_ERROR_ACCESS` means the
shell's user is not in that group yet. Check the board is seen with
`lsusb | grep 0483` before anything else; on one laptop it enumerated on
only one of the USB ports.

`gen_vectors.py` and `read_results.py` run with the project's Python
environment (`source .venv/bin/activate`), which `make` finds as
`python`; `make ... PYTHON=python3` overrides that.

STM32CubeMX (6.18.1, firmware package STM32Cube FW_F4 V1.28.3) is needed
only to regenerate `Drivers/` and the board-support sources from the
`.ioc`, since `Drivers/` is not committed: open `nucleo_f446re_fft.ioc`,
GENERATE CODE, then `mv Makefile.cubemx`'s newly generated sibling back —
CubeMX writes `Makefile`, this project keeps that file as
`Makefile.cubemx` and the wrapper as `Makefile`, so restore both from git
(`git checkout -- Makefile Makefile.cubemx`) after generating. The wrapper
never edits the generated one, so regenerating loses nothing.

## Running it

```bash
make flash  ARITH=q15 BRINGUP=1 # configuration-only image: the first thing to run on a fresh board
make capture ARITH=q15 BRINGUP=1
make vectors                    # regenerate vectors.h from the simulator
make check-host ARITH=q15       # host reference, and its self-test
make check-host ARITH=float
make flash  ARITH=q15           # compile and upload over ST-LINK
make capture ARITH=q15          # read 20 s of serial into build/
make flash  ARITH=float         # repeat for the other column
make capture ARITH=float
make report                     # compare both against the host
```

The bring-up capture should read `# clk 180000000`, `# fpu 1`,
`# flash_ws 5`, `# flash_acr 0x00000705`. `make check-host` asserts that
the constant vector transforms to a single exact value in bin 0 — 1000 for
Q15, whose per-stage halvings scale by 1/n, and 256000 for the unscaled
float kernel — and exactly zero in the other 255, which pins the scaling
and catches a bad sync or a corrupt `vectors.h` before anything is
flashed.

The instruction census the write-up quotes:

```bash
arm-none-eabi-objdump -d build/float/nucleo_f446re_fft.elf | awk '/<fft_inplace>:/,/^$/' \
    | grep -oE "\bv[a-z.0-9]+\b" | sort | uniq -c
arm-none-eabi-objdump -d build/q15/nucleo_f446re_fft.elf | awk '/<fft_q15_inplace>:/,/^$/' \
    | grep -oE "\b(smulbb|smuad|smlad|mul|mla|mls|asrs?|lsls?|ldrsh|strh)\b" | sort | uniq -c
```

The `>:` in the pattern matters: without it the range also opens at every
call site of the function and sweeps in the caller.

## How it fits together

| file | runs on | role |
|---|---|---|
| `gen_vectors.py` | laptop | emits `vectors.h` from a scenario |
| `host/ref_host.c` | laptop | the same kernel, same protocol, to stdout |
| `Core/Src/harness.c` | the board | loads, transforms, times, measures RAM, prints |
| `Core/Src/retarget.c` | the board | routes `printf` to USART2, the ST-LINK's virtual COM port |
| `read_results.py` | laptop | parses both streams, gates the configuration, compares |
| `fft_q15.c` / `fft.c` | **both** | the kernel, compiled by gcc and by arm-none-eabi-gcc |

That last row is the design, carried over from the Uno: one source file,
two compilers, two machines, two answer sheets. The kernel is copied in by
`make sync` before every build and the other arithmetic's copy deleted, so
the copies cannot go stale; `c_frontend` was not modified for this, and the
copies are gitignored.

**The board reports its configuration and the reader refuses a wrong
one.** `# clk` is `SystemCoreClock` after the PLL is up; `# fpu` is 1 only
if the hard-float flags reached the compiler *and* CPACR enabled the unit;
`# flash_ws` and `# flash_acr` are the wait states and cache bits in
effect. The chip boots at 16 MHz from its internal oscillator, and a
soft-float build is one missing flag away; either gives correct output and
numbers eleven or several times off, which nothing else would catch. The
first CubeMX generation of this project did in fact produce an HSI clock
tree while the GUI showed HSE; the `.ioc` committed here is the corrected
one.

**Makefile.cubemx is never edited.** The wrapper `Makefile` includes it and
appends the synced kernel, `harness.c`, `retarget.c`, the arithmetic
define, `-O2` in place of CubeMX's `-Og`, and `-u _printf_float` for
newlib-nano. Two things about the generated file are not obvious: its rules
depend on a file literally named `Makefile`, and its ELF rule fixes its
object list at the point it is read, so the wrapper restates
`$(ELF): $(OBJECTS)` after appending sources. Each arithmetic builds into
its own `build/<arith>/`, since the two share object names.

**Timing is in cycles**, from `DWT->CYCCNT` with interrupts off, the
input reload subtracted (the transform is destructive, and both raw
figures are printed so the subtraction can be judged); microseconds are
derived from `SystemCoreClock`.

**Peak RAM is measured.** Static RAM comes from `arm-none-eabi-size`
(flash = text + data, RAM = data + bss). Heap is the kernel's, read after
the plan is built and before the first `printf`, which allocates stdout's
buffer. Stack peak is the canary method from the Uno with newlib's symbols:
free RAM is painted from `sbrk(0)` upward at startup and scanned back from
the heap's current end afterwards.

**The board reports every three seconds**, and `read_results.py` keeps the
last run that reached `END`, for the same reason as on the Uno: whether
opening the port resets the board depends on the host tool.

## Notes for the next piece

The pipeline on this part does not fit as written — ~1.57 MiB of working
set against 128 KiB, with 126 KiB of it free after this harness — and the
restructure that makes it fit is its own piece of work: static storage in
place of `malloc`, a streaming range stage, one channel's range–Doppler
plane at a time accumulated into the power map, a cap on detections, and a
second pass over the input for the DoA snapshots. About 100 KiB at the
report's full scenario, which keeps the result verifiable against the
laptop.
