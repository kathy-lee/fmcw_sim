# FFT comparison with CMSIS-DSP on a NUCLEO-F446RE

Compares the project's FFTs with their CMSIS-DSP counterparts in one binary.

CMSIS-DSP (Apache-2.0) lives in `embedded/third_party/CMSIS-DSP`, as a git
submodule pinned to tag `v1.18.0` (commit `4b6f2659`). Its license is in
`embedded/third_party/CMSIS-DSP/LICENSE`. The source is compiled unmodified.

## Steps of implementation

### Step 1: Get the library

After cloning this repository, from the repository root:

```bash
git submodule update --init embedded/third_party/CMSIS-DSP
git -C embedded/third_party/CMSIS-DSP describe --tags   # expect v1.18.0
```

`Drivers/` is not committed either. Copy it from a sibling harness that has it,
or generate it by opening `nucleo_f446re_fft.ioc` in STM32CubeMX and restoring
`Makefile` and `Makefile.cubemx` from git afterwards. Generated from CubeMX
6.18.1 with firmware package FW_F4 V1.28.3.

The `Makefile` builds these CMSIS-DSP files, all under `Source/`:

| Mode | Files |
|---|---|
| q15 | `TransformFunctions/arm_cfft_q15.c`, `TransformFunctions/arm_cfft_radix4_q15.c` |
| float | `TransformFunctions/arm_cfft_f32.c`, `TransformFunctions/arm_cfft_radix8_f32.c` |
| both | `CommonTables/arm_const_structs.c`, `CommonTables/arm_common_tables.c`, `TransformFunctions/arm_bitreversal2.c` |

### Step 2: Generate the vectors

`vectors.h` holds the FFT inputs and is committed, so this is only needed after
changing `gen_vectors.py` or its scenario. From this folder:

```bash
python gen_vectors.py --self-check
```

It prints one `self-check passed` line for 128 samples and one for 32, then
`wrote .../vectors.h`.

### Step 3: Build

From this folder:

```bash
make distclean
make build ARITH=q15
make build ARITH=float
```

`make distclean` removes both builds and the kernel copies. Use it after
changing the submodule version, because make does not track that change.
Each successful build prints a size table.

### Step 4: Flash and capture

With the board connected, from this folder:

```bash
make flash ARITH=q15 && make capture ARITH=q15
make flash ARITH=float && make capture ARITH=float
```

Each capture prints `captured build/stream_<arith>_nucleo.txt`. `PORT` defaults
to `/dev/ttyACM0`.

### Step 5: Read the results

```bash
make report
make map ARITH=q15
make map ARITH=float
```

`make report` lists any problems, then the cycles, output differences and memory
for each mode. A scale problem is a finding about the two FFTs: the results are
still shown, but the differences are not comparable until it is resolved. A
wrong clock, FPU or loop-unrolling setting refuses the capture. `make map` shows
what each CMSIS-DSP function and table adds to flash.

If `python` is not found, run `make report PYTHON=python3`.

Example output of `make report`:

```text
NUCLEO-F446RE FFT, project vs CMSIS-DSP

q15:
      n     own cyc   cmsis cyc  cmsis/own    max diff    rms diff
    128       35048        9660       0.28           8     2.40036
     32        6868        1911       0.28           6     2.33519
  flash 39176 B, static RAM 3684 B, heap 992 B, stack peak 888 B

float:
      n     own cyc   cmsis cyc  cmsis/own    max diff    rms diff
    128       18398       12706       0.69   0.0737305  0.00655588
     32        3646        2748       0.75   0.0117188  0.00242437
  flash 41268 B, static RAM 4836 B, heap 1312 B, stack peak 888 B
```

Example output of `make map ARITH=q15` (address, size in hex, type, symbol):

```text
0800940c 00000010 R arm_cfft_sR_q15_len128
0800941c 00000010 R arm_cfft_sR_q15_len32
0800950c 00000030 R armBitRevIndexTable_fixed_32
08008738 00000050 T arm_bitreversal_16
080096bc 00000060 R twiddleCoef_32_q15
080080ec 000000c2 T arm_cfft_radix4by2_inverse_q15
08008028 000000c2 T arm_cfft_radix4by2_q15
080081b0 000000d6 T arm_cfft_q15
0800942c 000000e0 R armBitRevIndexTable_fixed_128
0800953c 00000180 R twiddleCoef_128_q15
080084e0 00000256 T arm_radix4_butterfly_inverse_q15
08008288 00000256 T arm_radix4_butterfly_q15
```

Example output of `make map ARITH=float`:

```text
08009824 00000010 R arm_cfft_sR_f32_len128
08009834 00000010 R arm_cfft_sR_f32_len32
08008b48 00000052 T arm_bitreversal_32
080099e4 00000060 R armBitRevIndexTable32
08009e44 00000100 R twiddleCoef_32
08009844 000001a0 R armBitRevIndexTable128
08009a44 00000400 R twiddleCoef_128
0800861c 0000052a T arm_radix8_butterfly_f32
08007f18 00000704 T arm_cfft_f32
```

## Results

Measured on the board at 180 MHz, `-O2`, with CMSIS-DSP v1.18.0 and
`ARM_MATH_LOOPUNROLL`.

### Timing

Cycles per transform, on the radar vector:

| N | own q15 | CMSIS q15 | own float | CMSIS float |
|---:|---:|---:|---:|---:|
| 128 | 35048 | **9660** | 18398 | 12706 |
| 32 | 6868 | **1911** | 3646 | 2748 |

CMSIS-DSP takes 0.28x the cycles of the project's FFT in q15, and 0.69x (128) and
0.75x (32) in float. The ranking also flips: among the project's FFTs float is
faster than q15 by about 1.9x, while among the CMSIS-DSP FFTs q15 is faster than
float by about 1.3 to 1.4x.

### Precision

How far the outputs differ. Both FFTs transform the same radar chirp, and each
output's real and imaginary parts are compared element by element: max is the
largest single difference, RMS the typical one. Peak is the largest output value,
for scale. q15 values are output integer counts; float outputs are not scaled
down, hence their size.

| Mode | N | peak (own FFT) | max diff | RMS diff | max / peak |
|---|---:|---:|---:|---:|---:|
| q15 | 128 | 8961 | 8 | 2.40 | 0.09% |
| q15 | 32 | 9148 | 6 | 2.34 | 0.07% |
| float | 128 | 1146910 | 0.0737 | 0.00656 | 6e-6% |
| float | 32 | 292702 | 0.0117 | 0.00242 | 4e-6% |

The q15 FFTs round differently at each stage, so they differ by a few counts.
The float differences are at the level of normal float rounding.

The DC check found no scale mismatch in either mode. These figures show how far
the two disagree, not which is more accurate, so the effect of CMSIS-DSP's
rounding on detection still has to be checked before the front end uses it.

### Memory

For the image that holds both FFTs at both sizes:

| | q15 | float |
|---|---:|---:|
| image flash | 39176 B | 41268 B |
| of which CMSIS-DSP code and tables | 2662 B | 5024 B |
| static RAM, including harness buffers | 3684 B | 4836 B |
| heap, the project FFT's tables | 992 B | 1312 B |
| stack peak | 888 B | 888 B |

CMSIS-DSP keeps its tables in flash, so it needs none of that heap. In q15, 792 B
of its flash is the inverse transform, which is linked but never called, because
`arm_cfft_q15` chooses the direction at run time.

### Conclusion

Note these numbers above cover the FFTs alone, not the front end's timing.

**Verdict on `arm_cfft_q15` for the front end:** its speed supports using it, at
3.6x the project's q15 FFT for 2.7 KB of flash. It cannot replace `fft_q15`
unchecked, because its outputs are not bit-exact, differing by up to 8 counts
here. Detection results have to be compared on the same frames first.

## Tests

The reader's tests need no board or firmware build. From the repository root:

```bash
pytest tests/test_nucleo_fft_cmsis_reader.py -v
```

## How to change the library version

From the repository root, replacing `<tag>`:

```bash
git -C embedded/third_party/CMSIS-DSP fetch --tags
git -C embedded/third_party/CMSIS-DSP checkout <tag>
git add embedded/third_party/CMSIS-DSP
git commit -m "fix(third_party): pin CMSIS-DSP submodule to <tag>"
```

The repository stores only the submodule's commit hash. Without the `git add`
and commit, a fresh clone still gets the old version. Update the tag and commit
named at the top of this file as well, then rebuild as above.
