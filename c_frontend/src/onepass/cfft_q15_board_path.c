/* CMSIS-DSP's 16-bit complex FFT, compiled on the laptop with the
 * arithmetic it has on the board.
 *
 * The library holds two versions of this FFT. A core with the DSP
 * instructions (the Cortex-M4 on the board) gets one written around them;
 * every other target gets plain C. The two round differently — the first
 * halves a sum as (a + b) >> 1, the second as (a >> 1) + (b >> 1) — so
 * their outputs differ by a count here and there, and a laptop reference
 * built the ordinary way would not match the board bit for bit.
 *
 * ARM_MATH_DSP is the macro that selects the version. The headers are
 * included first, without it, so that they define each DSP instruction as
 * a C function (dsp/none.h); it is then defined, and the two FFT sources
 * are compiled into this file. They take the board's version and run it
 * on those C functions. No CMSIS-DSP file is changed. */
#include "arm_math.h"
#include "arm_common_tables.h"
#include "arm_const_structs.h"

#ifdef ARM_MATH_DSP
#error "build this file for the laptop: the headers must not select the DSP version themselves"
#endif
#define ARM_MATH_DSP 1

#include "arm_cfft_radix4_q15.c"
#include "arm_cfft_q15.c"
