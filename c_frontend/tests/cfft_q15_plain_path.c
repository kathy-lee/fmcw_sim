/* CMSIS-DSP's 16-bit complex FFT in its plain C version, under other
 * names, so that a test can run it beside the version the board has
 * (src/onepass/cfft_q15_board_path.c) and see that the two are not the
 * same arithmetic.
 *
 * ARM_MATH_DSP stays undefined here, which selects the plain C version.
 * Every function the two FFT sources define is renamed; the tables and the
 * bit reversal are shared. */
#define arm_cfft_q15                     plain_cfft_q15
#define arm_cfft_radix4_q15              plain_cfft_radix4_q15
#define arm_cfft_radix4by2_q15           plain_cfft_radix4by2_q15
#define arm_cfft_radix4by2_inverse_q15   plain_cfft_radix4by2_inverse_q15
#define arm_radix4_butterfly_q15         plain_radix4_butterfly_q15
#define arm_radix4_butterfly_inverse_q15 plain_radix4_butterfly_inverse_q15

#include "arm_math.h"
#include "arm_common_tables.h"
#include "arm_const_structs.h"

#ifdef ARM_MATH_DSP
#error "this file must build the plain C version"
#endif

#include "arm_cfft_radix4_q15.c"
#include "arm_cfft_q15.c"
