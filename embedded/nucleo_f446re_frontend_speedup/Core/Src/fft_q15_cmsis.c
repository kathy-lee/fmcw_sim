/* The functions of fft_q15.h on top of CMSIS-DSP's arm_cfft_q15, so the
 * kernels that call them build unchanged against either FFT.
 *
 * arm_cfft_q15 scales as the project FFT does — the output is the DFT
 * divided by n — but rounds differently inside, so its bins differ from
 * the project FFT's by a few counts. Results from this build are close to
 * the laptop reference, not bit-identical.
 *
 * The tables are CMSIS-DSP's constants in flash: a plan allocates nothing,
 * and its twiddle and bitrev stay NULL. */
#include <stddef.h>
#include "fft_q15.h"
/* both the kernel's header and CMSIS-DSP's define these two, with the same
 * values; this file uses neither */
#undef Q15_MIN
#undef Q15_MAX
#include "arm_math.h"
#include "arm_const_structs.h"

int fft_q15_plan_init(FftPlanQ15 *p, int n)
{
    p->twiddle = NULL;
    p->bitrev = NULL;
    p->n = 0;
    if (n != 32 && n != 128) return -1;   /* the sizes the front end uses */
    p->n = n;
    return 0;
}

void fft_q15_plan_free(FftPlanQ15 *p)
{
    p->twiddle = NULL;
    p->bitrev = NULL;
    p->n = 0;
}

/* arm_cfft_q15 is built for inputs near full scale: it drops the lowest
 * bit of every input before its first stage, truncates at each later
 * stage, and shifts the result back up by one at the end. For a block of
 * small values — the Doppler FFT's input is the range FFT's output, already
 * divided by its length — those truncations are as large as the values.
 *
 * So the block is first shifted up as far as its largest component allows,
 * keeping one spare bit, transformed, and shifted back down with rounding.
 * The truncation errors are divided by the same factor on the way down.
 * A block that already fills the word is transformed as it is.
 *
 * *sat is left alone: CMSIS-DSP saturates without counting, so the
 * caller's count then covers only the steps outside the FFT. */
void fft_q15_inplace(const FftPlanQ15 *p, cplx_q15 *x, long long *sat)
{
    (void)sat;
    const arm_cfft_instance_q15 *inst =
        (p->n == 32) ? &arm_cfft_sR_q15_len32 : &arm_cfft_sR_q15_len128;
    q15_t *v = (q15_t *)x;
    int len = 2 * p->n;

    uint32_t mag = 0;                       /* OR of every |component| */
    for (int i = 0; i < len; i++)
        mag |= (uint32_t)(v[i] < 0 ? -(int32_t)v[i] : v[i]);
    if (mag == 0) return;                   /* the transform of zeros */

    /* bits free above the largest component, less the sign bit and one
     * spare bit */
    int up = __builtin_clz(mag) - 18;
    if (up <= 0) {
        arm_cfft_q15(inst, v, 0, 1);
        return;
    }
    for (int i = 0; i < len; i++) v[i] = (q15_t)(v[i] * (1 << up));
    arm_cfft_q15(inst, v, 0, 1);
    int half = 1 << (up - 1);
    for (int i = 0; i < len; i++) v[i] = (q15_t)((v[i] + half) >> up);
}

void fftshift_q15_inplace(cplx_q15 *x, int n)
{
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        cplx_q15 t = x[i];
        x[i] = x[i + h];
        x[i + h] = t;
    }
}
