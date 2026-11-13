/* fft_q15_inplace as the board builds it with FFT=cmsis: the block
 * scaling of Core/Src/fft_q15_cmsis.c on top of CMSIS-DSP's arm_cfft_q15
 * in the version the board runs. */
#include <string.h>
#include "fmcw.h"
#include "fft_q15.h"
/* both the kernel's header and CMSIS-DSP's define these two */
#undef Q15_MIN
#undef Q15_MAX
#include "arm_math.h"
#include "arm_const_structs.h"
#include "check.h"

#define NMAX 128

/* CMSIS-DSP's plain C version of the same FFT (tests/cfft_q15_plain_path.c) */
void plain_cfft_q15(const arm_cfft_instance_q15 *S, q15_t *p1,
                    uint8_t ifftFlag, uint8_t bitReverseFlag);

static uint32_t rng = 12345u;
static int rnd(int lo, int hi)
{
    rng = rng * 1103515245u + 12345u;
    return lo + (int)((rng >> 16) % (uint32_t)(hi - lo + 1));
}

static void fill(cplx_q15 *x, int n, int amp)
{
    for (int j = 0; j < n; j++) {
        x[j].re = (int16_t)rnd(-amp, amp);
        x[j].im = (int16_t)rnd(-amp, amp);
    }
}

static const arm_cfft_instance_q15 *inst_of(int n)
{
    return n == 32 ? &arm_cfft_sR_q15_len32 : &arm_cfft_sR_q15_len128;
}

/* The largest distance of got from the DFT of x divided by n, over all
 * components. */
static double max_err(const cplx_q15 *x, const cplx_q15 *got, int n)
{
    double worst = 0.0;
    for (int k = 0; k < n; k++) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < n; j++) {
            double ang = -2.0 * FMCW_PI * k * j / (double)n;
            re += x[j].re * cos(ang) - x[j].im * sin(ang);
            im += x[j].re * sin(ang) + x[j].im * cos(ang);
        }
        double e_re = fabs(got[k].re - re / n);
        double e_im = fabs(got[k].im - im / n);
        if (e_re > worst) worst = e_re;
        if (e_im > worst) worst = e_im;
    }
    return worst;
}

int main(void)
{
    static const int sizes[2] = { 32, 128 };
    FftPlanQ15 p = {0};
    cplx_q15 x[NMAX], y[NMAX], want[NMAX];
    long long sat = 0;

    /* ---- 1. only the two sizes the front end uses ---- */
    CHECK(fft_q15_plan_init(&p, 64) == -1);
    CHECK(fft_q15_plan_init(&p, 32) == 0);
    CHECK(fft_q15_plan_init(&p, 128) == 0);

    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        const arm_cfft_instance_q15 *inst = inst_of(n);
        size_t bytes = (size_t)n * sizeof(cplx_q15);
        CHECK(fft_q15_plan_init(&p, n) == 0);

        /* ---- 2. zeros stay zeros ---- */
        memset(x, 0, bytes);
        memset(want, 0, bytes);
        fft_q15_inplace(&p, x, &sat);
        CHECK(memcmp(x, want, bytes) == 0);

        /* ---- 3. a block of small values is transformed accurately ----
         *
         * With every component within +-15 the block is shifted up by at
         * least 10 bits. Whatever arm_cfft_q15 loses inside is divided by
         * 1024 on the way back, so the result is the true value rounded
         * to an integer: at most half a count away, plus that remainder
         * (allowed here: 16 counts of the scaled transform). Without the
         * scaling, or with the shift back truncating instead of rounding,
         * the error reaches a whole count. */
        for (int rep = 0; rep < 20; rep++) {
            fill(x, n, 15);
            memcpy(y, x, bytes);
            fft_q15_inplace(&p, y, &sat);
            CHECK(max_err(x, y, n) <= 0.5 + 16.0 / 1024.0);
        }

        /* ---- 4. where the scaling starts ----
         *
         * Largest component 8192 or more: no bit is free above it besides
         * the spare one, and the block is transformed as it is. */
        fill(x, n, 8000);
        x[5].im = 8192;
        memcpy(y, x, bytes);
        memcpy(want, x, bytes);
        fft_q15_inplace(&p, y, &sat);
        arm_cfft_q15(inst, (q15_t *)want, 0, 1);
        CHECK(memcmp(y, want, bytes) == 0);

        /* -32768 has no positive counterpart in 16 bits; also unscaled */
        fill(x, n, 8000);
        x[3].re = -32768;
        memcpy(y, x, bytes);
        memcpy(want, x, bytes);
        fft_q15_inplace(&p, y, &sat);
        arm_cfft_q15(inst, (q15_t *)want, 0, 1);
        CHECK(memcmp(y, want, bytes) == 0);

        /* Largest component 8191: shifted up by one bit, transformed, and
         * shifted back with rounding. */
        fill(x, n, 8000);
        x[5].im = -8191;
        memcpy(y, x, bytes);
        for (int j = 0; j < n; j++) {
            want[j].re = (int16_t)(2 * x[j].re);
            want[j].im = (int16_t)(2 * x[j].im);
        }
        fft_q15_inplace(&p, y, &sat);
        arm_cfft_q15(inst, (q15_t *)want, 0, 1);
        for (int j = 0; j < n; j++) {
            want[j].re = (int16_t)((want[j].re + 1) >> 1);
            want[j].im = (int16_t)((want[j].im + 1) >> 1);
        }
        CHECK(memcmp(y, want, bytes) == 0);

        /* ---- 5. the FFT underneath is the board's version ----
         *
         * CMSIS-DSP's version for cores with DSP instructions and its
         * plain C version round differently. If arm_cfft_q15 here were
         * the plain one, the laptop reference would no longer be the
         * board's arithmetic, and no other laptop test would notice. */
        int differing = 0;
        for (int rep = 0; rep < 8; rep++) {
            fill(x, n, 16000);
            memcpy(y, x, bytes);
            memcpy(want, x, bytes);
            arm_cfft_q15(inst, (q15_t *)y, 0, 1);
            plain_cfft_q15(inst, (q15_t *)want, 0, 1);
            if (memcmp(y, want, bytes) != 0) differing++;
        }
        CHECK(differing > 0);
    }

    /* the wrapper does not count saturation */
    CHECK(sat == 0);
    fft_q15_plan_free(&p);
    CHECK_DONE();
}
