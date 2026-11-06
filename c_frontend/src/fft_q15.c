/* Integer FFT. Contracts in fft_q15.h; numerics notes here.
 *
 * The structure deliberately mirrors fft.c line for line, so the two can be
 * read side by side and the differences are exactly the fixed-point ones:
 * Q15 twiddles instead of float32, a rounding multiply instead of cmul, and
 * the per-stage halving that float does not need. */
#include <math.h>
#include <stdlib.h>
#include "fmcw.h"      /* FMCW_PI — the twiddle angles are computed in double */
#include "fft_q15.h"

int fft_q15_plan_init(FftPlanQ15 *p, int n)
{
    if (n < 2 || (n & (n - 1)) != 0) return -1;
    p->n = n;
    p->twiddle = malloc((size_t)(n / 2) * sizeof(cplx_q15));
    p->bitrev = malloc((size_t)n * sizeof(int));
    if (!p->twiddle || !p->bitrev) { fft_q15_plan_free(p); return -1; }

    /* Twiddles are computed in double, then rounded onto the Q15 grid.
     *
     * Watch w[0] = exp(0) = 1.0: Q15 spans [-1, 1), so 1.0 is NOT
     * representable and clamps to 32767. The resulting 3e-5 relative error
     * is far below the 12-bit input resolution, but it is why a transform
     * of a constant comes back very slightly short rather than exact for
     * large inputs — a classic fixed-point corner, not a bug. */
    for (int k = 0; k < n / 2; k++) {
        double ang = -2.0 * FMCW_PI * k / n;
        long re = lround(cos(ang) * (double)Q15_ONE);
        long im = lround(sin(ang) * (double)Q15_ONE);
        if (re > Q15_MAX) re = Q15_MAX;
        if (im > Q15_MAX) im = Q15_MAX;
        if (re < Q15_MIN) re = Q15_MIN;
        if (im < Q15_MIN) im = Q15_MIN;
        p->twiddle[k].re = (int16_t)re;
        p->twiddle[k].im = (int16_t)im;
    }

    int bits = 0;
    while ((1 << bits) < n) bits++;
    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        p->bitrev[i] = r;
    }
    return 0;
}

void fft_q15_plan_free(FftPlanQ15 *p)
{
    free(p->twiddle);
    free(p->bitrev);
    p->twiddle = NULL;
    p->bitrev = NULL;
    p->n = 0;
}

/* Bit-reverse permute, then log2(n) butterfly stages. Each stage is the
 * float butterfly plus one halving:
 *
 *     a' = (a + w*b) / 2
 *     b' = (a - w*b) / 2
 *
 * The halving is what makes the stage safe rather than merely scaled: it
 * restores the |x| <= 32767 invariant that the addition just doubled. Note
 * the halving itself can never clip — its input range [-65536, 65534] maps
 * exactly onto int16 — so the only saturation site here is the twiddle
 * multiply, and only when the caller broke the input precondition. */
void fft_q15_inplace(const FftPlanQ15 *p, cplx_q15 *x, long long *sat)
{
    int n = p->n;
    for (int i = 0; i < n; i++) {
        int j = p->bitrev[i];
        if (j > i) { cplx_q15 t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        int step = n / len;             /* twiddle stride for this stage */
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; k++) {
                cplx_q15 w = p->twiddle[k * step];
                cplx_q15 a = x[start + k];
                cplx_q15 b = cq15_mul_tw(x[start + k + half], w, sat);
                x[start + k].re = shr1_round((int32_t)a.re + b.re);
                x[start + k].im = shr1_round((int32_t)a.im + b.im);
                x[start + k + half].re = shr1_round((int32_t)a.re - b.re);
                x[start + k + half].im = shr1_round((int32_t)a.im - b.im);
            }
        }
    }
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
