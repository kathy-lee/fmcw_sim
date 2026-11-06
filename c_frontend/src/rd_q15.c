/* Range-Doppler stage in Q15. Contracts in rd_q15.h; numerics notes here.
 *
 * The float version accumulates its means in double before subtracting, to
 * stay inside float32 rounding noise. Here the means are integer sums
 * divided by a power of two, which is exact up to the final rounding shift
 * — so this stage needs no floating-point arithmetic anywhere, which is the
 * point of the exercise. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fmcw.h"
#include "rd_q15.h"

void hann_fill_q15(int16_t *w, int n)
{
    for (int i = 0; i < n; i++) {
        double v = 0.5 - 0.5 * cos(2.0 * FMCW_PI * i / (n - 1));
        long q = lround(v * (double)Q15_ONE);
        if (q > Q15_MAX) q = Q15_MAX;      /* only reachable for odd n */
        w[i] = (int16_t)q;
    }
}

static int log2i(int n)
{
    int b = 0;
    while ((1 << b) < n) b++;
    return b;
}

/* Mean of `n` accumulated components, rounded. n is a power of two, so the
 * divide is a shift; adding half the divisor first gives the same
 * half-toward-+infinity rounding used everywhere else in the integer path.
 * The result always fits int16: the inputs do, and an average cannot exceed
 * its largest term. */
static int16_t mean_shift(int32_t sum, int n, int shift)
{
    return (int16_t)((sum + (n >> 1)) >> shift);
}

int rd_q15_plan_init(RdPlanQ15 *p, int n_chirps, int n_samples)
{
    memset(p, 0, sizeof *p);
    p->n_chirps = n_chirps;
    p->n_samples = n_samples;
    p->shift_chirps = log2i(n_chirps);
    p->shift_samples = log2i(n_samples);
    p->win_range = malloc((size_t)n_samples * sizeof(int16_t));
    p->win_doppler = malloc((size_t)n_chirps * sizeof(int16_t));
    p->scratch_row = malloc((size_t)n_samples * sizeof(cplx_q15));
    p->scratch_col = malloc((size_t)n_chirps * sizeof(cplx_q15));
    if (!p->win_range || !p->win_doppler || !p->scratch_row || !p->scratch_col)
        { rd_q15_plan_free(p); return -1; }
    if (fft_q15_plan_init(&p->range_plan, n_samples) != 0 ||
        fft_q15_plan_init(&p->doppler_plan, n_chirps) != 0)
        { rd_q15_plan_free(p); return -1; }
    hann_fill_q15(p->win_range, n_samples);
    hann_fill_q15(p->win_doppler, n_chirps);
    return 0;
}

void rd_q15_plan_free(RdPlanQ15 *p)
{
    free(p->win_range);
    free(p->win_doppler);
    free(p->scratch_row);
    free(p->scratch_col);
    p->win_range = p->win_doppler = NULL;
    p->scratch_row = p->scratch_col = NULL;
    fft_q15_plan_free(&p->range_plan);
    fft_q15_plan_free(&p->doppler_plan);
}

void rd_q15_process_channel(RdPlanQ15 *p, const cplx_q15 *in,
                            cplx_q15 *out, long long *sat)
{
    int C = p->n_chirps, S = p->n_samples, R = S / 2;

    /* Per chirp (fast time): DC removal -> Hann -> range FFT -> keep the
     * S/2 positive-beat-frequency bins.
     *
     * The subtraction here provably cannot clip: adc_to_q15 left-aligns by
     * 3, bounding every component by 16384, and |x - mean| <= 2*16384 still
     * fits int16. The saturating form is kept as a cheap invariant check —
     * if this ever contributes to the counter, the input did not come from
     * the quantizer. */
    for (int c = 0; c < C; c++) {
        const cplx_q15 *row = in + (size_t)c * S;
        int32_t sre = 0, sim = 0;
        for (int s = 0; s < S; s++) { sre += row[s].re; sim += row[s].im; }
        cplx_q15 mean = { mean_shift(sre, S, p->shift_samples),
                          mean_shift(sim, S, p->shift_samples) };
        for (int s = 0; s < S; s++) {
            cplx_q15 v = cq15_sub_sat(row[s], mean, sat);
            p->scratch_row[s].re = q15_mul(v.re, p->win_range[s]);
            p->scratch_row[s].im = q15_mul(v.im, p->win_range[s]);
        }
        fft_q15_inplace(&p->range_plan, p->scratch_row, sat);
        for (int r = 0; r < R; r++) out[(size_t)c * R + r] = p->scratch_row[r];
    }

    /* Slow-time (clutter) mean removal per range bin: subtracting the
     * across-chirp mean notches zero-Doppler returns.
     *
     * Unlike the fast-time removal above, this one operates on full-range
     * post-FFT values, so it can genuinely clip — this is the site the
     * saturation counter exists for. */
    for (int r = 0; r < R; r++) {
        int32_t sre = 0, sim = 0;
        for (int c = 0; c < C; c++) {
            sre += out[(size_t)c * R + r].re;
            sim += out[(size_t)c * R + r].im;
        }
        cplx_q15 mean = { mean_shift(sre, C, p->shift_chirps),
                          mean_shift(sim, C, p->shift_chirps) };
        for (int c = 0; c < C; c++) {
            cplx_q15 *v = &out[(size_t)c * R + r];
            *v = cq15_sub_sat(*v, mean, sat);
        }
    }

    /* Per range bin (slow time): gather the chirp column applying the
     * Doppler Hann, FFT, fftshift, scatter back — the corner turn. */
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cplx_q15 v = out[(size_t)c * R + r];
            p->scratch_col[c].re = q15_mul(v.re, p->win_doppler[c]);
            p->scratch_col[c].im = q15_mul(v.im, p->win_doppler[c]);
        }
        fft_q15_inplace(&p->doppler_plan, p->scratch_col, sat);
        fftshift_q15_inplace(p->scratch_col, C);
        for (int c = 0; c < C; c++) out[(size_t)c * R + r] = p->scratch_col[c];
    }
}
