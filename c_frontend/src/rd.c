/* Range-Doppler stage. Contracts in rd.h; numerics notes here.
 *
 * Faithfulness to the Python reference is the design goal (spec section 6):
 * complex data stays float32, but every mean is accumulated in double before
 * subtraction — at least as accurate as NumPy's pairwise float32 summation,
 * so the golden comparison stays within float32 rounding noise. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "rd.h"

void hann_fill(double *w, int n)
{
    for (int i = 0; i < n; i++)
        w[i] = 0.5 - 0.5 * cos(2.0 * FMCW_PI * i / (n - 1));
}

int rd_plan_init(RdPlan *p, int n_chirps, int n_samples)
{
    /* Zero first so rd_plan_free is safe on any partial-init failure path
     * (free(NULL) is a no-op; fft_plan_free tolerates a zeroed plan). */
    memset(p, 0, sizeof *p);
    p->n_chirps = n_chirps;
    p->n_samples = n_samples;
    p->win_range = malloc((size_t)n_samples * sizeof(double));
    p->win_doppler = malloc((size_t)n_chirps * sizeof(double));
    p->scratch_row = malloc((size_t)n_samples * sizeof(cplx));
    p->scratch_col = malloc((size_t)n_chirps * sizeof(cplx));
    if (!p->win_range || !p->win_doppler || !p->scratch_row || !p->scratch_col)
        { rd_plan_free(p); return -1; }
    if (fft_plan_init(&p->range_plan, n_samples) != 0 ||
        fft_plan_init(&p->doppler_plan, n_chirps) != 0)
        { rd_plan_free(p); return -1; }
    hann_fill(p->win_range, n_samples);
    hann_fill(p->win_doppler, n_chirps);
    return 0;
}

void rd_plan_free(RdPlan *p)
{
    free(p->win_range);
    free(p->win_doppler);
    free(p->scratch_row);
    free(p->scratch_col);
    p->win_range = p->win_doppler = NULL;
    p->scratch_row = p->scratch_col = NULL;
    fft_plan_free(&p->range_plan);
    fft_plan_free(&p->doppler_plan);
}

void rd_process_channel(RdPlan *p, const cplx *in, cplx *out)
{
    int C = p->n_chirps, S = p->n_samples, R = S / 2;

    /* Per chirp (fast time): DC removal -> Hann -> range FFT -> keep the
     * S/2 positive-beat-frequency bins (the input is effectively real-ish
     * beat data; the upper half mirrors the lower). */
    for (int c = 0; c < C; c++) {
        const cplx *row = in + (size_t)c * S;
        double mre = 0.0, mim = 0.0;
        for (int s = 0; s < S; s++) { mre += row[s].re; mim += row[s].im; }
        mre /= S; mim /= S;
        for (int s = 0; s < S; s++) {
            double re = ((double)row[s].re - mre) * p->win_range[s];
            double im = ((double)row[s].im - mim) * p->win_range[s];
            p->scratch_row[s] = (cplx){(float)re, (float)im};
        }
        fft_inplace(&p->range_plan, p->scratch_row);
        for (int r = 0; r < R; r++) out[(size_t)c * R + r] = p->scratch_row[r];
    }

    /* Slow-time (clutter) mean removal per range bin: subtracting the
     * across-chirp mean notches zero-Doppler returns (stationary clutter). */
    for (int r = 0; r < R; r++) {
        double mre = 0.0, mim = 0.0;
        for (int c = 0; c < C; c++) {
            mre += out[(size_t)c * R + r].re;
            mim += out[(size_t)c * R + r].im;
        }
        mre /= C; mim /= C;
        for (int c = 0; c < C; c++) {
            cplx *v = &out[(size_t)c * R + r];
            v->re = (float)((double)v->re - mre);
            v->im = (float)((double)v->im - mim);
        }
    }

    /* Per range bin (slow time): gather the chirp column, apply the Doppler
     * Hann while gathering, FFT, fftshift (zero Doppler to the centre),
     * scatter back. The gather/scatter is the "corner turn" done in-place
     * via a scratch column — real DSPs burn real engineering on this step. */
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cplx v = out[(size_t)c * R + r];
            p->scratch_col[c] = (cplx){(float)(v.re * p->win_doppler[c]),
                                       (float)(v.im * p->win_doppler[c])};
        }
        fft_inplace(&p->doppler_plan, p->scratch_col);
        fftshift_inplace(p->scratch_col, C);
        for (int c = 0; c < C; c++) out[(size_t)c * R + r] = p->scratch_col[c];
    }
}
