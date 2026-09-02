/* The loop bodies are rd_q15_process_channel's, re-cut at the row
 * boundary: the fast-time work runs per row as it lands (in place — the
 * arriving row IS the workspace), the slow-time work runs once the plane
 * is complete. Same operations in the same order is what makes the output
 * bit-identical to the frozen stage, and the parity test holds it there.
 * mean_shift and log2i are replicated from rd_q15.c, where they are
 * static; hann_fill_q15 is the frozen one. */
#include <stdlib.h>
#include <string.h>
#include "rd_q15.h"          /* hann_fill_q15 */
#include "rd_stream_q15.h"

static int log2i(int n)
{
    int b = 0;
    while ((1 << b) < n) b++;
    return b;
}

static int16_t mean_shift(int32_t sum, int n, int shift)
{
    return (int16_t)((sum + (n >> 1)) >> shift);
}

int rd_stream_q15_plan_init(RdStreamPlanQ15 *p, int n_chirps, int n_samples)
{
    memset(p, 0, sizeof *p);
    p->n_chirps = n_chirps;
    p->n_samples = n_samples;
    p->shift_chirps = log2i(n_chirps);
    p->shift_samples = log2i(n_samples);
    p->win_range = malloc((size_t)n_samples * sizeof(int16_t));
    p->win_doppler = malloc((size_t)n_chirps * sizeof(int16_t));
    if (!p->win_range || !p->win_doppler)
        { rd_stream_q15_plan_free(p); return -1; }
    if (fft_q15_plan_init(&p->range_plan, n_samples) != 0 ||
        fft_q15_plan_init(&p->doppler_plan, n_chirps) != 0)
        { rd_stream_q15_plan_free(p); return -1; }
    hann_fill_q15(p->win_range, n_samples);
    hann_fill_q15(p->win_doppler, n_chirps);
    return 0;
}

void rd_stream_q15_plan_free(RdStreamPlanQ15 *p)
{
    free(p->win_range);
    free(p->win_doppler);
    p->win_range = p->win_doppler = NULL;
    fft_q15_plan_free(&p->range_plan);
    fft_q15_plan_free(&p->doppler_plan);
}

void rd_stream_q15_row(RdStreamPlanQ15 *p, cplx_q15 *row, int c,
                       cplx_q15 *plane, long long *sat)
{
    int S = p->n_samples, R = S / 2;
    int32_t sre = 0, sim = 0;
    for (int s = 0; s < S; s++) { sre += row[s].re; sim += row[s].im; }
    cplx_q15 mean = { mean_shift(sre, S, p->shift_samples),
                      mean_shift(sim, S, p->shift_samples) };
    for (int s = 0; s < S; s++) {
        cplx_q15 v = cq15_sub_sat(row[s], mean, sat);
        row[s].re = q15_mul(v.re, p->win_range[s]);
        row[s].im = q15_mul(v.im, p->win_range[s]);
    }
    fft_q15_inplace(&p->range_plan, row, sat);
    for (int r = 0; r < R; r++) plane[(size_t)c * R + r] = row[r];
}

void rd_stream_q15_plane_finish(RdStreamPlanQ15 *p, cplx_q15 *plane,
                                cplx_q15 *col, long long *sat)
{
    int C = p->n_chirps, R = p->n_samples / 2;
    for (int r = 0; r < R; r++) {
        int32_t sre = 0, sim = 0;
        for (int c = 0; c < C; c++) {
            sre += plane[(size_t)c * R + r].re;
            sim += plane[(size_t)c * R + r].im;
        }
        cplx_q15 mean = { mean_shift(sre, C, p->shift_chirps),
                          mean_shift(sim, C, p->shift_chirps) };
        for (int c = 0; c < C; c++) {
            cplx_q15 *v = &plane[(size_t)c * R + r];
            *v = cq15_sub_sat(*v, mean, sat);
        }
    }
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            cplx_q15 v = plane[(size_t)c * R + r];
            col[c].re = q15_mul(v.re, p->win_doppler[c]);
            col[c].im = q15_mul(v.im, p->win_doppler[c]);
        }
        fft_q15_inplace(&p->doppler_plan, col, sat);
        fftshift_q15_inplace(col, C);
        for (int c = 0; c < C; c++) plane[(size_t)c * R + r] = col[c];
    }
}
