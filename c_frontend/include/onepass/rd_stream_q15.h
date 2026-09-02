#ifndef RD_STREAM_Q15_H
#define RD_STREAM_Q15_H
#include "fft_q15.h"

/* The range-Doppler stage restructured for rows that arrive one at a time,
 * so the raw frame never needs to exist in memory. Same chain as rd_q15.h
 * in the same order — per-chirp DC removal, range Hann, range FFT, keep
 * S/2 bins; then (once the plane is complete) slow-time mean removal,
 * Doppler Hann, Doppler FFT, fftshift — and bit-exact against it.
 *
 * The plan owns only windows and FFT tables (heap, built once); all frame
 * storage is the caller's. One plan serves every channel. */

typedef struct {
    int n_chirps, n_samples;      /* C, S; a plane is C x S/2 */
    int shift_chirps, shift_samples;
    FftPlanQ15 range_plan;        /* size S */
    FftPlanQ15 doppler_plan;      /* size C */
    int16_t *win_range;           /* Q15 hann, length S */
    int16_t *win_doppler;         /* Q15 hann, length C */
} RdStreamPlanQ15;

/* Both dims powers of two >= 2. Returns 0, or -1 (plan then safe to free). */
int rd_stream_q15_plan_init(RdStreamPlanQ15 *p, int n_chirps, int n_samples);
void rd_stream_q15_plan_free(RdStreamPlanQ15 *p);

/* One arriving chirp: DC removal -> range Hann -> range FFT -> the S/2
 * positive-beat bins into plane row c. row (length S) is the workspace and
 * is clobbered — receive or copy into it, then hand it over. */
void rd_stream_q15_row(RdStreamPlanQ15 *p, cplx_q15 *row, int c,
                       cplx_q15 *plane, long long *sat);

/* After all C rows of a plane: slow-time (clutter) mean removal, then per
 * range bin Doppler Hann -> FFT -> fftshift. col (length C) is caller
 * workspace — the row buffer, reused, is big enough. */
void rd_stream_q15_plane_finish(RdStreamPlanQ15 *p, cplx_q15 *plane,
                                cplx_q15 *col, long long *sat);
#endif
