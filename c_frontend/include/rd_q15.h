#ifndef RD_Q15_H
#define RD_Q15_H
#include "fft_q15.h"

/* Range-Doppler stage in Q15 — the integer counterpart of rd.h, same chain
 * in the same order:
 *
 *   per-chirp DC removal -> range Hann -> range FFT (keep S/2 bins)
 *   -> slow-time mean (clutter) removal -> Doppler Hann -> Doppler FFT
 *   -> fftshift
 *
 * Both means are computed with integer arithmetic and a shift, not a
 * divide: the FFT already requires both dimensions to be powers of two, so
 * the division is free and the stage needs no FPU at all.
 */

typedef struct {
    int n_chirps, n_samples;      /* C, S; output is C x S/2 */
    int shift_chirps;             /* log2(C) — the slow-time mean's divide */
    int shift_samples;            /* log2(S) — the fast-time mean's divide */
    FftPlanQ15 range_plan;        /* size S */
    FftPlanQ15 doppler_plan;      /* size C */
    int16_t *win_range;           /* Q15 hann, length S */
    int16_t *win_doppler;         /* Q15 hann, length C */
    cplx_q15 *scratch_row;        /* fast-time workspace, length S */
    cplx_q15 *scratch_col;        /* slow-time workspace, length C */
} RdPlanQ15;

/* Precompute windows, FFT plans, and scratch space. Both dims must be
 * powers of two >= 2. Returns 0 on success, -1 on failure (plan then safe
 * to pass to rd_q15_plan_free). */
int rd_q15_plan_init(RdPlanQ15 *p, int n_chirps, int n_samples);

/* Release everything the plan owns; safe after a failed init. */
void rd_q15_plan_free(RdPlanQ15 *p);

/* np.hanning on the Q15 grid: round(w[i] * 32768), clamped at 32767. For
 * even n the window's maximum is 0.99996, so the clamp only ever engages
 * for odd n, where the centre tap is exactly 1.0 — a value Q15 cannot
 * represent. n >= 2. */
void hann_fill_q15(int16_t *w, int n);

/* Process one channel. in: C x S row-major Q15 (chirp-major, from
 * adc_to_q15). out: C x S/2 row-major RD map, at 1/(S*C) of the float
 * pipeline's scale — see q15_rd_scale.
 *
 * *sat accumulates saturation from this stage (NULL to ignore). Two sites
 * can contribute: the slow-time mean removal, which subtracts from
 * full-range post-FFT values and is the one that realistically fires, and
 * the FFT's twiddle multiply if the magnitude precondition is broken. The
 * per-chirp DC removal cannot clip — its inputs are bounded by the
 * left-align-by-3 guard bit — so a nonzero count never originates there.
 *
 * Uses the plan's scratch buffers: one plan must not be shared across
 * threads. */
void rd_q15_process_channel(RdPlanQ15 *p, const cplx_q15 *in,
                            cplx_q15 *out, long long *sat);
#endif
