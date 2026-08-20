#ifndef RD_H
#define RD_H
#include "fft.h"

/* Range-Doppler stage for one virtual channel: the C port of
 * processing/range_doppler.py's float path with its defaults
 * (hann window, dc_removal=True, clutter_removal=True).
 *
 * Chain: per-chirp DC removal -> range Hann -> range FFT (keep S/2 bins)
 *        -> slow-time mean (clutter) removal -> Doppler Hann
 *        -> Doppler FFT -> fftshift.
 */

typedef struct {
    int n_chirps, n_samples;      /* C, S; output is C x S/2 */
    FftPlan range_plan;           /* size S */
    FftPlan doppler_plan;         /* size C */
    double *win_range;            /* hann, length S */
    double *win_doppler;          /* hann, length C */
    cplx *scratch_row;            /* fast-time workspace, length S */
    cplx *scratch_col;            /* slow-time workspace, length C */
} RdPlan;

/* Precompute windows, FFT plans, and scratch space. Both dims must be
 * powers of two >= 2 (FFT requirement). Returns 0 on success, -1 on
 * failure (plan then safe to pass to rd_plan_free). */
int rd_plan_init(RdPlan *p, int n_chirps, int n_samples);

/* Release everything the plan owns; safe after a failed init. */
void rd_plan_free(RdPlan *p);

/* np.hanning: w[i] = 0.5 - 0.5*cos(2*pi*i/(n-1)) — the SYMMETRIC window
 * (zero at both endpoints), not the periodic variant. n >= 2. */
void hann_fill(double *w, int n);

/* Process one channel. in: C x S row-major (chirp-major, matching the
 * exported cube layout). out: C x S/2 row-major RD map. Uses the plan's
 * scratch buffers — one plan must not be shared across threads. */
void rd_process_channel(RdPlan *p, const cplx *in, cplx *out);
#endif
