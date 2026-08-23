#ifndef FFT_Q15_H
#define FFT_Q15_H
#include "fmcw_q15.h"

/* Iterative radix-2 decimation-in-time FFT in Q15 — the integer counterpart
 * of fft.h, same structure, same twiddle and bit-reversal tables.
 *
 * Scaling: one rounding >>1 after every butterfly stage, so the output is
 * the DFT divided by n. That is the whole bit-growth policy — a static,
 * compile-time-known factor rather than a per-frame exponent, which is what
 * keeps the downstream scale ledger to a single constant.
 *
 * Range: exactly safe for any input with |x| <= 32767, where that is a bound
 * on the COMPLEX MAGNITUDE, not on the components. Given |a|,|b| <= 32767
 * and |w| = 1, we get |a +/- w*b| <= 65534 and halving returns it to
 * <= 32767, so the invariant reproduces itself stage after stage. A value
 * with both components near full scale has magnitude 46341 and breaks it;
 * *sat counts the resulting clipping rather than letting it pass silently.
 */

typedef struct {
    int n;
    cplx_q15 *twiddle;   /* Q15, exp(-2*pi*i*k/n), k in [0, n/2) */
    int *bitrev;         /* bit-reversal permutation of [0, n) */
} FftPlanQ15;

/* Precompute the tables for size n, a power of two >= 2. Returns 0, or -1
 * on invalid n or allocation failure (the plan is then safe to free, but
 * not to transform with). */
int fft_q15_plan_init(FftPlanQ15 *p, int n);

/* Release the plan's tables. Safe on a failed or zeroed plan. */
void fft_q15_plan_free(FftPlanQ15 *p);

/* Transform x (length p->n) in place; x is overwritten with its DFT scaled
 * by 1/n. *sat accumulates saturation counts (NULL to ignore). */
void fft_q15_inplace(const FftPlanQ15 *p, cplx_q15 *x, long long *sat);

/* Swap the two halves of x (length n, n even), moving the zero-frequency
 * bin to the array centre — np.fft.fftshift on an even-length axis. */
void fftshift_q15_inplace(cplx_q15 *x, int n);
#endif
