#ifndef FFT_H
#define FFT_H
#include "fmcw.h"

/* Iterative radix-2 decimation-in-time FFT.
 *
 * Convention: forward transform, unnormalized — X[k] = sum_n x[n]*exp(-2*pi*i*n*k/N),
 * matching np.fft.fft. (NumPy puts the 1/N factor in the inverse transform.)
 *
 * Usage: fft_plan_init once per size, fft_inplace per transform (any number
 * of times), fft_plan_free when done. A plan is immutable after init, so one
 * plan can serve many transforms.
 */

typedef struct {
    int n;
    cplx *twiddle;   /* twiddle[k] = exp(-2*pi*i*k/n), k in [0, n/2) */
    int *bitrev;     /* bit-reversal permutation of [0, n) */
} FftPlan;

/* Precompute twiddle factors and the bit-reversal table for size n.
 * n must be a power of two >= 2. Returns 0 on success, -1 on invalid n or
 * allocation failure (in which case the plan is safe to pass to
 * fft_plan_free, but not to fft_inplace). */
int fft_plan_init(FftPlan *p, int n);

/* Release the plan's tables. Safe to call on a failed/zeroed plan. */
void fft_plan_free(FftPlan *p);

/* Transform x (length p->n) in place: x is overwritten with its DFT. */
void fft_inplace(const FftPlan *p, cplx *x);

/* Swap the two halves of x (length n, n even): moves the zero-frequency bin
 * to the array centre, same as np.fft.fftshift on an even-length axis. */
void fftshift_inplace(cplx *x, int n);
#endif
