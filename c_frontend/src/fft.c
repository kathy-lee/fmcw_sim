#include <math.h>
#include <stdlib.h>
#include "fft.h"

/* Build the two lookup tables the transform needs, so fft_inplace itself
 * does no trig and no index arithmetic beyond table reads:
 *
 * - twiddle[k] = exp(-2*pi*i*k/n): the "rotation constants" the butterflies
 *   multiply by. Only n/2 are needed — every stage indexes a stride through
 *   this one table. Computed in double, stored as float32 (spec: complex
 *   data is float32; constants are computed at full precision first).
 *
 * - bitrev[i]: i with its low log2(n) bits mirrored (e.g. n=8: 6=110b ->
 *   3=011b). A radix-2 DIT FFT consumes its input in this shuffled order;
 *   pre-swapping lets the butterflies run strictly in place.
 */
int fft_plan_init(FftPlan *p, int n)
{
    /* (n & (n-1)) == 0 is the classic power-of-two test: a power of two has
     * exactly one bit set, and n-1 flips all bits below it. */
    if (n < 2 || (n & (n - 1)) != 0) return -1;
    p->n = n;
    p->twiddle = malloc((size_t)(n / 2) * sizeof(cplx));
    p->bitrev = malloc((size_t)n * sizeof(int));
    if (!p->twiddle || !p->bitrev) { fft_plan_free(p); return -1; }

    for (int k = 0; k < n / 2; k++) {
        double ang = -2.0 * FMCW_PI * k / n;
        p->twiddle[k] = (cplx){(float)cos(ang), (float)sin(ang)};
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

/* free(NULL) is a defined no-op in C, so this is safe after a failed init. */
void fft_plan_free(FftPlan *p)
{
    free(p->twiddle);
    free(p->bitrev);
    p->twiddle = NULL;
    p->bitrev = NULL;
    p->n = 0;
}

/* The transform: bit-reverse permute, then log2(n) butterfly stages.
 *
 * Stage with block length `len` combines pairs of half-blocks: for each pair
 * of elements (a, b) that sit `half` apart,
 *
 *     a' = a + w*b
 *     b' = a - w*b        (the "butterfly": 2 outputs from 2 inputs, in place)
 *
 * where w walks the twiddle table with stride n/len — early stages (small
 * len) take widely spaced twiddles, the last stage (len == n) uses every one.
 * Cost: (n/2)*log2(n) complex multiplies, versus n^2 for the naive DFT.
 */
void fft_inplace(const FftPlan *p, cplx *x)
{
    int n = p->n;
    for (int i = 0; i < n; i++) {
        int j = p->bitrev[i];
        if (j > i) { cplx t = x[i]; x[i] = x[j]; x[j] = t; }  /* swap once, not twice */
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        int step = n / len;             /* twiddle stride for this stage */
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; k++) {
                cplx w = p->twiddle[k * step];
                cplx a = x[start + k];
                cplx b = cmul(w, x[start + k + half]);
                x[start + k] = cadd(a, b);
                x[start + k + half] = csub(a, b);
            }
        }
    }
}

/* For even n, np.fft.fftshift is exactly "swap the front and back halves":
 * bin 0 (zero frequency / zero Doppler) moves to index n/2. */
void fftshift_inplace(cplx *x, int n)
{
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        cplx t = x[i];
        x[i] = x[i + h];
        x[i + h] = t;
    }
}
