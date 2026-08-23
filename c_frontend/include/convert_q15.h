#ifndef CONVERT_Q15_H
#define CONVERT_Q15_H
#include "fmcw.h"
#include "fmcw_q15.h"

/* The two ends of one mapping, between the float path's physical units and
 * the Q15 integer path.
 *
 * Forward (adc_to_q15) is more than a format change: it models a 12-bit ADC
 * with a fixed input range, so a sample beyond that range clips rather than
 * rescaling the grid. Reverse (q15_to_cplx) applies the accumulated scale
 * so integer values return to units the float path can be compared against.
 *
 * The stored integer is left-aligned by 3, not 4. Aligning by 4 would fill
 * the container and leave the DC removal that follows — which can double a
 * magnitude — no room; by 3 that subtraction provably cannot overflow. It
 * costs nothing real, since the noise floor is set by the 12-bit step
 * relative to full scale either way.
 */

/* Quantize n samples onto the 12-bit grid spanning +/- adc_full_scale and
 * left-align into int16. Counts clipped components into *sat (NULL to
 * ignore). in and out may not overlap. */
void adc_to_q15(const cplx *in, int n, double adc_full_scale,
                cplx_q15 *out, long long *sat);

/* Convert n samples back to float, multiplying by `scale`. Lossless in
 * itself: an int16 fits exactly in a float32 mantissa. It does not undo
 * quantization — that damage is already in the values. */
void q15_to_cplx(const cplx_q15 *in, int n, double scale, cplx *out);

/* Reconstruction factor for an RD cube: multiply an integer RD value by
 * this to compare it against the float pipeline's output.
 *
 * Forward, a sample is scaled by K = 2^3 * 2^11 / adc_full_scale, and each
 * FFT divides by its length (one rounding >>1 per stage), so the cube ends
 * up at K/(n_samples*n_chirps) of the float value. This inverts that.
 * Pass n_samples = n_chirps = 1 to invert a bare quantization. */
double q15_rd_scale(double adc_full_scale, int n_samples, int n_chirps);
#endif
