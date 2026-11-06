/* Float <-> Q15 boundary. Contracts in convert_q15.h; numerics here. */
#include <math.h>
#include "convert_q15.h"

/* The modelled converter: 12-bit signed, so codes run [-2048, 2047]. Note
 * the asymmetry is real and not a rounding artifact — two's complement has
 * one more negative code than positive, and a sample at exactly +full_scale
 * therefore clips by one code while -full_scale does not. */
#define ADC_BITS        12
#define ADC_CODE_MAX    ((1 << (ADC_BITS - 1)) - 1)
#define ADC_CODE_MIN    (-(1 << (ADC_BITS - 1)))
#define ADC_ALIGN_SHIFT 3

/* One component onto the grid. The range test is written against the
 * rounding boundary (+/- 0.5 of the extreme code) rather than against the
 * rounded result, so the clip decision is made before lround is asked to
 * convert a value that might not fit in a long.
 *
 * Takes the reciprocal of the step rather than the step: this runs once per
 * I and Q component of every sample, and a division there costs far more
 * than the multiply, for a 1-ULP difference that could only change a value
 * sitting exactly on a rounding boundary. */
static int16_t quantize_one(double x, double inv_step, long long *sat)
{
    double r = x * inv_step;
    long q;
    if (r >= (double)ADC_CODE_MAX + 0.5) {
        if (sat) (*sat)++;
        q = ADC_CODE_MAX;
    } else if (r <= (double)ADC_CODE_MIN - 0.5) {
        if (sat) (*sat)++;
        q = ADC_CODE_MIN;
    } else {
        q = lround(r);              /* half away from zero */
    }
    return (int16_t)(q << ADC_ALIGN_SHIFT);
}

void adc_to_q15(const cplx *in, int n, double adc_full_scale,
                cplx_q15 *out, long long *sat)
{
    double inv_step = (double)(1 << (ADC_BITS - 1)) / adc_full_scale;
    for (int i = 0; i < n; i++) {
        out[i].re = quantize_one((double)in[i].re, inv_step, sat);
        out[i].im = quantize_one((double)in[i].im, inv_step, sat);
    }
}

void q15_to_cplx(const cplx_q15 *in, int n, double scale, cplx *out)
{
    for (int i = 0; i < n; i++) {
        out[i].re = (float)((double)in[i].re * scale);
        out[i].im = (float)((double)in[i].im * scale);
    }
}

double q15_rd_scale(double adc_full_scale, int n_samples, int n_chirps)
{
    double k = (double)(1 << ADC_ALIGN_SHIFT)
             * (double)(1 << (ADC_BITS - 1)) / adc_full_scale;
    return (double)n_samples * (double)n_chirps / k;
}
