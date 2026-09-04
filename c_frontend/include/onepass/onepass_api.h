#ifndef ONEPASS_API_H
#define ONEPASS_API_H
#include <stdint.h>

/* Flat FFI surface over the one-pass composition, for ctypes: opaque
 * handle, caller-allocated outputs, no structs across the boundary — the
 * fmcw_api.h conventions. The handle allocates the composition's storage
 * on the heap; on the target that storage is static and this file is not
 * compiled. Detection rows are fmcw_q15_process_frame's 7 columns, so the
 * two are directly comparable. */

void *fmcw_onepass_create(const char *params_cfg);
void fmcw_onepass_destroy(void *h);
int fmcw_onepass_dims(void *h, int out[3]);

/* Feed the row for (v, c): 2*S int16, interleaved I/Q. v-major order. */
int fmcw_onepass_row(void *h, int v, int c, const int16_t *row);

/* Finish the frame; writes up to max_out rows of
 * [range_bin, doppler_bin, azimuth_deg, x, y, v_radial, snr_db].
 * Returns the count, or -1. */
int fmcw_onepass_finish(void *h, double *det, int max_out);

/* Quantize n complex samples (2n floats, interleaved) onto the 12-bit
 * left-aligned Q15 grid — adc_to_q15 exactly. Returns the saturation
 * count, or -1 on a NULL pointer. */
long long fmcw_q15_quantize(const float *adc, long long n,
                            double adc_full_scale, int16_t *out_q);
#endif
