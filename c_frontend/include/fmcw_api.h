#ifndef FMCW_API_H
#define FMCW_API_H
#include <stdint.h>

/* Flat C API over the front-end kernels, designed for FFI callers
 * (ctypes): an opaque context handle, no structs across the boundary,
 * caller-allocated outputs. Layouts (row-major, dims from fmcw_dims):
 *   adc    V*C*S   complex float32 = 2*V*C*S floats, interleaved I/Q
 *   rd     V*C*(S/2) complex float32
 *   power  C*(S/2) float64
 *   mask   C*(S/2) uint8 (0/1)
 *   det    rows of 7 float64:
 *          [range_bin, doppler_bin, azimuth_deg, x, y, v_radial, snr_db]
 * Every function tolerates a NULL handle/pointer (returns -1 / no-op).
 * Compute only: no file I/O, no timing. One handle per thread. */

/* Build a context (params + buffers + plans) from a params.cfg file.
 * Returns NULL on any failure (bad path, bad config, allocation). */
void *fmcw_create(const char *params_cfg);

/* Release a context; NULL-safe. */
void fmcw_destroy(void *h);

/* out[0..2] = n_virtual, n_chirps, n_samples. Returns 0, or -1. */
int fmcw_dims(void *h, int out[3]);

/* Whole chain: adc -> detections. Returns the detection count written
 * (at most max_out rows into det), or -1. */
int fmcw_process_frame(void *h, const float *adc, double *det, int max_out);

/* Individual stages, same layouts; each returns 0 (count for
 * fmcw_pointcloud), or -1. */
int fmcw_rd(void *h, const float *adc, float *rd);
int fmcw_nci(void *h, const float *rd, double *power);
int fmcw_cfar(void *h, const double *power, uint8_t *mask);
int fmcw_pointcloud(void *h, const float *rd, const uint8_t *mask,
                    double *det, int max_out);
#endif
