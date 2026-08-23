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

/* ---- fixed-point (Q15) entries ---- */

/* The integer chain, mirroring the float entries above one for one and
 * sharing the same handle, so both can run on one frame in one process.
 *
 * Input is the same float ADC frame the float path takes, because
 * quantization is the integer path's first stage — moving it outside would
 * mean the two paths no longer see identical input. Outputs come back in
 * their native integer types rather than pre-converted, so a caller can see
 * how many counts a weak target actually occupies:
 *   rd_q     V*C*(S/2) complex int16 = 2*V*C*(S/2) int16, interleaved I/Q
 *   power_q  C*(S/2) int64
 *   mask     C*(S/2) uint8 (0/1), same as the float path
 *   det      rows of 7 float64, same as the float path — DoA and the
 *            Cartesian conversion are float in both.
 *
 * fmcw_q15_process_frame deliberately has the float facade's exact
 * signature and contract, so the two are drop-in swappable. */
int fmcw_q15_rd(void *h, const float *adc, int16_t *rd_q);
int fmcw_q15_nci(void *h, const int16_t *rd_q, int64_t *power_q);
int fmcw_q15_cfar(void *h, const int64_t *power_q, uint8_t *mask);
int fmcw_q15_pointcloud(void *h, const int16_t *rd_q, const uint8_t *mask,
                        double *det, int max_out);
int fmcw_q15_process_frame(void *h, const float *adc, double *det, int max_out);

/* Multiply an integer RD value by this to reach the float path's units.
 * A constant for the life of the handle — with shift-every-stage the whole
 * scale ledger collapses to one number. Returns 0.0 on a NULL handle. */
double fmcw_q15_rd_scale(void *h);

/* Saturation counts since the last reset: out[0] from input quantization
 * (signal beyond the ADC's full scale), out[1] from the range-Doppler
 * stage (the slow-time mean removal, or the FFT if its magnitude
 * precondition was broken). Nonzero means an input assumption failed, not
 * merely that precision was lost. Returns 0, or -1. */
int fmcw_q15_saturations(void *h, long long out[2]);
void fmcw_q15_reset_saturations(void *h);
#endif
