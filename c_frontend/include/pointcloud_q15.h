#ifndef POINTCLOUD_Q15_H
#define POINTCLOUD_Q15_H
#include "fmcw_q15.h"
#include "pointcloud.h"

/* Point-cloud assembly reading a Q15 RD cube.
 *
 * This is where the integer path hands back to the float one. DoA and the
 * Cartesian conversion are deliberately NOT integerized, so the body is
 * pointcloud_build's, and everything downstream of the snapshot gather —
 * tdm_compensate, doa_bartlett, the SNR and x/y arithmetic — is the very
 * same float code the float path runs. Any angle difference between the
 * two paths is therefore attributable to the integer front end alone.
 *
 * Only the gather differs: each detected cell's n_virtual-element snapshot
 * is converted to float as it is read. Converting per snapshot rather than
 * up front matters — a whole-cube conversion would cost V*C*(S/2) work per
 * frame regardless of how few cells actually fired.
 *
 * rd_scale returns the integers to the float pipeline's units (see
 * q15_rd_scale). The Bartlett angle would not need it, being
 * scale-invariant, but snr_db is an absolute quantity and would otherwise
 * sit at a constant offset from the float path's. */
int pointcloud_q15_build(const cplx_q15 *rd, const uint8_t *mask,
                         const FrontendParams *p, DoaPlan *doa,
                         double rd_scale, Detection *out, int max_out);
#endif
