#ifndef CFAR_H
#define CFAR_H
#include <stdint.h>
#include "params.h"

/* CA-CFAR detection mask: the C port of cfar_2d_ca() plus the min-range
 * gate from process_mimo_frame(). power: n_chirps x (n_samples/2) linear
 * power map (NCI output), row-major. mask: same shape, set to 1 where
 * (power + 1e-12) > cfar_alpha * training-mean, 0 elsewhere; range columns
 * below min_range_bins are always 0. Edge cells use replicated borders
 * (scipy mode="nearest"). Uses p's guard/training geometry and alpha. */
void cfar_ca_mask(const double *power, const FrontendParams *p, uint8_t *mask);
#endif
