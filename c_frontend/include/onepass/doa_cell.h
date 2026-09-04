#ifndef DOA_CELL_H
#define DOA_CELL_H
#include "fmcw.h"

/* Bartlett DoA for a single snapshot with the steering vector computed on
 * the fly — the memory-lean sibling of doa.h's precomputed-matrix form,
 * for callers that cannot afford the 181 x V steering table. Arithmetic is
 * doa_plan_init + doa_bartlett's exactly: the same double sin/cos cast to
 * float per element, the same double accumulation, so the two agree to the
 * last bit on the same snapshot.
 *
 * Selection is the spectral-maxima rule from doa.c, fixed at one source:
 * strict local maxima with the one-sided edge rule, argmax fallback on a
 * flat spectrum, strongest maximum wins (ties to the lower index, as the
 * frozen top-1 scan resolves them). Returns the grid index 0..180; azimuth
 * in degrees is index - 90. spectrum is caller scratch, length 181, and
 * holds the full spectrum on return. */
int doa_bartlett_cell(const cplx *snap, const double *pos_wl, int n_elem,
                      double *spectrum);
#endif
