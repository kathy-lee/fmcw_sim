#ifndef DOA_CELL_F32_H
#define DOA_CELL_F32_H
#include "fmcw.h"

/* Bartlett DoA for a single snapshot of a uniform linear array, in single
 * precision throughout — the form for a core whose FPU has no double, where
 * doa_cell.h's scan runs in software.
 *
 * w_grid[g] = exp(j*2*pi*d*sin((g-90) deg)), g = 0..180, for element
 * spacing d in wavelengths: the steering element of element e at grid
 * point g is w_grid[g]^e. The scan builds it by repeated multiplication
 * starting from 1, so it calls no sin or cos and needs 181 table entries
 * whatever the element count.
 *
 * The contract is doa_bartlett_cell's: the same spectral-maxima selection
 * (strict local maxima with the one-sided edge rule, argmax fallback on a
 * flat spectrum, strongest maximum wins, ties to the lower index). Returns
 * the grid index 0..180; azimuth in degrees is index - 90. spectrum is
 * caller scratch, length 181, and holds the full spectrum on return.
 *
 * The spectrum is not bit-identical to doa_bartlett_cell's, so where two
 * maxima are nearly equal, or the maximum is nearly flat, the index can
 * differ from the double-precision one. */
int doa_bartlett_cell_f32(const cplx *snap, const cplx *w_grid, int n_elem,
                          float *spectrum);
#endif
