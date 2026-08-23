#ifndef NCI_Q15_H
#define NCI_Q15_H
#include <stdint.h>
#include "fmcw_q15.h"

/* Non-coherent integration in integers — the counterpart of nci.h.
 *
 * rd holds n_virtual contiguous channel maps of n_cells cplx_q15 samples;
 * power[i] = sum over v of (re^2 + im^2), accumulated in int64.
 *
 * This stage is EXACT: integer multiply-add introduces no rounding at all,
 * so unlike every other stage in the integer path it loses nothing to the
 * fixed-point port. int64 is required rather than cautious — one sample's
 * squared magnitude already reaches 2^31, past what int32 holds. */
void nci_q15_integrate(const cplx_q15 *rd, int n_virtual, int n_cells,
                       int64_t *power);
#endif
