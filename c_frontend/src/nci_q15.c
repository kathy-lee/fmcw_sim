/* Integer NCI. Channel-major accumulation (v outer, cell inner) mirrors
 * nci.c, walking both arrays sequentially. The float version orders its
 * summation to match NumPy's for bit-stability; here the order is
 * irrelevant to the result, since integer addition is exact and
 * associative — one of the few places fixed point is unambiguously the
 * simpler proposition. */
#include <stddef.h>    /* size_t — fmcw_q15.h's chain provides only stdint */
#include "nci_q15.h"

void nci_q15_integrate(const cplx_q15 *rd, int n_virtual, int n_cells,
                       int64_t *power)
{
    for (int i = 0; i < n_cells; i++) power[i] = 0;
    for (int v = 0; v < n_virtual; v++) {
        const cplx_q15 *ch = rd + (size_t)v * (size_t)n_cells;
        for (int i = 0; i < n_cells; i++) power[i] += cq15_mag2(ch[i]);
    }
}
