/* NCI. Channel-major accumulation (v outer, cell inner) keeps the per-cell
 * summation order identical to NumPy's axis-0 sum, so the golden comparison
 * is bit-stable, and walks both arrays sequentially (cache-friendly). */
#include "nci.h"

void nci_integrate(const cplx *rd, int n_virtual, int n_cells, double *power)
{
    for (int i = 0; i < n_cells; i++) power[i] = 0.0;
    for (int v = 0; v < n_virtual; v++) {
        const cplx *ch = rd + (size_t)v * (size_t)n_cells;
        for (int i = 0; i < n_cells; i++) power[i] += cmag2(ch[i]);
    }
}
