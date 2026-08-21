#ifndef NCI_H
#define NCI_H
#include "fmcw.h"

/* Non-coherent integration: the C port of point_cloud.py's integrate().
 * rd holds n_virtual contiguous channel maps of n_cells cplx samples each
 * (the rd-cube buffer layout); power[i] = sum over v of |rd_v[i]|^2,
 * accumulated in double. power has n_cells entries. */
void nci_integrate(const cplx *rd, int n_virtual, int n_cells, double *power);
#endif
