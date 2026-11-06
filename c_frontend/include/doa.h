#ifndef DOA_H
#define DOA_H
#include "fmcw.h"

/* Bartlett DoA: the C port of doa.py's "bartlett" path. The steering
 * matrix a(theta)[e] = exp(j*2*pi*pos_wl[e]*sin(theta)) is precomputed for
 * the fixed 181-point grid (-90..+90 deg, 1 deg step) — it depends only on
 * the array geometry, never on the frame. */
typedef struct {
    int n_grid, n_elem;     /* 181, V */
    cplx *steer;            /* n_grid rows x n_elem, row-major */
    double *spectrum;       /* scratch, n_grid; last doa_bartlett result */
} DoaPlan;

/* Build the steering matrix and scratch. Returns 0, or -1 on allocation
 * failure (plan then safe to pass to doa_plan_free). */
int doa_plan_init(DoaPlan *p, const double *pos_wl, int n_elem);

/* Release plan memory; safe after a failed init. */
void doa_plan_free(DoaPlan *p);

/* P(theta) = |a(theta)^H snap|^2 over the grid, then select the spectral
 * maxima: strict local maxima (one-sided rule at the grid edges, ties are
 * not maxima), argmax fallback if none, top n_sources by power, returned
 * sorted ascending in az_deg. Returns the number of angles written (>= 1).
 * Faithful port of doa.py's _peaks — unrelated to clustering.py's
 * "peak-picking" method, which operates on RD cells. Uses the plan's
 * scratch — one plan must not be shared across threads. */
int doa_bartlett(DoaPlan *p, const cplx *snap, int n_sources, double *az_deg);
#endif
