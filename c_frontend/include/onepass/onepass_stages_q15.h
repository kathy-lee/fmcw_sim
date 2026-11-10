#ifndef ONEPASS_STAGES_Q15_H
#define ONEPASS_STAGES_Q15_H
#include "fmcw.h"
#include "onepass_q15.h"

/* onepass_q15_finish split into its stages, so a caller can time each one
 * and choose the DoA scan. Called in this order on a plan whose rows have
 * all been consumed,
 *
 *     onepass_q15_power(p);
 *     n = onepass_q15_cells(p);
 *     for (i = 0; i < n; i++) onepass_q15_det(p, i, onepass_doa_f64, NULL, &d);
 *
 * they run the same expressions in the same order as onepass_q15_finish and
 * give the same detections to the last bit.
 *
 * The storage contract is the finish's: power is dead once
 * onepass_q15_cells returns, and every snapshot is gathered there, before
 * any DoA scan writes its spectrum. */

/* The DoA scan for one compensated snapshot of p->prm->n_virtual elements.
 * Returns the grid index 0..180; azimuth in degrees is index - 90. */
typedef int (*OnepassDoaFn)(void *ctx, const cplx *snap,
                            const OnepassPlanQ15 *p);

/* Sum every plane's squared magnitudes into p->power. */
void onepass_q15_power(OnepassPlanQ15 *p);

/* CFAR over p->power into p->cells, then gather each cell's snapshot.
 * Sets p->n_cells and p->overflow; returns the cell count. */
int onepass_q15_cells(OnepassPlanQ15 *p);

/* Detection i, 0 <= i < p->n_cells: TDM compensation, SNR, the DoA scan
 * and Cartesian position. */
void onepass_q15_det(OnepassPlanQ15 *p, int i, OnepassDoaFn doa,
                     void *doa_ctx, OnepassDet *out);

/* The double-precision Bartlett scan of onepass_q15_finish, writing
 * p->spectrum. ctx is unused. */
int onepass_doa_f64(void *ctx, const cplx *snap, const OnepassPlanQ15 *p);
#endif
