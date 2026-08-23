#ifndef CFAR_Q15_H
#define CFAR_Q15_H
#include <stdint.h>
#include "params.h"

/* CA-CFAR on an int64 power map — the integer counterpart of cfar.h, same
 * geometry, same edge replication, same min-range gate.
 *
 * Two things are simpler here than in the float version:
 *
 * - No epsilon. The float path adds 1e-12 to avoid dividing by a zero
 *   noise estimate; the integer test multiplies instead of dividing, so
 *   the hazard does not exist.
 * - The sliding and reference paths agree BIT-EXACTLY, not merely within
 *   rounding. Integer addition is exact, so a running window sum and a
 *   direct sum of the same cells are the same number — where the float
 *   versions differ by ~1e-12 relative and need a borderline exemption.
 *
 * The one thing that is harder: CFAR is scale-invariant, so the power map's
 * scale cancels and nothing here needs to know K, S or C — but alpha is a
 * float and must become fixed-point. See cfar_q15.c for the width choice
 * and its overflow bound. */

#define CFAR_Q15_ALPHA_SHIFT 16

typedef struct {
    int pd, pr;          /* padded dims: (n_d + 2*Wd) x (n_r + 2*Wr) */
    int64_t alpha_q16;   /* round(cfar_alpha * 65536), fixed at plan time */
    int64_t *padded;     /* pd x pr edge-replicated copy of the power map */
    int64_t *rowsum;     /* pd x n_r horizontal window sums */
    int64_t *win_full;   /* n_d x n_r full-window sums */
    int64_t *win_guard;  /* n_d x n_r guard-window sums */
} CfarPlanQ15;

/* Allocate scratch and fix alpha for the geometry in *p. Returns 0, or -1
 * on allocation failure (plan then safe to pass to cfar_q15_plan_free). */
int cfar_q15_plan_init(CfarPlanQ15 *pl, const FrontendParams *p);

/* Release plan memory; safe after a failed init. */
void cfar_q15_plan_free(CfarPlanQ15 *pl);

/* Production mask: sliding-sum implementation, O(1) per cell. */
void cfar_q15_ca_mask(CfarPlanQ15 *pl, const int64_t *power,
                      const FrontendParams *p, uint8_t *mask);

/* Reference mask: naive window loops, no scratch. Same contract, and
 * required to agree with the production path exactly. */
void cfar_q15_ca_mask_ref(const int64_t *power, const FrontendParams *p,
                          uint8_t *mask);
#endif
