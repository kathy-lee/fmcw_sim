#ifndef CFAR_H
#define CFAR_H
#include <stdint.h>
#include "params.h"

/* CA-CFAR: the C port of cfar_2d_ca() plus the min-range gate from
 * process_mimo_frame(). Input is the n_chirps x (n_samples/2) linear power
 * map (NCI output), row-major; output mask is the same shape, 1 where
 * (power + 1e-12) > cfar_alpha * training-mean, 0 elsewhere, with range
 * columns below min_range_bins always 0. Edge cells see replicated borders
 * (scipy mode="nearest").
 *
 * Two implementations with identical results on any cell whose ratio is
 * not within float-rounding distance of the threshold:
 *   cfar_ca_mask     — separable sliding window sums over a padded copy,
 *                      O(1) per cell; the production path.
 *   cfar_ca_mask_ref — direct per-cell window loops, O(window) per cell;
 *                      the readable reference the fast path is tested
 *                      against. */

/* Scratch for the sliding-sum path, sized once from the config dimensions
 * (no allocation per frame). */
typedef struct {
    int pd, pr;        /* padded dims: (n_d + 2*Wd) x (n_r + 2*Wr) */
    double *padded;    /* pd x pr edge-replicated copy of power + 1e-12 */
    double *rowsum;    /* pd x n_r horizontal window sums */
    double *win_full;  /* n_d x n_r full-window sums */
    double *win_guard; /* n_d x n_r guard-window sums */
} CfarPlan;

/* Allocate scratch for the dimensions/geometry in *p. Returns 0, or -1 on
 * allocation failure (plan then safe to pass to cfar_plan_free). */
int cfar_plan_init(CfarPlan *pl, const FrontendParams *p);

/* Release plan memory; safe after a failed init. */
void cfar_plan_free(CfarPlan *pl);

/* Production mask: sliding-sum implementation. plan must have been built
 * from the same params; its scratch is overwritten (one plan per thread). */
void cfar_ca_mask(CfarPlan *plan, const double *power,
                  const FrontendParams *p, uint8_t *mask);

/* Reference mask: naive window loops, no scratch. Same contract. */
void cfar_ca_mask_ref(const double *power, const FrontendParams *p,
                      uint8_t *mask);
#endif
