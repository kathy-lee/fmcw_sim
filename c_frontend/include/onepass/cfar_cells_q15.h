#ifndef CFAR_CELLS_Q15_H
#define CFAR_CELLS_Q15_H
#include <stdint.h>
#include "params.h"

/* CA-CFAR as a cell walk instead of a mask: same geometry, edge
 * replication, integer threshold and min-range gate as cfar_q15_ca_mask_ref,
 * but zero scratch and a capped output — the walk recomputes each window
 * directly (O(window) per cell) and emits passing cells into a fixed list.
 *
 * The cap is a ranked cap: when more cells pass than fit, the kept set is
 * the top max_cells by margin (power<<16) - thresh_q16, and *overflow is
 * set. Output is sorted by (doppler_bin, range_bin) — the same scan order
 * a mask walk would visit — so downstream consumers see the mask path's
 * ordering whenever nothing spilled. */

typedef struct {
    int16_t doppler_bin, range_bin;
    int64_t power;        /* the integrated cell value, captured here
                             because the map's storage may be reused */
    int64_t thresh_q16;   /* alpha_q16 * noise; the decision was
                             (power << CFAR_Q15_ALPHA_SHIFT) > thresh_q16 */
} CfarCellQ15;

int cfar_cells_q15(const int64_t *power, const FrontendParams *p,
                   CfarCellQ15 *cells, int max_cells, int *overflow);
#endif
