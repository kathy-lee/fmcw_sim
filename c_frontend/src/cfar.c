/* CA-CFAR. Numerics notes:
 *
 * The +1e-12 mirrors Python exactly: process_mimo_frame computes
 * 10*log10(power + 1e-12) and cfar_2d_ca converts straight back with
 * 10^(db/10), so the linear value it actually thresholds is power + 1e-12.
 * C skips the dB round-trip — same math, one less lossy conversion.
 *
 * Edge handling: scipy uniform_filter(mode="nearest") samples clamped
 * indices outside the map. The reference clamps per access; the sliding
 * path pads once with replicated borders — identical sums either way,
 * including the known quirk that cells on the outermost rows/columns get
 * copies of themselves in their training ring (inflated noise estimate,
 * suppressed detections there — pinned by the corner case in test_cfar).
 *
 * The sliding path is the standard practice optimization: as a window
 * steps one cell, its sum changes by the entering minus the leaving edge,
 * so per-cell cost is O(1) instead of O(window). Done separably (rows,
 * then columns) with accumulators bounded by one window's worth of data —
 * the form that survives a fixed-point port, unlike a summed-area table
 * whose entries grow with the whole map. The two orders of summation
 * differ only by float rounding (~1e-12 relative), far below the margin
 * of any non-borderline cell. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "cfar.h"

static int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

/* Size every scratch buffer from the config once. Zero first so
 * cfar_plan_free is safe on any partial-init failure path. */
int cfar_plan_init(CfarPlan *pl, const FrontendParams *p)
{
    memset(pl, 0, sizeof *pl);
    int D = p->n_chirps, R = p->n_samples / 2;
    int Wd = p->cfar_guard_d + p->cfar_training_d;
    int Wr = p->cfar_guard_r + p->cfar_training_r;
    pl->pd = D + 2 * Wd;
    pl->pr = R + 2 * Wr;
    pl->padded = malloc((size_t)pl->pd * pl->pr * sizeof(double));
    pl->rowsum = malloc((size_t)pl->pd * R * sizeof(double));
    pl->win_full = malloc((size_t)D * R * sizeof(double));
    pl->win_guard = malloc((size_t)D * R * sizeof(double));
    if (!pl->padded || !pl->rowsum || !pl->win_full || !pl->win_guard)
        { cfar_plan_free(pl); return -1; }
    return 0;
}

/* Release the scratch; free(NULL) makes this a no-op on a zeroed plan. */
void cfar_plan_free(CfarPlan *pl)
{
    free(pl->padded); free(pl->rowsum);
    free(pl->win_full); free(pl->win_guard);
    pl->padded = pl->rowsum = pl->win_full = pl->win_guard = NULL;
}

/* Box sums of half-widths (hd, hr) around every map cell, written to out
 * (D x R). Map cell (d, r) sits at padded (d + Wd, r + Wr); its box spans
 * padded rows d+Wd-hd .. d+Wd+hd and columns likewise. Horizontal pass:
 * per padded row, a running window over columns. Vertical pass: per
 * column, a running window over the row sums. */
static void box_sums(const double *padded, double *rowsum, int PR,
                     int D, int R, int Wd, int Wr, int hd, int hr,
                     double *out)
{
    int PD = D + 2 * Wd;
    for (int pd = 0; pd < PD; pd++) {
        const double *row = padded + (size_t)pd * PR;
        double *rs = rowsum + (size_t)pd * R;
        int lo = Wr - hr, hi = Wr + hr;
        double s = 0.0;
        for (int c = lo; c <= hi; c++) s += row[c];
        rs[0] = s;
        for (int r = 1; r < R; r++) {
            s += row[hi + r] - row[lo + r - 1];
            rs[r] = s;
        }
    }
    for (int r = 0; r < R; r++) {
        int lo = Wd - hd, hi = Wd + hd;
        double s = 0.0;
        for (int pd = lo; pd <= hi; pd++) s += rowsum[(size_t)pd * R + r];
        out[r] = s;
        for (int d = 1; d < D; d++) {
            s += rowsum[(size_t)(hi + d) * R + r]
               - rowsum[(size_t)(lo + d - 1) * R + r];
            out[(size_t)d * R + r] = s;
        }
    }
}

/* Production mask, O(1) per cell. Four steps: (1) copy power + 1e-12 into
 * the padded buffer, realizing the edge replication once; (2) sliding box
 * sums for the full window; (3) same for the guard window; (4) threshold
 * (full - guard)/n_train per cell, then zero the min-range columns. */
void cfar_ca_mask(CfarPlan *pl, const double *power,
                  const FrontendParams *p, uint8_t *mask)
{
    int D = p->n_chirps, R = p->n_samples / 2;
    int Gd = p->cfar_guard_d, Gr = p->cfar_guard_r;
    int Wd = Gd + p->cfar_training_d, Wr = Gr + p->cfar_training_r;
    int n_train = (2 * Wd + 1) * (2 * Wr + 1) - (2 * Gd + 1) * (2 * Gr + 1);

    for (int pd = 0; pd < pl->pd; pd++) {
        int sd = clampi(pd - Wd, D);
        const double *src = power + (size_t)sd * R;
        double *dst = pl->padded + (size_t)pd * pl->pr;
        for (int pr = 0; pr < pl->pr; pr++)
            dst[pr] = src[clampi(pr - Wr, R)] + 1e-12;
    }
    box_sums(pl->padded, pl->rowsum, pl->pr, D, R, Wd, Wr, Wd, Wr, pl->win_full);
    box_sums(pl->padded, pl->rowsum, pl->pr, D, R, Wd, Wr, Gd, Gr, pl->win_guard);

    for (int d = 0; d < D; d++) {
        for (int r = 0; r < R; r++) {
            size_t i = (size_t)d * R + r;
            double noise = (pl->win_full[i] - pl->win_guard[i]) / n_train;
            mask[i] = (power[i] + 1e-12 > p->cfar_alpha * noise);
        }
    }
    for (int d = 0; d < D; d++)
        for (int r = 0; r < p->min_range_bins && r < R; r++)
            mask[(size_t)d * R + r] = 0;
}

/* Reference mask, O(window) per cell: for each cell, walk the full window
 * with clamped indices, accumulating the total and the guard-box subtotal
 * (membership decided on UNclamped coordinates, matching the
 * two-uniform_filter formulation). Deliberately the transparent, check-by-
 * eye form — test_cfar holds the production path to cell-for-cell
 * agreement with it. */
void cfar_ca_mask_ref(const double *power, const FrontendParams *p,
                      uint8_t *mask)
{
    int D = p->n_chirps, R = p->n_samples / 2;
    int Gd = p->cfar_guard_d, Gr = p->cfar_guard_r;
    int Wd = Gd + p->cfar_training_d, Wr = Gr + p->cfar_training_r;
    int n_full = (2 * Wd + 1) * (2 * Wr + 1);
    int n_inner = (2 * Gd + 1) * (2 * Gr + 1);
    int n_train = n_full - n_inner;

    for (int d = 0; d < D; d++) {
        for (int r = 0; r < R; r++) {
            double sum_full = 0.0, sum_inner = 0.0;
            for (int dd = d - Wd; dd <= d + Wd; dd++) {
                int cd = clampi(dd, D);
                for (int rr = r - Wr; rr <= r + Wr; rr++) {
                    int cr = clampi(rr, R);
                    double v = power[(size_t)cd * R + cr] + 1e-12;
                    sum_full += v;
                    if (dd >= d - Gd && dd <= d + Gd &&
                        rr >= r - Gr && rr <= r + Gr)
                        sum_inner += v;
                }
            }
            double noise = (sum_full - sum_inner) / n_train;
            double lin = power[(size_t)d * R + r] + 1e-12;
            mask[(size_t)d * R + r] = (lin > p->cfar_alpha * noise);
        }
    }
    for (int d = 0; d < D; d++)
        for (int r = 0; r < p->min_range_bins && r < R; r++)
            mask[(size_t)d * R + r] = 0;
}
