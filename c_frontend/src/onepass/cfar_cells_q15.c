/* The cell walk shares cfar_q15_ca_mask_ref's arithmetic to the letter —
 * same clamped window loops, same truncating divide, same Q16 alpha
 * rounding — so the two agree cell for cell and the list needs no
 * tolerance against the mask. What it adds is the ranked cap: a full map
 * of detections cannot be stored where this walk is meant to run, so the
 * strongest survivors by margin are kept and the spill is reported rather
 * than silently dropped. */
#include <math.h>
#include <stddef.h>
#include "cfar_q15.h"
#include "cfar_cells_q15.h"

static int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

static int64_t margin(const CfarCellQ15 *c)
{
    return (c->power << CFAR_Q15_ALPHA_SHIFT) - c->thresh_q16;
}

int cfar_cells_q15(const int64_t *power, const FrontendParams *p,
                   CfarCellQ15 *cells, int max_cells, int *overflow)
{
    int D = p->n_chirps, R = p->n_samples / 2;
    int Gd = p->cfar_guard_d, Gr = p->cfar_guard_r;
    int Wd = Gd + p->cfar_training_d, Wr = Gr + p->cfar_training_r;
    int n_train = (2 * Wd + 1) * (2 * Wr + 1) - (2 * Gd + 1) * (2 * Gr + 1);
    int64_t alpha_q16 =
        (int64_t)llround(p->cfar_alpha * (double)(1 << CFAR_Q15_ALPHA_SHIFT));
    int n = 0;
    *overflow = 0;

    for (int d = 0; d < D; d++) {
        for (int r = p->min_range_bins; r < R; r++) {
            int64_t sum_full = 0, sum_inner = 0;
            for (int dd = d - Wd; dd <= d + Wd; dd++) {
                int cd = clampi(dd, D);
                for (int rr = r - Wr; rr <= r + Wr; rr++) {
                    int cr = clampi(rr, R);
                    int64_t v = power[(size_t)cd * R + cr];
                    sum_full += v;
                    if (dd >= d - Gd && dd <= d + Gd &&
                        rr >= r - Gr && rr <= r + Gr)
                        sum_inner += v;
                }
            }
            int64_t noise = (sum_full - sum_inner) / n_train;
            int64_t pw = power[(size_t)d * R + r];
            int64_t thresh = alpha_q16 * noise;
            if ((pw << CFAR_Q15_ALPHA_SHIFT) <= thresh) continue;

            CfarCellQ15 c = { (int16_t)d, (int16_t)r, pw, thresh };
            if (n < max_cells) {
                cells[n++] = c;
            } else {
                *overflow = 1;
                int weakest = 0;
                for (int i = 1; i < n; i++)
                    if (margin(&cells[i]) < margin(&cells[weakest]))
                        weakest = i;
                if (margin(&c) > margin(&cells[weakest]))
                    cells[weakest] = c;
            }
        }
    }

    /* restore scan order (insertion sort; n <= max_cells, small) */
    for (int i = 1; i < n; i++) {
        CfarCellQ15 key = cells[i];
        int j = i - 1;
        while (j >= 0 && (cells[j].doppler_bin > key.doppler_bin ||
               (cells[j].doppler_bin == key.doppler_bin &&
                cells[j].range_bin > key.range_bin))) {
            cells[j + 1] = cells[j];
            j--;
        }
        cells[j + 1] = key;
    }
    return n;
}
