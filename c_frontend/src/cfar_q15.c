/* Integer CA-CFAR. Contracts in cfar_q15.h; numerics notes here.
 *
 * The float test is  lin > alpha * (full - guard) / n_train.  Turning that
 * into integers means deciding where the divide goes, and the choice is
 * forced by int64's headroom rather than by taste.
 *
 * Clearing the denominator first looks tidier:
 *
 *     lin * n_train * 65536  >  alpha_q16 * (full - guard)
 *
 * but it does not fit. With 12 channels a power cell reaches
 * 12 * 2 * 32767^2 = 2.58e10; a 325-cell window with a 45-cell guard box
 * leaves n_train = 280, so (full - guard) reaches 7.2e12; and a calibrated
 * alpha near 14.8 gives alpha_q16 = 9.7e5. The right side is then 7.0e18
 * against INT64_MAX of 9.22e18 — a margin of 1.3x, which is not a margin.
 *
 * Dividing first instead:
 *
 *     noise = (full - guard) / n_train        (integer, truncating)
 *     lin << 16  >  alpha_q16 * noise
 *
 * bounds both sides by roughly 2.5e16, a margin of ~370x, and costs almost
 * nothing: the truncation discards at most n_train - 1 from a sum of order
 * n_train * lin, a relative error near 1e-9. The shift stays on the left so
 * the comparison keeps alpha's full Q16 precision rather than truncating
 * the threshold downward and biasing the mask toward detections.
 *
 * Edge handling is byte-identical in intent to cfar.c: replicate borders,
 * with the same quirk that outermost cells get copies of themselves in
 * their training ring and are therefore suppressed. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cfar_q15.h"

static int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

static int64_t alpha_fixed(double alpha)
{
    return (int64_t)llround(alpha * (double)(1 << CFAR_Q15_ALPHA_SHIFT));
}

int cfar_q15_plan_init(CfarPlanQ15 *pl, const FrontendParams *p)
{
    memset(pl, 0, sizeof *pl);
    int D = p->n_chirps, R = p->n_samples / 2;
    int Wd = p->cfar_guard_d + p->cfar_training_d;
    int Wr = p->cfar_guard_r + p->cfar_training_r;
    pl->pd = D + 2 * Wd;
    pl->pr = R + 2 * Wr;
    pl->alpha_q16 = alpha_fixed(p->cfar_alpha);
    pl->padded = malloc((size_t)pl->pd * pl->pr * sizeof(int64_t));
    pl->rowsum = malloc((size_t)pl->pd * R * sizeof(int64_t));
    pl->win_full = malloc((size_t)D * R * sizeof(int64_t));
    pl->win_guard = malloc((size_t)D * R * sizeof(int64_t));
    if (!pl->padded || !pl->rowsum || !pl->win_full || !pl->win_guard)
        { cfar_q15_plan_free(pl); return -1; }
    return 0;
}

void cfar_q15_plan_free(CfarPlanQ15 *pl)
{
    free(pl->padded); free(pl->rowsum);
    free(pl->win_full); free(pl->win_guard);
    pl->padded = pl->rowsum = pl->win_full = pl->win_guard = NULL;
}

/* Box sums of half-widths (hd, hr) around every map cell, written to out
 * (D x R) — the integer twin of cfar.c's box_sums. Because integer
 * addition is exact, the running window here produces exactly the same
 * totals as summing each window from scratch, which is what lets the
 * reference comparison be an equality rather than a tolerance. */
static void box_sums(const int64_t *padded, int64_t *rowsum, int PR,
                     int D, int R, int Wd, int Wr, int hd, int hr,
                     int64_t *out)
{
    int PD = D + 2 * Wd;
    for (int pd = 0; pd < PD; pd++) {
        const int64_t *row = padded + (size_t)pd * PR;
        int64_t *rs = rowsum + (size_t)pd * R;
        int lo = Wr - hr, hi = Wr + hr;
        int64_t s = 0;
        for (int c = lo; c <= hi; c++) s += row[c];
        rs[0] = s;
        for (int r = 1; r < R; r++) {
            s += row[hi + r] - row[lo + r - 1];
            rs[r] = s;
        }
    }
    for (int r = 0; r < R; r++) {
        int lo = Wd - hd, hi = Wd + hd;
        int64_t s = 0;
        for (int pd = lo; pd <= hi; pd++) s += rowsum[(size_t)pd * R + r];
        out[r] = s;
        for (int d = 1; d < D; d++) {
            s += rowsum[(size_t)(hi + d) * R + r]
               - rowsum[(size_t)(lo + d - 1) * R + r];
            out[(size_t)d * R + r] = s;
        }
    }
}

void cfar_q15_ca_mask(CfarPlanQ15 *pl, const int64_t *power,
                      const FrontendParams *p, uint8_t *mask)
{
    int D = p->n_chirps, R = p->n_samples / 2;
    int Gd = p->cfar_guard_d, Gr = p->cfar_guard_r;
    int Wd = Gd + p->cfar_training_d, Wr = Gr + p->cfar_training_r;
    int n_train = (2 * Wd + 1) * (2 * Wr + 1) - (2 * Gd + 1) * (2 * Gr + 1);

    for (int pd = 0; pd < pl->pd; pd++) {
        int sd = clampi(pd - Wd, D);
        const int64_t *src = power + (size_t)sd * R;
        int64_t *dst = pl->padded + (size_t)pd * pl->pr;
        for (int pr = 0; pr < pl->pr; pr++)
            dst[pr] = src[clampi(pr - Wr, R)];
    }
    box_sums(pl->padded, pl->rowsum, pl->pr, D, R, Wd, Wr, Wd, Wr, pl->win_full);
    box_sums(pl->padded, pl->rowsum, pl->pr, D, R, Wd, Wr, Gd, Gr, pl->win_guard);

    for (int d = 0; d < D; d++) {
        for (int r = 0; r < R; r++) {
            size_t i = (size_t)d * R + r;
            int64_t noise = (pl->win_full[i] - pl->win_guard[i]) / n_train;
            mask[i] = ((power[i] << CFAR_Q15_ALPHA_SHIFT)
                       > pl->alpha_q16 * noise);
        }
    }
    for (int d = 0; d < D; d++)
        for (int r = 0; r < p->min_range_bins && r < R; r++)
            mask[(size_t)d * R + r] = 0;
}

void cfar_q15_ca_mask_ref(const int64_t *power, const FrontendParams *p,
                          uint8_t *mask)
{
    int D = p->n_chirps, R = p->n_samples / 2;
    int Gd = p->cfar_guard_d, Gr = p->cfar_guard_r;
    int Wd = Gd + p->cfar_training_d, Wr = Gr + p->cfar_training_r;
    int n_full = (2 * Wd + 1) * (2 * Wr + 1);
    int n_inner = (2 * Gd + 1) * (2 * Gr + 1);
    int n_train = n_full - n_inner;
    int64_t alpha_q16 = alpha_fixed(p->cfar_alpha);

    for (int d = 0; d < D; d++) {
        for (int r = 0; r < R; r++) {
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
            mask[(size_t)d * R + r] =
                ((power[(size_t)d * R + r] << CFAR_Q15_ALPHA_SHIFT)
                 > alpha_q16 * noise);
        }
    }
    for (int d = 0; d < D; d++)
        for (int r = 0; r < p->min_range_bins && r < R; r++)
            mask[(size_t)d * R + r] = 0;
}
