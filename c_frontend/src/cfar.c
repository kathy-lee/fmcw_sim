/* CA-CFAR. Numerics notes:
 *
 * The +1e-12 mirrors Python exactly: process_mimo_frame computes
 * 10*log10(power + 1e-12) and cfar_2d_ca converts straight back with
 * 10^(db/10), so the linear value it actually thresholds is power + 1e-12.
 * C skips the dB round-trip — same math, one less lossy conversion.
 *
 * Edge handling: scipy uniform_filter(mode="nearest") samples clamped
 * indices outside the map; the clamp below reproduces it exactly,
 * including its known quirk that cells on the outermost rows/columns get
 * copies of themselves in their training ring (inflated noise estimate,
 * suppressed detections there — pinned by the corner case in test_cfar).
 *
 * The training sum is (full window) - (guard window), both accumulated in
 * double; guard membership is decided on UNclamped coordinates, matching
 * the two-uniform_filter formulation. Brute force O(D*R*window) —
 * milliseconds at 64x128; a summed-area table would drop the window
 * factor but is not needed at this size. */
#include <stddef.h>
#include "cfar.h"

static int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

void cfar_ca_mask(const double *power, const FrontendParams *p, uint8_t *mask)
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
