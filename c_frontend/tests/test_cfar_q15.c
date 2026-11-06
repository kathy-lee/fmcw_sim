#include <math.h>
#include <string.h>
#include "cfar.h"
#include "cfar_q15.h"
#include "check.h"

#define D 16
#define R 16

static uint32_t rng = 987654321u;
static int64_t noise_cell(void)
{
    rng = rng * 1103515245u + 12345u;
    return 1000000 + (int64_t)((rng >> 16) % 100001u) - 50000;   /* 1e6 +/- 5% */
}

int main(void)
{
    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_chirps = D;
    p.n_samples = 2 * R;
    p.cfar_guard_d = 1;
    p.cfar_guard_r = 1;
    p.cfar_training_d = 2;
    p.cfar_training_r = 2;
    p.min_range_bins = 2;
    p.cfar_alpha = 4.0;

    static int64_t power[D * R];
    static double powerf[D * R];
    static uint8_t m_fast[D * R], m_ref[D * R], m_float[D * R];

    for (int i = 0; i < D * R; i++) power[i] = noise_cell();
    power[8 * R + 8] = 1000000000;          /* interior target */
    power[0 * R + (R - 1)] = 1000000000;    /* corner target   */

    CfarPlanQ15 pl = {0};
    CHECK(cfar_q15_plan_init(&pl, &p) == 0);
    cfar_q15_ca_mask(&pl, power, &p, m_fast);
    cfar_q15_ca_mask_ref(power, &p, m_ref);

    /* ---- sliding == reference, EXACTLY ----
     *
     * The headline property of the integer port. Integer addition is exact,
     * so a running window sum and a from-scratch sum of the same cells are
     * the same number — no tolerance, no borderline exemption, unlike the
     * float pair which differ by ~1e-12 relative. */
    for (int i = 0; i < D * R; i++) CHECK(m_fast[i] == m_ref[i]);

    /* ---- detection behaviour ---- */
    CHECK(m_fast[8 * R + 8] == 1);          /* interior target found */

    /* The corner target is suppressed, and not because it is weak: edge
     * replication clones an outermost cell into its own training ring
     * (12 of the 40 training cells here), so its noise estimate rises with
     * it and the ratio cannot clear alpha at any amplitude. Pinned here the
     * same way test_cfar pins it for the float path. */
    CHECK(m_fast[0 * R + (R - 1)] == 0);

    /* Flat noise stays below threshold. */
    int detections = 0;
    for (int i = 0; i < D * R; i++) detections += m_fast[i];
    CHECK(detections == 1);

    /* The min-range gate zeroes the low columns unconditionally. */
    for (int d = 0; d < D; d++)
        for (int r = 0; r < p.min_range_bins; r++)
            CHECK(m_fast[d * R + r] == 0);

    /* ---- agreement with the float implementation ----
     *
     * Same power map as float64 (exact: these values are far inside 2^53).
     * The two can only disagree where the cell sits within the combined
     * slack of alpha's Q16 rounding and the integer noise truncation, so
     * cells whose float ratio is within 1e-4 of 1.0 are exempt — the same
     * borderline rule the Python comparison uses. */
    for (int i = 0; i < D * R; i++) powerf[i] = (double)power[i];

    CfarPlan plf = {0};
    CHECK(cfar_plan_init(&plf, &p) == 0);
    cfar_ca_mask(&plf, powerf, &p, m_float);

    int Gd = p.cfar_guard_d, Gr = p.cfar_guard_r;
    int Wd = Gd + p.cfar_training_d, Wr = Gr + p.cfar_training_r;
    int n_train = (2 * Wd + 1) * (2 * Wr + 1) - (2 * Gd + 1) * (2 * Gr + 1);
    int exempt = 0;
    for (int d = 0; d < D; d++) {
        for (int r = p.min_range_bins; r < R; r++) {
            /* Recompute the float noise directly, so the comparison does
             * not lean on either implementation's internals. */
            double sum_full = 0.0, sum_inner = 0.0;
            for (int dd = d - Wd; dd <= d + Wd; dd++) {
                int cd = dd < 0 ? 0 : (dd >= D ? D - 1 : dd);
                for (int rr = r - Wr; rr <= r + Wr; rr++) {
                    int cr = rr < 0 ? 0 : (rr >= R ? R - 1 : rr);
                    double v = powerf[cd * R + cr] + 1e-12;
                    sum_full += v;
                    if (dd >= d - Gd && dd <= d + Gd &&
                        rr >= r - Gr && rr <= r + Gr) sum_inner += v;
                }
            }
            double noise = (sum_full - sum_inner) / n_train;
            double ratio = (powerf[d * R + r] + 1e-12) / (p.cfar_alpha * noise);
            if (fabs(ratio - 1.0) <= 1e-4) { exempt++; continue; }
            CHECK(m_fast[d * R + r] == m_float[d * R + r]);
        }
    }
    printf("borderline cells exempted: %d\n", exempt);

    cfar_plan_free(&plf);
    cfar_q15_plan_free(&pl);
    CHECK_DONE();
}
