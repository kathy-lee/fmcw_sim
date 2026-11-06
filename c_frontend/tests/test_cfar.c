#include <string.h>
#include "params.h"
#include "cfar.h"
#include "check.h"

int main(void)
{
    /* Golden mask with a FIXED alpha baked into the vector: this pins the
     * window mechanics (training/guard sums, edge clamping, min-range
     * gate), not the alpha calibration — that stays in Python. Both the
     * production (sliding-sum) and reference (naive) implementations must
     * reproduce it, and each other, cell for cell: the vector generator
     * guarantees no cell sits close enough to the threshold for their
     * different summation orders to disagree. */
    enum { D = 16, R = 24 };
    double power[D * R];
    uint8_t ref[D * R], mask_ref[D * R], mask_fast[D * R];
    CHECK(load_bin("tests/vectors/cfar_power.bin", power, sizeof power) == 0);
    CHECK(load_bin("tests/vectors/cfar_mask.bin", ref, sizeof ref) == 0);

    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_chirps = D; p.n_samples = 2 * R;
    p.cfar_guard_d = 1; p.cfar_guard_r = 2;
    p.cfar_training_d = 2; p.cfar_training_r = 3;
    p.cfar_alpha = 8.0; p.min_range_bins = 4;

    cfar_ca_mask_ref(power, &p, mask_ref);

    CfarPlan plan;
    CHECK(cfar_plan_init(&plan, &p) == 0);
    cfar_ca_mask(&plan, power, &p, mask_fast);
    cfar_plan_free(&plan);

    for (int i = 0; i < D * R; i++) {
        CHECK(mask_ref[i] == ref[i]);        /* reference vs golden */
        CHECK(mask_fast[i] == ref[i]);       /* sliding-sum vs golden */
        CHECK(mask_fast[i] == mask_ref[i]);  /* three-way agreement */
    }
    CHECK(mask_fast[5 * R + 10] == 1);   /* interior peak detected */
    CHECK(mask_fast[12 * R + 21] == 1);  /* near-edge peak: padded window detects */
    CHECK(mask_fast[0 * R + 23] == 0);   /* corner: edge replication clones the cell
                                            into its own training ring -> suppressed */
    CHECK(mask_fast[7 * R + 2] == 0);    /* strong but inside the min-range gate */
    CHECK_DONE();
}
