#include <string.h>
#include "params.h"
#include "cfar.h"
#include "check.h"

int main(void)
{
    /* Golden mask with a FIXED alpha baked into the vector: this test pins
     * the window mechanics (training/guard sums, edge clamping, min-range
     * gate), not the alpha calibration — that stays in Python. */
    enum { D = 16, R = 24 };
    double power[D * R];
    uint8_t ref[D * R], mask[D * R];
    CHECK(load_bin("tests/vectors/cfar_power.bin", power, sizeof power) == 0);
    CHECK(load_bin("tests/vectors/cfar_mask.bin", ref, sizeof ref) == 0);

    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_chirps = D; p.n_samples = 2 * R;
    p.cfar_guard_d = 1; p.cfar_guard_r = 2;
    p.cfar_training_d = 2; p.cfar_training_r = 3;
    p.cfar_alpha = 8.0; p.min_range_bins = 4;

    cfar_ca_mask(power, &p, mask);

    for (int i = 0; i < D * R; i++) CHECK(mask[i] == ref[i]);
    CHECK(mask[5 * R + 10] == 1);    /* interior peak detected */
    CHECK(mask[12 * R + 21] == 1);   /* near-edge peak: clamped window detects */
    CHECK(mask[0 * R + 23] == 0);    /* corner: edge replication clones the cell
                                        into its own training ring -> suppressed */
    CHECK(mask[7 * R + 2] == 0);     /* strong but inside the min-range gate */
    CHECK_DONE();
}
