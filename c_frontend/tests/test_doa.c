#include "fmcw.h"
#include "doa.h"
#include "check.h"

int main(void)
{
    /* Golden two-source snapshot: spectrum within float32 rounding of the
     * NumPy reference, peak ANGLES exactly equal — they are grid indices,
     * so any mismatch means the _peaks port (edge rule, tie rule, top-n,
     * ascending sort) is wrong, not float noise. */
    double pos[8], spec_ref[181], peaks_ref[2], az[181];
    cplx snap[8];
    CHECK(load_bin("tests/vectors/doa_pos.bin", pos, sizeof pos) == 0);
    CHECK(load_bin("tests/vectors/doa_snap.bin", snap, sizeof snap) == 0);
    CHECK(load_bin("tests/vectors/doa_spec.bin", spec_ref, sizeof spec_ref) == 0);
    CHECK(load_bin("tests/vectors/doa_peaks.bin", peaks_ref, sizeof peaks_ref) == 0);

    DoaPlan plan;
    CHECK(doa_plan_init(&plan, pos, 8) == 0);
    CHECK(plan.n_grid == 181);

    int k = doa_bartlett(&plan, snap, 2, az);
    CHECK(k == 2);
    CHECK(az[0] == peaks_ref[0]);
    CHECK(az[1] == peaks_ref[1]);

    double smax = 0.0;
    for (int i = 0; i < 181; i++) if (spec_ref[i] > smax) smax = spec_ref[i];
    for (int i = 0; i < 181; i++)
        CHECK_CLOSE(plan.spectrum[i], spec_ref[i], 1e-5 * smax);
    doa_plan_free(&plan);

    /* Flat spectrum (all elements at position 0 -> P constant): no strict
     * local maximum exists, so the argmax fallback must return exactly one
     * angle, the first grid point (-90), matching NumPy argmax-first. */
    double pos0[3] = {0.0, 0.0, 0.0};
    cplx ones[3] = {{1, 0}, {1, 0}, {1, 0}};
    DoaPlan flat;
    CHECK(doa_plan_init(&flat, pos0, 3) == 0);
    k = doa_bartlett(&flat, ones, 2, az);
    CHECK(k == 1);
    CHECK(az[0] == -90.0);
    doa_plan_free(&flat);
    CHECK_DONE();
}
