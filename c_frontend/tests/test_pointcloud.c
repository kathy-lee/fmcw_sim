#include <string.h>
#include "params.h"
#include "doa.h"
#include "pointcloud.h"
#include "check.h"

int main(void)
{
    enum { V = 4, C = 8, R = 6 };
    cplx stack[V * C * R];
    uint8_t mask[C * R];
    double ref[3 * 7];              /* rows: r, d, az, x, y, v_radial, snr */
    cplx snap0_ref[V];
    CHECK(load_bin("tests/vectors/pc_in.bin", stack, sizeof stack) == 0);
    CHECK(load_bin("tests/vectors/pc_mask.bin", mask, sizeof mask) == 0);
    CHECK(load_bin("tests/vectors/pc_out.bin", ref, sizeof ref) == 0);
    CHECK(load_bin("tests/vectors/pc_snap0.bin", snap0_ref, sizeof snap0_ref) == 0);

    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_virtual = V; p.n_chirps = C; p.n_samples = 2 * R;
    p.n_tx = 2; p.n_rx = 2; p.doa_n_sources = 1;
    p.range_per_bin = 0.3;
    for (int d = 0; d < C; d++)                 /* linspace(-2, 1.5, 8) */
        p.velocity_axis[d] = -2.0 + 0.5 * d;
    double pos[V] = {0.0, 0.5, 1.0, 1.5};

    /* TDM compensation alone vs the Python-compensated first snapshot
     * (cell d=0, r=5): the phase formula is the whole trick here. */
    cplx raw[V], comp[V];
    for (int v = 0; v < V; v++) raw[v] = stack[v * C * R + 0 * R + 5];
    tdm_compensate(raw, comp, V, 0, C, 2, 2);
    for (int v = 0; v < V; v++) {
        CHECK_CLOSE(comp[v].re, snap0_ref[v].re, 1e-6);
        CHECK_CLOSE(comp[v].im, snap0_ref[v].im, 1e-6);
    }

    /* Full assembly vs the Python-computed rows (argwhere order). */
    DoaPlan doa;
    CHECK(doa_plan_init(&doa, pos, V) == 0);
    Detection det[C * R];
    int n = pointcloud_build(stack, mask, &p, &doa, det, C * R);
    CHECK(n == 3);
    for (int i = 0; i < n; i++) {
        const double *e = ref + 7 * i;
        CHECK(det[i].range_bin == (int)e[0]);
        CHECK(det[i].doppler_bin == (int)e[1]);
        CHECK(det[i].azimuth_deg == e[2]);       /* grid angle: exact */
        CHECK_CLOSE(det[i].x, e[3], 1e-4 * (fabs(e[3]) > 1.0 ? fabs(e[3]) : 1.0));
        CHECK_CLOSE(det[i].y, e[4], 1e-4 * (fabs(e[4]) > 1.0 ? fabs(e[4]) : 1.0));
        CHECK_CLOSE(det[i].v_radial, e[5], 1e-12);
        CHECK_CLOSE(det[i].snr_db, e[6], 1e-4 * (fabs(e[6]) > 1.0 ? fabs(e[6]) : 1.0));
    }
    doa_plan_free(&doa);
    CHECK_DONE();
}
