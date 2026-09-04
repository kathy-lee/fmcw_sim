#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rd_q15.h"
#include "nci_q15.h"
#include "cfar_q15.h"
#include "pointcloud_q15.h"
#include "convert_q15.h"
#include "onepass_q15.h"
#include "check.h"

#define V 12
#define C 32
#define S 128
#define R (S / 2)

static uint32_t rng;
static int16_t sample(void)
{
    rng = rng * 1664525u + 1013904223u;
    return (int16_t)((rng >> 16) % 32769u) - 16384;
}

/* emit sink for the composition */
typedef struct { OnepassDet d[ONEPASS_MAX_DET]; int n; } Sink;
static void collect(void *ctx, const OnepassDet *d)
{
    Sink *s = ctx;
    if (s->n < ONEPASS_MAX_DET) s->d[s->n++] = *d;
}

int main(void)
{
    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_virtual = V; p.n_chirps = C; p.n_samples = S;
    p.n_tx = 3; p.n_rx = 4;
    p.doa_n_sources = 1;
    p.cfar_guard_d = 2; p.cfar_guard_r = 4;
    p.cfar_training_d = 4; p.cfar_training_r = 8;
    p.min_range_bins = 6;
    p.range_per_bin = 0.075;
    p.cfar_alpha = 12.0;
    p.adc_full_scale = 1.0;
    for (int d = 0; d < C; d++)
        p.velocity_axis[d] = -8.0 + 0.5 * d;
    p.n_velocity = C;
    for (int e = 0; e < V; e++) p.pos_wl[e] = 0.5 * e;
    p.n_pos = V;

    /* frozen chain storage */
    static cplx_q15 in[V * C * S], cube[V * C * R];
    static int64_t power_ref[C * R];
    static uint8_t mask[C * R];
    static Detection det_ref[C * R];

    /* composition storage (host: separate allocations, no overlay) */
    static cplx_q15 planes[V * C * R];
    static int64_t power_op[C * R];
    static CfarCellQ15 cells[ONEPASS_MAX_DET];
    static cplx_q15 snapshots[ONEPASS_MAX_DET * V];
    static double spectrum[181];
    static cplx_q15 rowbuf[S];

    RdPlanQ15 rdp;
    CHECK(rd_q15_plan_init(&rdp, C, S) == 0);
    DoaPlan doa = {0};
    CHECK(doa_plan_init(&doa, p.pos_wl, V) == 0);
    double scale = q15_rd_scale(p.adc_full_scale, S, C);

    OnepassPlanQ15 op;
    CHECK(onepass_q15_init(&op, &p, planes, power_op, cells,
                           snapshots, spectrum, rowbuf) == 0);

    static const uint32_t seeds[] = { 11111u, 22222u, 33333u };
    for (int t = 0; t < 3; t++) {
        /* Noise plus one strong coherent target (range bin 20, shifted
         * Doppler bin 20, a linear phase across the array): NCI over 12
         * channels concentrates noise so tightly around its mean that
         * noise alone never crosses an honest threshold — without a real
         * target, n_ref >= 1 below would compare nothing. */
        rng = seeds[t];
        for (int v = 0; v < V; v++)
            for (int c = 0; c < C; c++)
                for (int s = 0; s < S; s++) {
                    double th = 2.0 * FMCW_PI
                                * (20.0 * s / S + 4.0 * c / C + 0.25 * v);
                    cplx_q15 *q = &in[((size_t)v * C + c) * S + s];
                    q->re = (int16_t)(sample() + (int)(6000.0 * cos(th)));
                    q->im = (int16_t)(sample() + (int)(6000.0 * sin(th)));
                }

        /* referee: the frozen stage-wise chain */
        long long sat_ref = 0;
        for (int v = 0; v < V; v++)
            rd_q15_process_channel(&rdp, in + (size_t)v * C * S,
                                   cube + (size_t)v * C * R, &sat_ref);
        nci_q15_integrate(cube, V, C * R, power_ref);
        cfar_q15_ca_mask_ref(power_ref, &p, mask);
        int n_ref = pointcloud_q15_build(cube, mask, &p, &doa, scale,
                                         det_ref, C * R);

        /* the composition, rows arriving v-major */
        op.sat = 0;
        for (int v = 0; v < V; v++) {
            for (int c = 0; c < C; c++) {
                memcpy(rowbuf, in + ((size_t)v * C + c) * S,
                       sizeof(cplx_q15) * S);
                onepass_q15_row(&op, v, c);
            }
        }
        Sink sink = { .n = 0 };
        int n_op = onepass_q15_finish(&op, collect, &sink);

        /* the resident planes are the frozen cube, bit for bit */
        CHECK(memcmp(planes, cube, sizeof cube) == 0);
        CHECK(op.sat == sat_ref);
        CHECK(op.overflow == 0);

        CHECK(n_op == n_ref);
        CHECK(n_op == sink.n);
        CHECK(n_ref >= 1);            /* a silent all-zero pass proves nothing */
        for (int i = 0; i < n_op; i++) {
            CHECK(sink.d[i].range_bin == det_ref[i].range_bin);
            CHECK(sink.d[i].doppler_bin == det_ref[i].doppler_bin);
            CHECK((double)(sink.d[i].az_idx - 90) == det_ref[i].azimuth_deg);
            CHECK(sink.d[i].x == det_ref[i].x);
            CHECK(sink.d[i].y == det_ref[i].y);
            CHECK(sink.d[i].v_radial == det_ref[i].v_radial);
            CHECK(sink.d[i].snr_db == det_ref[i].snr_db);
            CHECK(sink.d[i].power ==
                  power_ref[(size_t)sink.d[i].doppler_bin * R
                            + sink.d[i].range_bin]);
        }
    }

    onepass_q15_free(&op);
    doa_plan_free(&doa);
    rd_q15_plan_free(&rdp);
    CHECK_DONE();
}
