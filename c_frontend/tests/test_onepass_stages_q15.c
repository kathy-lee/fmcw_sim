#include <math.h>
#include <stdint.h>
#include <string.h>
#include "onepass_q15.h"
#include "onepass_stages_q15.h"
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

typedef struct { OnepassDet d[ONEPASS_MAX_DET]; int n; } Sink;
static void collect(void *ctx, const OnepassDet *d)
{
    Sink *s = ctx;
    if (s->n < ONEPASS_MAX_DET) s->d[s->n++] = *d;
}

static cplx_q15 planes[V * C * R];
static int64_t power[C * R];
static CfarCellQ15 cells[ONEPASS_MAX_DET];
static cplx_q15 snapshots[ONEPASS_MAX_DET * V];
static double spectrum[181];
static cplx_q15 rowbuf[S];

static void params(FrontendParams *p)
{
    memset(p, 0, sizeof *p);
    p->n_virtual = V; p->n_chirps = C; p->n_samples = S;
    p->n_tx = 3; p->n_rx = 4;
    p->doa_n_sources = 1;
    p->cfar_guard_d = 2; p->cfar_guard_r = 4;
    p->cfar_training_d = 4; p->cfar_training_r = 8;
    p->min_range_bins = 6;
    p->range_per_bin = 0.075;
    p->cfar_alpha = 12.0;
    p->adc_full_scale = 1.0;
    for (int d = 0; d < C; d++) p->velocity_axis[d] = -8.0 + 0.5 * d;
    p->n_velocity = C;
    for (int e = 0; e < V; e++) p->pos_wl[e] = 0.5 * e;
    p->n_pos = V;
}

/* Noise plus one strong coherent target, fed row by row, so the planes
 * hold at least one detection. */
static void fill_planes(OnepassPlanQ15 *op, uint32_t seed)
{
    rng = seed;
    op->sat = 0;
    for (int v = 0; v < V; v++)
        for (int c = 0; c < C; c++) {
            for (int s = 0; s < S; s++) {
                double th = 2.0 * FMCW_PI
                            * (20.0 * s / S + 4.0 * c / C + 0.25 * v);
                rowbuf[s].re = (int16_t)(sample() + (int)(6000.0 * cos(th)));
                rowbuf[s].im = (int16_t)(sample() + (int)(6000.0 * sin(th)));
            }
            onepass_q15_row(op, v, c);
        }
}

static void test_stages_match_finish(OnepassPlanQ15 *op)
{
    static const uint32_t seeds[] = { 11111u, 22222u, 33333u };
    for (int t = 0; t < 3; t++) {
        fill_planes(op, seeds[t]);

        Sink ref = { .n = 0 };
        int n_ref = onepass_q15_finish(op, collect, &ref);
        int overflow_ref = op->overflow;
        CHECK(n_ref >= 1);            /* an empty list would compare nothing */

        /* the planes are untouched by the finish, so the staged path runs
         * on the same input */
        onepass_q15_power(op);
        int n = onepass_q15_cells(op);
        CHECK(n == n_ref);
        CHECK(op->n_cells == n_ref);
        CHECK(op->overflow == overflow_ref);
        for (int i = 0; i < n && i < n_ref; i++) {
            OnepassDet d;
            onepass_q15_det(op, i, onepass_doa_f64, NULL, &d);
            const OnepassDet *r = &ref.d[i];
            CHECK(d.range_bin == r->range_bin);
            CHECK(d.doppler_bin == r->doppler_bin);
            CHECK(d.az_idx == r->az_idx);
            CHECK(d.x == r->x);
            CHECK(d.y == r->y);
            CHECK(d.v_radial == r->v_radial);
            CHECK(d.snr_db == r->snr_db);
            CHECK(d.power == r->power);
            CHECK(d.thresh_q16 == r->thresh_q16);
        }
    }
}

static void test_stages_zero_detections(OnepassPlanQ15 *op)
{
    memset(planes, 0, sizeof planes);
    onepass_q15_power(op);
    CHECK(onepass_q15_cells(op) == 0);
    CHECK(op->n_cells == 0);
    CHECK(op->overflow == 0);
}

int main(void)
{
    FrontendParams p;
    params(&p);
    OnepassPlanQ15 op;
    CHECK(onepass_q15_init(&op, &p, planes, power, cells,
                           snapshots, spectrum, rowbuf) == 0);

    test_stages_match_finish(&op);
    test_stages_zero_detections(&op);

    onepass_q15_free(&op);
    CHECK_DONE();
}
