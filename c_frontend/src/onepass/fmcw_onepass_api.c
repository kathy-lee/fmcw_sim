#include <stdlib.h>
#include <string.h>
#include "convert_q15.h"
#include "fmcw_onepass_api.h"
#include "onepass_q15.h"

typedef struct {
    FrontendParams prm;
    OnepassPlanQ15 op;
    cplx_q15 *planes;
    int64_t *power;
    CfarCellQ15 *cells;
    cplx_q15 *snapshots;
    double *spectrum;
    cplx_q15 *rowbuf;
} OnepassHandle;

void *fmcw_onepass_create(const char *params_cfg)
{
    if (!params_cfg) return NULL;
    OnepassHandle *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    if (params_load(params_cfg, &h->prm) != 0) { free(h); return NULL; }
    int V = h->prm.n_virtual, C = h->prm.n_chirps, R = h->prm.n_samples / 2;
    h->planes = malloc((size_t)V * C * R * sizeof(cplx_q15));
    h->power = malloc((size_t)C * R * sizeof(int64_t));
    h->cells = malloc(ONEPASS_MAX_DET * sizeof(CfarCellQ15));
    h->snapshots = malloc((size_t)ONEPASS_MAX_DET * V * sizeof(cplx_q15));
    h->spectrum = malloc(181 * sizeof(double));
    h->rowbuf = malloc((size_t)h->prm.n_samples * sizeof(cplx_q15));
    if (!h->planes || !h->power || !h->cells || !h->snapshots ||
        !h->spectrum || !h->rowbuf ||
        onepass_q15_init(&h->op, &h->prm, h->planes, h->power, h->cells,
                         h->snapshots, h->spectrum, h->rowbuf) != 0) {
        fmcw_onepass_destroy(h);
        return NULL;
    }
    return h;
}

void fmcw_onepass_destroy(void *vh)
{
    OnepassHandle *h = vh;
    if (!h) return;
    onepass_q15_free(&h->op);
    free(h->planes); free(h->power); free(h->cells);
    free(h->snapshots); free(h->spectrum); free(h->rowbuf);
    free(h);
}

int fmcw_onepass_dims(void *vh, int out[3])
{
    OnepassHandle *h = vh;
    if (!h || !out) return -1;
    out[0] = h->prm.n_virtual;
    out[1] = h->prm.n_chirps;
    out[2] = h->prm.n_samples;
    return 0;
}

int fmcw_onepass_row(void *vh, int v, int c, const int16_t *row)
{
    OnepassHandle *h = vh;
    if (!h || !row) return -1;
    if (v < 0 || v >= h->prm.n_virtual || c < 0 || c >= h->prm.n_chirps)
        return -1;
    memcpy(h->rowbuf, row, (size_t)h->prm.n_samples * sizeof(cplx_q15));
    onepass_q15_row(&h->op, v, c);
    return 0;
}

typedef struct { double *det; int n, max; } DetSink;

static void det_emit(void *ctx, const OnepassDet *d)
{
    DetSink *s = ctx;
    if (s->n >= s->max) return;
    double *row = s->det + (size_t)s->n * 7;
    row[0] = d->range_bin;
    row[1] = d->doppler_bin;
    row[2] = (double)(d->az_idx - 90);
    row[3] = d->x; row[4] = d->y;
    row[5] = d->v_radial; row[6] = d->snr_db;
    s->n++;
}

int fmcw_onepass_finish(void *vh, double *det, int max_out)
{
    OnepassHandle *h = vh;
    if (!h || !det || max_out < 0) return -1;
    DetSink s = { det, 0, max_out };
    onepass_q15_finish(&h->op, det_emit, &s);
    return s.n;
}

long long fmcw_q15_quantize(const float *adc, long long n,
                            double adc_full_scale, int16_t *out_q)
{
    if (!adc || !out_q || n < 0) return -1;
    long long sat = 0;
    adc_to_q15((const cplx *)adc, (int)n, adc_full_scale,
               (cplx_q15 *)out_q, &sat);
    return sat;
}
