/* FFI facade. The handle bundles everything a frame needs; the stage
 * functions cast the flat float pointers back to cplx (identical layout:
 * two floats) and call the kernels. Detections cross the boundary
 * flattened to 7 doubles per row so no struct layout is shared with the
 * caller. calloc + NULL-safe frees make every create failure path safe. */
#include <stdlib.h>
#include "fmcw_api.h"
#include "params.h"
#include "io.h"
#include "rd.h"
#include "nci.h"
#include "cfar.h"
#include "doa.h"
#include "pointcloud.h"

typedef struct {
    FrontendParams p;
    FrontendBuffers buf;
    RdPlan rd;
    CfarPlan cfar;
    DoaPlan doa;
} FmcwHandle;

void *fmcw_create(const char *params_cfg)
{
    if (!params_cfg) return NULL;
    FmcwHandle *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    if (params_load(params_cfg, &h->p) != 0) { free(h); return NULL; }
    if (buffers_alloc(&h->buf, &h->p) != 0) { free(h); return NULL; }
    if (rd_plan_init(&h->rd, h->p.n_chirps, h->p.n_samples) != 0 ||
        cfar_plan_init(&h->cfar, &h->p) != 0 ||
        doa_plan_init(&h->doa, h->p.pos_wl, h->p.n_virtual) != 0) {
        fmcw_destroy(h);   /* zeroed and inited plans both free safely */
        return NULL;
    }
    return h;
}

void fmcw_destroy(void *vh)
{
    FmcwHandle *h = vh;
    if (!h) return;
    doa_plan_free(&h->doa);
    cfar_plan_free(&h->cfar);
    rd_plan_free(&h->rd);
    buffers_free(&h->buf);
    free(h);
}

int fmcw_dims(void *vh, int out[3])
{
    FmcwHandle *h = vh;
    if (!h || !out) return -1;
    out[0] = h->p.n_virtual;
    out[1] = h->p.n_chirps;
    out[2] = h->p.n_samples;
    return 0;
}

int fmcw_rd(void *vh, const float *adc, float *rd)
{
    FmcwHandle *h = vh;
    if (!h || !adc || !rd) return -1;
    const cplx *in = (const cplx *)adc;
    cplx *out = (cplx *)rd;
    size_t ch_in = (size_t)h->p.n_chirps * h->p.n_samples;
    size_t ch_out = (size_t)h->p.n_chirps * (h->p.n_samples / 2);
    for (int v = 0; v < h->p.n_virtual; v++)
        rd_process_channel(&h->rd, in + (size_t)v * ch_in,
                           out + (size_t)v * ch_out);
    return 0;
}

int fmcw_nci(void *vh, const float *rd, double *power)
{
    FmcwHandle *h = vh;
    if (!h || !rd || !power) return -1;
    nci_integrate((const cplx *)rd, h->p.n_virtual,
                  (int)h->buf.cells_n, power);
    return 0;
}

int fmcw_cfar(void *vh, const double *power, uint8_t *mask)
{
    FmcwHandle *h = vh;
    if (!h || !power || !mask) return -1;
    cfar_ca_mask(&h->cfar, power, &h->p, mask);
    return 0;
}

int fmcw_pointcloud(void *vh, const float *rd, const uint8_t *mask,
                    double *det, int max_out)
{
    FmcwHandle *h = vh;
    if (!h || !rd || !mask || !det || max_out < 0) return -1;
    int cap = max_out < (int)h->buf.cells_n ? max_out : (int)h->buf.cells_n;
    int n = pointcloud_build((const cplx *)rd, mask, &h->p, &h->doa,
                             h->buf.det, cap);
    for (int i = 0; i < n; i++) {
        const Detection *d = &h->buf.det[i];
        double *row = det + (size_t)7 * i;
        row[0] = d->range_bin;
        row[1] = d->doppler_bin;
        row[2] = d->azimuth_deg;
        row[3] = d->x;
        row[4] = d->y;
        row[5] = d->v_radial;
        row[6] = d->snr_db;
    }
    return n;
}

int fmcw_process_frame(void *vh, const float *adc, double *det, int max_out)
{
    FmcwHandle *h = vh;
    if (!h || !adc || !det) return -1;
    if (fmcw_rd(vh, adc, (float *)h->buf.rd) != 0) return -1;
    if (fmcw_nci(vh, (const float *)h->buf.rd, h->buf.power) != 0) return -1;
    if (fmcw_cfar(vh, h->buf.power, h->buf.mask) != 0) return -1;
    return fmcw_pointcloud(vh, (const float *)h->buf.rd, h->buf.mask,
                           det, max_out);
}
