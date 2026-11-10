/* Each stage is the matching part of onepass_q15_finish with its
 * expressions and their order unchanged, so the staged path and the finish
 * agree to the last bit. */
#include <math.h>
#include <string.h>
#include "convert_q15.h"
#include "doa_cell.h"
#include "pointcloud.h"      /* tdm_compensate — the frozen float step */
#include "onepass_stages_q15.h"

#define ONEPASS_MAX_V 16

void onepass_q15_power(OnepassPlanQ15 *p)
{
    const FrontendParams *prm = p->prm;
    int V = prm->n_virtual, C = prm->n_chirps, R = prm->n_samples / 2;
    size_t ch = (size_t)C * R;

    memset(p->power, 0, ch * sizeof(int64_t));
    for (int v = 0; v < V; v++)
        nci_q15_accumulate(p->planes + (size_t)v * ch, (int)ch, p->power);
}

int onepass_q15_cells(OnepassPlanQ15 *p)
{
    const FrontendParams *prm = p->prm;
    int V = prm->n_virtual, C = prm->n_chirps, R = prm->n_samples / 2;
    size_t ch = (size_t)C * R;

    int n = cfar_cells_q15(p->power, prm, p->cells, ONEPASS_MAX_DET,
                           &p->overflow);
    p->n_cells = n;
    /* power is dead from here on: gather every snapshot before spectrum
     * is written, so both may live in its storage */
    for (int i = 0; i < n; i++) {
        size_t cell = (size_t)p->cells[i].doppler_bin * R
                      + p->cells[i].range_bin;
        snapshot_q15_gather(p->planes, V, ch, cell,
                            p->snapshots + (size_t)i * V);
    }
    return n;
}

void onepass_q15_det(OnepassPlanQ15 *p, int i, OnepassDoaFn doa,
                     void *doa_ctx, OnepassDet *out)
{
    const FrontendParams *prm = p->prm;
    int V = prm->n_virtual, C = prm->n_chirps;
    int d = p->cells[i].doppler_bin, r = p->cells[i].range_bin;
    /* zeroed: the compiler cannot see that V >= 1, and warns that
     * tdm_compensate may read raw unset */
    cplx raw[ONEPASS_MAX_V] = {0}, snap[ONEPASS_MAX_V];
    for (int v = 0; v < V; v++) {
        cplx_q15 q = p->snapshots[(size_t)i * V + v];
        raw[v] = (cplx){ (float)((double)q.re * p->rd_scale),
                         (float)((double)q.im * p->rd_scale) };
    }
    tdm_compensate(raw, snap, V, d, C, prm->n_tx, prm->n_rx);
    double psum = 0.0;
    for (int v = 0; v < V; v++) psum += cmag2(snap[v]);
    double snr_db = 10.0 * log10(psum + 1e-12);

    int az_idx = doa(doa_ctx, snap, p);
    double az = (double)(az_idx - 90);
    double rng = r * prm->range_per_bin;
    double azr = az * FMCW_PI / 180.0;

    *out = (OnepassDet){
        r, d, az_idx,
        rng * cos(azr), rng * sin(azr),
        prm->velocity_axis[d], snr_db,
        p->cells[i].power, p->cells[i].thresh_q16
    };
}

int onepass_doa_f64(void *ctx, const cplx *snap, const OnepassPlanQ15 *p)
{
    (void)ctx;
    return doa_bartlett_cell(snap, p->prm->pos_wl, p->prm->n_virtual,
                             p->spectrum);
}
