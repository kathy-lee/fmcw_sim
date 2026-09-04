/* The emit loop replicates pointcloud_q15_build's per-cell body operation
 * for operation — gather-convert, tdm_compensate, power sum, snr_db, DoA,
 * Cartesian; same expressions, same order — so its detections match the
 * frozen chain's to the last bit. */
#include <math.h>
#include <string.h>
#include "convert_q15.h"
#include "doa_cell.h"
#include "pointcloud.h"      /* tdm_compensate — the frozen float step */
#include "onepass_q15.h"

#define ONEPASS_MAX_V 16

int onepass_q15_init(OnepassPlanQ15 *p, const FrontendParams *prm,
                     cplx_q15 *planes, int64_t *power, CfarCellQ15 *cells,
                     cplx_q15 *snapshots, double *spectrum, cplx_q15 *rowbuf)
{
    memset(p, 0, sizeof *p);
    if (prm->n_virtual > ONEPASS_MAX_V) return -1;
    p->prm = prm;
    p->planes = planes; p->power = power; p->cells = cells;
    p->snapshots = snapshots; p->spectrum = spectrum; p->rowbuf = rowbuf;
    p->rd_scale = q15_rd_scale(prm->adc_full_scale, prm->n_samples,
                               prm->n_chirps);
    return rd_stream_q15_plan_init(&p->rd, prm->n_chirps, prm->n_samples);
}

void onepass_q15_free(OnepassPlanQ15 *p)
{
    rd_stream_q15_plan_free(&p->rd);
}

void onepass_q15_row(OnepassPlanQ15 *p, int v, int c)
{
    int C = p->prm->n_chirps, R = p->prm->n_samples / 2;
    cplx_q15 *plane = p->planes + (size_t)v * C * R;
    rd_stream_q15_row(&p->rd, p->rowbuf, c, plane, &p->sat);
    if (c == C - 1)
        rd_stream_q15_plane_finish(&p->rd, plane, p->rowbuf, &p->sat);
}

void nci_q15_accumulate(const cplx_q15 *plane, int n_cells, int64_t *power)
{
    for (int i = 0; i < n_cells; i++) power[i] += cq15_mag2(plane[i]);
}

void snapshot_q15_gather(const cplx_q15 *planes, int n_virtual,
                         size_t plane_stride, size_t cell, cplx_q15 *snap)
{
    for (int v = 0; v < n_virtual; v++)
        snap[v] = planes[(size_t)v * plane_stride + cell];
}

int onepass_q15_finish(OnepassPlanQ15 *p, OnepassEmit emit, void *ctx)
{
    const FrontendParams *prm = p->prm;
    int V = prm->n_virtual, C = prm->n_chirps, R = prm->n_samples / 2;
    size_t ch = (size_t)C * R;

    memset(p->power, 0, ch * sizeof(int64_t));
    for (int v = 0; v < V; v++)
        nci_q15_accumulate(p->planes + (size_t)v * ch, (int)ch, p->power);

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

    for (int i = 0; i < n; i++) {
        int d = p->cells[i].doppler_bin, r = p->cells[i].range_bin;
        cplx raw[ONEPASS_MAX_V], snap[ONEPASS_MAX_V];
        for (int v = 0; v < V; v++) {
            cplx_q15 q = p->snapshots[(size_t)i * V + v];
            raw[v] = (cplx){ (float)((double)q.re * p->rd_scale),
                             (float)((double)q.im * p->rd_scale) };
        }
        tdm_compensate(raw, snap, V, d, C, prm->n_tx, prm->n_rx);
        double psum = 0.0;
        for (int v = 0; v < V; v++) psum += cmag2(snap[v]);
        double snr_db = 10.0 * log10(psum + 1e-12);

        int az_idx = doa_bartlett_cell(snap, prm->pos_wl, V, p->spectrum);
        double az = (double)(az_idx - 90);
        double rng = r * prm->range_per_bin;
        double azr = az * FMCW_PI / 180.0;

        OnepassDet det = {
            r, d, az_idx,
            rng * cos(azr), rng * sin(azr),
            prm->velocity_axis[d], snr_db,
            p->cells[i].power, p->cells[i].thresh_q16
        };
        emit(ctx, &det);
    }
    return n;
}
