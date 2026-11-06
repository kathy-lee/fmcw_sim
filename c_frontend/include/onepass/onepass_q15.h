#ifndef ONEPASS_Q15_H
#define ONEPASS_Q15_H
#include <stddef.h>
#include "params.h"
#include "rd_stream_q15.h"
#include "cfar_cells_q15.h"

/* The detection front end as a one-pass composition over rows that arrive
 * one at a time: range-process each row on arrival (the raw frame is never
 * stored), keep all V complex planes resident, then at frame end integrate,
 * detect, and estimate angles — emitting one detection at a time.
 *
 * Storage is caller-provided; the plan owns only the FFT tables and
 * windows (heap, built once at init). Lifetime contract the caller may
 * exploit: power is never read after the CFAR walk, and all snapshots are
 * gathered before spectrum is first written — so snapshots and spectrum
 * may overlay power's storage (snapshots first, disjoint ranges).
 *
 * Detections come back in (doppler_bin, range_bin) order — the same order
 * a mask walk emits — whenever the cell list did not overflow. */

#define ONEPASS_MAX_DET 64

typedef struct {
    int range_bin, doppler_bin, az_idx;          /* azimuth deg = az_idx - 90 */
    double x, y, v_radial, snr_db;
    int64_t power, thresh_q16;
} OnepassDet;

typedef void (*OnepassEmit)(void *ctx, const OnepassDet *d);

typedef struct {
    const FrontendParams *prm;
    RdStreamPlanQ15 rd;
    double rd_scale;
    cplx_q15 *planes;
    int64_t *power;
    CfarCellQ15 *cells;
    cplx_q15 *snapshots;
    double *spectrum;
    cplx_q15 *rowbuf;
    int overflow;
    int n_cells;      /* valid from just after finish's CFAR walk onward */
    long long sat;
} OnepassPlanQ15;

int onepass_q15_init(OnepassPlanQ15 *p, const FrontendParams *prm,
                     cplx_q15 *planes, int64_t *power, CfarCellQ15 *cells,
                     cplx_q15 *snapshots, double *spectrum, cplx_q15 *rowbuf);
void onepass_q15_free(OnepassPlanQ15 *p);

/* Consume the row for (v, c) from p->rowbuf (clobbered). v-major arrival
 * order assumed; the plane's Doppler pass runs inside the c == C-1 call. */
void onepass_q15_row(OnepassPlanQ15 *p, int v, int c);

/* Integrate -> CFAR -> gather -> per-detection TDM + Bartlett + Cartesian.
 * Calls emit once per detection; returns the count. Sets p->overflow. */
int onepass_q15_finish(OnepassPlanQ15 *p, OnepassEmit emit, void *ctx);

/* Add one plane's squared magnitudes into the int64 power map (exact;
 * caller zeroes power before the first plane). */
void nci_q15_accumulate(const cplx_q15 *plane, int n_cells, int64_t *power);

/* The V-element snapshot at flat cell index (doppler_bin * R + range_bin). */
void snapshot_q15_gather(const cplx_q15 *planes, int n_virtual,
                         size_t plane_stride, size_t cell, cplx_q15 *snap);
#endif
