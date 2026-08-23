/* Point-cloud assembly from a Q15 cube. Contract in pointcloud_q15.h.
 *
 * Structurally this is pointcloud.c with one line changed — the gather —
 * which is the point: keeping the loop separate rather than converting the
 * cube up front means the conversion cost scales with detections, not with
 * cube size. */
#include <math.h>
#include "pointcloud_q15.h"

int pointcloud_q15_build(const cplx_q15 *rd, const uint8_t *mask,
                         const FrontendParams *p, DoaPlan *doa,
                         double rd_scale, Detection *out, int max_out)
{
    int V = p->n_virtual, C = p->n_chirps, R = p->n_samples / 2;
    size_t ch = (size_t)C * R;
    cplx raw[FMCW_MAX_LIST], snap[FMCW_MAX_LIST];
    double az[181];
    int n = 0;

    for (int d = 0; d < C; d++) {
        for (int r = 0; r < R; r++) {
            if (!mask[(size_t)d * R + r]) continue;
            /* The integer-to-float handoff, V values wide, once per
             * detected cell. */
            for (int v = 0; v < V; v++) {
                cplx_q15 q = rd[(size_t)v * ch + (size_t)d * R + r];
                raw[v] = (cplx){(float)((double)q.re * rd_scale),
                                (float)((double)q.im * rd_scale)};
            }
            tdm_compensate(raw, snap, V, d, C, p->n_tx, p->n_rx);
            double psum = 0.0;
            for (int v = 0; v < V; v++) psum += cmag2(snap[v]);
            double snr_db = 10.0 * log10(psum + 1e-12);

            int k = doa_bartlett(doa, snap, p->doa_n_sources, az);
            for (int a = 0; a < k && n < max_out; a++) {
                double rng = r * p->range_per_bin;
                double azr = az[a] * FMCW_PI / 180.0;
                out[n++] = (Detection){
                    r, d, az[a], rng * cos(azr), rng * sin(azr),
                    p->velocity_axis[d], snr_db
                };
            }
        }
    }
    return n;
}
