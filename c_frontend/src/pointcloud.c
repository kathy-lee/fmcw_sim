/* Point-cloud assembly. The snapshot is gathered with stride C*R (channel-
 * major rd cube), compensated, and handed to the DoA plan; all trig runs in
 * double, results stored float32 (cplx) — matching the type policy. Angles
 * come back as exact grid degrees, so cos/sin of them are deterministic. */
#include <math.h>
#include "pointcloud.h"

void tdm_compensate(const cplx *in, cplx *out, int n_virtual, int doppler_bin,
                    int n_chirps, int n_tx, int n_rx)
{
    for (int v = 0; v < n_virtual; v++) {
        int i_tx = v / n_rx;
        double ph = -2.0 * FMCW_PI * (doppler_bin - n_chirps / 2.0) * i_tx
                    / ((double)n_chirps * n_tx);
        double c = cos(ph), s = sin(ph);
        out[v] = (cplx){(float)(in[v].re * c - in[v].im * s),
                        (float)(in[v].re * s + in[v].im * c)};
    }
}

int pointcloud_build(const cplx *rd, const uint8_t *mask,
                     const FrontendParams *p, DoaPlan *doa,
                     Detection *out, int max_out)
{
    int V = p->n_virtual, C = p->n_chirps, R = p->n_samples / 2;
    size_t ch = (size_t)C * R;
    cplx raw[FMCW_MAX_LIST], snap[FMCW_MAX_LIST];
    double az[181];
    int n = 0;

    for (int d = 0; d < C; d++) {
        for (int r = 0; r < R; r++) {
            if (!mask[(size_t)d * R + r]) continue;
            for (int v = 0; v < V; v++)
                raw[v] = rd[(size_t)v * ch + (size_t)d * R + r];
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
