#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#include <stdint.h>
#include "params.h"
#include "doa.h"

/* One detected point: RD-cell provenance (bins, azimuth) plus the derived
 * Cartesian row — together, the detections-CSV columns minus the frame
 * index. */
typedef struct {
    int range_bin, doppler_bin;
    double azimuth_deg, x, y, v_radial, snr_db;
} Detection;

/* Remove the TDM inter-TX Doppler phase from one virtual-array snapshot:
 * out[v] = in[v] * exp(j*phase), phase = -2*pi*(doppler_bin - n_chirps/2)
 * * (v / n_rx) / (n_chirps * n_tx). Port of tdm_doppler_compensation();
 * doppler_bin is the fftshifted bin index. in and out may not alias. */
void tdm_compensate(const cplx *in, cplx *out, int n_virtual, int doppler_bin,
                    int n_chirps, int n_tx, int n_rx);

/* The detect-then-angle loop of mimo_point_cloud(): for each mask==1 cell
 * (Doppler-major, np.argwhere order), TDM-compensate the V-element
 * snapshot from rd (V contiguous n_chirps x n_samples/2 maps),
 * snr_db = 10*log10(sum |snap|^2 + 1e-12), azimuth(s) via doa_bartlett
 * with p->doa_n_sources, then x = r*range_per_bin*cos(az),
 * y = r*range_per_bin*sin(az), v_radial = velocity_axis[d]. Writes at most
 * max_out detections; returns the count. */
int pointcloud_build(const cplx *rd, const uint8_t *mask,
                     const FrontendParams *p, DoaPlan *doa,
                     Detection *out, int max_out);
#endif
