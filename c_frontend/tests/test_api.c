#include <stdio.h>
#include <string.h>
#include "fmcw_api.h"
#include "check.h"

/* A minimal valid config (V=6 as 2x3, C=4, S=8 — power-of-two FFT dims). */
static const char *CFG =
    "n_virtual=6\nn_chirps=4\nn_samples=8\nn_tx=2\nn_rx=3\n"
    "doa_n_sources=1\ncfar_guard_d=1\ncfar_guard_r=1\n"
    "cfar_training_d=1\ncfar_training_r=2\nmin_range_bins=1\n"
    "range_per_bin=0.5\npfa=1e-4\ncfar_alpha=8.0\nadc_full_scale=1.0\n"
    "velocity_axis=-1.0,-0.5,0.0,0.5\n"
    "pos_wl=0.0,0.5,1.0,1.5,2.0,2.5\n";

int main(void)
{
    FILE *f = fopen("tests/tmp_api.cfg", "w");
    CHECK(f != NULL);
    CHECK(fputs(CFG, f) >= 0);
    fclose(f);

    /* Failure paths first: bad path -> NULL; every entry point must
     * tolerate a NULL handle (the ctypes caller's last line of defense). */
    CHECK(fmcw_create("tests/tmp_api_missing.cfg") == NULL);
    fmcw_destroy(NULL);
    int dims[3];
    CHECK(fmcw_dims(NULL, dims) == -1);
    CHECK(fmcw_process_frame(NULL, NULL, NULL, 0) == -1);
    CHECK(fmcw_rd(NULL, NULL, NULL) == -1);
    CHECK(fmcw_nci(NULL, NULL, NULL) == -1);
    CHECK(fmcw_cfar(NULL, NULL, NULL) == -1);
    CHECK(fmcw_pointcloud(NULL, NULL, NULL, NULL, 0) == -1);
    CHECK(fmcw_q15_process_frame(NULL, NULL, NULL, 0) == -1);
    CHECK(fmcw_q15_rd(NULL, NULL, NULL) == -1);
    CHECK(fmcw_q15_nci(NULL, NULL, NULL) == -1);
    CHECK(fmcw_q15_cfar(NULL, NULL, NULL) == -1);
    CHECK(fmcw_q15_pointcloud(NULL, NULL, NULL, NULL, 0) == -1);
    CHECK(fmcw_q15_rd_scale(NULL) == 0.0);
    CHECK(fmcw_q15_saturations(NULL, NULL) == -1);
    fmcw_q15_reset_saturations(NULL);

    void *h = fmcw_create("tests/tmp_api.cfg");
    CHECK(h != NULL);
    CHECK(fmcw_dims(h, dims) == 0);
    CHECK(dims[0] == 6 && dims[1] == 4 && dims[2] == 8);

    /* All-zeros frame: power is 0 everywhere, so lin = 1e-12 vs threshold
     * alpha * 1e-12 with alpha > 1 -> no cell fires, 0 detections. Also
     * exercises the whole chain end to end without crashing. */
    enum { V = 6, C = 4, S = 8, R = S / 2, MAXD = C * R };
    float adc[V * C * S * 2];
    memset(adc, 0, sizeof adc);
    double det[MAXD * 7];
    CHECK(fmcw_process_frame(h, adc, det, MAXD) == 0);

    /* Per-stage chain on the same zeros: shapes line up, mask all zero. */
    float rd[V * C * R * 2];
    double power[C * R];
    uint8_t mask[C * R];
    CHECK(fmcw_rd(h, adc, rd) == 0);
    CHECK(fmcw_nci(h, rd, power) == 0);
    for (int i = 0; i < C * R; i++) CHECK(power[i] == 0.0);
    CHECK(fmcw_cfar(h, power, mask) == 0);
    for (int i = 0; i < C * R; i++) CHECK(mask[i] == 0);
    CHECK(fmcw_pointcloud(h, rd, mask, det, MAXD) == 0);

    /* ---- fixed-point chain on the same zeros ---- */

    /* The reconstruction factor: a positive constant for the handle's
     * lifetime. With adc_full_scale = 1.0 it is S*C/16384 = 8*4/16384. */
    double scale = fmcw_q15_rd_scale(h);
    CHECK(scale > 0.0);
    CHECK_CLOSE(scale, 32.0 / 16384.0, 1e-12);

    long long sat[2];
    fmcw_q15_reset_saturations(h);
    CHECK(fmcw_q15_saturations(h, sat) == 0);
    CHECK(sat[0] == 0 && sat[1] == 0);

    /* The facade takes the same float frame and returns the same row
     * layout as its float twin — deliberately drop-in swappable. */
    CHECK(fmcw_q15_process_frame(h, adc, det, MAXD) == 0);

    int16_t rd_q[V * C * R * 2];
    int64_t power_q[C * R];
    CHECK(fmcw_q15_rd(h, adc, rd_q) == 0);
    CHECK(fmcw_q15_nci(h, rd_q, power_q) == 0);
    for (int i = 0; i < C * R; i++) CHECK(power_q[i] == 0);
    CHECK(fmcw_q15_cfar(h, power_q, mask) == 0);
    for (int i = 0; i < C * R; i++) CHECK(mask[i] == 0);
    CHECK(fmcw_q15_pointcloud(h, rd_q, mask, det, MAXD) == 0);

    /* A silent frame must not have clipped anything. */
    CHECK(fmcw_q15_saturations(h, sat) == 0);
    CHECK(sat[0] == 0 && sat[1] == 0);

    fmcw_destroy(h);
    CHECK_DONE();
}
