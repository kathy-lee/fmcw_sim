/* fmcw_frontend — Phase 1 pipeline driver.
 *
 * usage: fmcw_frontend --config params.cfg --frames <dir> [--dump <dir>]
 *
 * Loads the config, allocates the ADC/RD buffers once (printing the memory
 * summary), then streams frames <dir>/frame_0000.bin, frame_0001.bin, ...
 * until the first missing index. Each frame runs the range-Doppler stage per
 * virtual channel; with --dump, the resulting V x C x S/2 RD cube is written
 * to <dir>/rd_%04d.bin (cplx float32, C-order) for the pytest golden harness.
 * Exit 0 with "processed N frame(s)" on stdout; exit 1 on any error.
 * Phase 2 adds NCI -> CFAR -> DoA -> point-cloud CSV behind --out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmcw.h"
#include "params.h"
#include "io.h"
#include "rd.h"

/* Write one RD cube (n cplx samples) to <dir>/rd_<index>.bin.
 * Returns 0 on success, -1 (with a message on stderr) on open/write error. */
static int dump_rd(const char *dir, int index, const cplx *rd, size_t n)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/rd_%04d.bin", dir, index);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    size_t wrote = fwrite(rd, sizeof(cplx), n, f);
    fclose(f);
    if (wrote != n) { fprintf(stderr, "short write to %s\n", path); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    const char *config = NULL, *frames_dir = NULL, *dump_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames_dir = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
        else { fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]); return 1; }
    }
    if (!config || !frames_dir) {
        fprintf(stderr,
                "usage: fmcw_frontend --config params.cfg --frames <dir> [--dump <dir>]\n");
        return 1;
    }

    FrontendParams p;
    if (params_load(config, &p) != 0) return 1;

    FrontendBuffers buf;
    if (buffers_alloc(&buf, &p) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    print_memory_summary(&buf, stdout);

    RdPlan rd_plan;
    if (rd_plan_init(&rd_plan, p.n_chirps, p.n_samples) != 0) {
        fprintf(stderr, "rd plan init failed (dims must be powers of two)\n");
        buffers_free(&buf);
        return 1;
    }

    size_t ch_in = (size_t)p.n_chirps * p.n_samples;
    size_t ch_out = (size_t)p.n_chirps * (p.n_samples / 2);
    int n_frames = 0, status = 0;

    for (int i = 0; ; i++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/frame_%04d.bin", frames_dir, i);
        FILE *probe = fopen(path, "rb");
        if (!probe) break;               /* first missing index ends the run */
        fclose(probe);
        if (load_frame(path, buf.adc, buf.adc_n) != 0) { status = 1; break; }

        for (int v = 0; v < p.n_virtual; v++)
            rd_process_channel(&rd_plan, buf.adc + (size_t)v * ch_in,
                               buf.rd + (size_t)v * ch_out);

        if (dump_dir && dump_rd(dump_dir, i, buf.rd, buf.rd_n) != 0)
            { status = 1; break; }
        n_frames++;
    }

    rd_plan_free(&rd_plan);
    buffers_free(&buf);
    if (status == 0 && n_frames == 0) {
        fprintf(stderr, "no frames found in %s\n", frames_dir);
        status = 1;
    }
    if (status == 0) printf("processed %d frame(s)\n", n_frames);
    return status;
}
