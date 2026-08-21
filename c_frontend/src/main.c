/* fmcw_frontend — MIMO detection front end over exported ADC frames.
 *
 * usage: fmcw_frontend --config params.cfg --frames <dir>
 *                      [--out detections.csv] [--dump <dir>]
 *
 * Per frame: RD per channel -> NCI -> CA-CFAR -> point cloud; --out appends
 * detections as CSV, --dump writes rd_/power_/mask_%04d.bin. Compute stages
 * are bracketed with CLOCK_MONOTONIC timers (file I/O deliberately outside,
 * matching the Python pipeline's compute-only timings) and a mean/max-
 * microseconds table prints at exit. Frame files are read as
 * frame_%04d.bin from index 0 until the first missing index. Exit 0 with
 * "processed N frame(s), M detection(s)"; exit 1 with a message on stderr
 * for any error. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fmcw.h"
#include "params.h"
#include "io.h"
#include "rd.h"
#include "nci.h"
#include "cfar.h"
#include "doa.h"
#include "pointcloud.h"

/* Monotonic stamp in seconds: immune to NTP/wall-clock jumps, which is why
 * this file asks for POSIX (ISO C11's timespec_get is wall time). */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { double sum, max; } StageTime;

static void stage_add(StageTime *t, double dt)
{
    t->sum += dt;
    if (dt > t->max) t->max = dt;
}

int main(int argc, char **argv)
{
    const char *config = NULL, *frames_dir = NULL, *out_csv = NULL, *dump_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_csv = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
        else { fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]); return 1; }
    }
    if (!config || !frames_dir) {
        fprintf(stderr, "usage: fmcw_frontend --config params.cfg --frames <dir>"
                        " [--out detections.csv] [--dump <dir>]\n");
        return 1;
    }

    FrontendParams p;
    if (params_load(config, &p) != 0) return 1;

    FrontendBuffers buf;
    if (buffers_alloc(&buf, &p) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    print_memory_summary(&buf, stdout);

    RdPlan rd_plan;
    DoaPlan doa_plan;
    FILE *csv = NULL;
    int status = 1;
    if (rd_plan_init(&rd_plan, p.n_chirps, p.n_samples) != 0) {
        fprintf(stderr, "rd plan init failed (dims must be powers of two)\n");
        buffers_free(&buf);
        return 1;
    }
    if (doa_plan_init(&doa_plan, p.pos_wl, p.n_virtual) != 0) {
        fprintf(stderr, "doa plan init failed\n");
        goto done;
    }
    if (out_csv) {
        csv = fopen(out_csv, "w");
        if (!csv || csv_write_header(csv) != 0) {
            fprintf(stderr, "cannot write %s\n", out_csv);
            goto done;
        }
    }

    size_t ch_in = (size_t)p.n_chirps * p.n_samples;
    size_t ch_out = (size_t)p.n_chirps * (p.n_samples / 2);
    enum { T_RD, T_NCI, T_CFAR, T_PC, T_TOTAL, T_N };
    static const char *stage_name[T_N] = {"rd", "nci", "cfar", "doa_pc", "total"};
    StageTime st[T_N] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    int n_frames = 0;
    long n_det_total = 0;
    status = 0;

    for (int i = 0; ; i++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/frame_%04d.bin", frames_dir, i);
        FILE *probe = fopen(path, "rb");
        if (!probe) break;               /* first missing index ends the run */
        fclose(probe);
        if (load_frame(path, buf.adc, buf.adc_n) != 0) { status = 1; break; }

        double t0 = now_s();
        for (int v = 0; v < p.n_virtual; v++)
            rd_process_channel(&rd_plan, buf.adc + (size_t)v * ch_in,
                               buf.rd + (size_t)v * ch_out);
        double t1 = now_s();
        nci_integrate(buf.rd, p.n_virtual, (int)buf.cells_n, buf.power);
        double t2 = now_s();
        cfar_ca_mask(buf.power, &p, buf.mask);
        double t3 = now_s();
        int n_det = pointcloud_build(buf.rd, buf.mask, &p, &doa_plan,
                                     buf.det, (int)buf.cells_n);
        double t4 = now_s();
        stage_add(&st[T_RD], t1 - t0);
        stage_add(&st[T_NCI], t2 - t1);
        stage_add(&st[T_CFAR], t3 - t2);
        stage_add(&st[T_PC], t4 - t3);
        stage_add(&st[T_TOTAL], t4 - t0);
        n_det_total += n_det;

        if (csv && csv_write_detections(csv, i, buf.det, n_det) != 0)
            { status = 1; break; }
        if (dump_dir) {
            if (dump_stage(dump_dir, "rd", i, buf.rd, buf.rd_n * sizeof(cplx)) != 0 ||
                dump_stage(dump_dir, "power", i, buf.power,
                           buf.cells_n * sizeof(double)) != 0 ||
                dump_stage(dump_dir, "mask", i, buf.mask, buf.cells_n) != 0)
                { status = 1; break; }
        }
        n_frames++;
    }

    if (status == 0 && n_frames == 0) {
        fprintf(stderr, "no frames found in %s\n", frames_dir);
        status = 1;
    }
    if (status == 0) {
        printf("timing report (%d frames):\n", n_frames);
        printf("  %-8s %12s %12s\n", "stage", "mean us", "max us");
        for (int t = 0; t < T_N; t++)
            printf("  %-8s %12.1f %12.1f\n", stage_name[t],
                   st[t].sum / n_frames * 1e6, st[t].max * 1e6);
        printf("processed %d frame(s), %ld detection(s)\n", n_frames, n_det_total);
    }

done:
    if (csv) fclose(csv);
    doa_plan_free(&doa_plan);
    rd_plan_free(&rd_plan);
    buffers_free(&buf);
    return status;
}
