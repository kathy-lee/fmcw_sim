/* fmcw_frontend — MIMO detection front end over exported ADC frames.
 *
 * usage: fmcw_frontend --config params.cfg --frames <dir>
 *                      [--out detections.csv] [--dump <dir>] [--fixed]
 *
 * Per frame: RD per channel -> NCI -> CA-CFAR -> point cloud; --out appends
 * detections as CSV, --dump writes rd_/power_/mask_%04d.bin. Compute stages
 * are bracketed with CLOCK_MONOTONIC timers (file I/O deliberately outside,
 * matching the Python pipeline's compute-only timings) and a mean/max-
 * microseconds table prints at exit. Frame files are read as
 * frame_%04d.bin from index 0 until the first missing index. Exit 0 with
 * "processed N frame(s), M detection(s)"; exit 1 with a message on stderr
 * for any error.
 *
 * --fixed runs the integer chain instead, printing the same table so the
 * two arithmetics are measured the same way, plus the saturation counts.
 *
 * Quantization gets its own row rather than being folded into rd. A real
 * converter hands over integers, so adc_to_q15 exists only because this
 * pipeline's input is a float simulation — its cost is scaffolding, not
 * part of what an integer front end would run. It belongs in the error
 * budget, which the SQNR comparison covers, not in the compute budget. */
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
#include "convert_q15.h"
#include "rd_q15.h"
#include "nci_q15.h"
#include "cfar_q15.h"
#include "pointcloud_q15.h"

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
    int fixed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_csv = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--fixed")) fixed = 1;
        else { fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]); return 1; }
    }
    if (!config || !frames_dir) {
        fprintf(stderr, "usage: fmcw_frontend --config params.cfg --frames <dir>"
                        " [--out detections.csv] [--dump <dir>] [--fixed]\n");
        return 1;
    }

    FrontendParams p;
    if (params_load(config, &p) != 0) return 1;

    FrontendBuffers buf;
    if (buffers_alloc(&buf, &p) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    print_memory_summary(&buf, stdout);

    RdPlan rd_plan;
    DoaPlan doa_plan = {0};
    CfarPlan cfar_plan = {0};
    RdPlanQ15 rdq_plan = {0};
    CfarPlanQ15 cfarq_plan = {0};
    long long sat[2] = {0, 0};
    double rd_scale = q15_rd_scale(p.adc_full_scale, p.n_samples, p.n_chirps);
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
    if (cfar_plan_init(&cfar_plan, &p) != 0) {
        fprintf(stderr, "cfar plan init failed\n");
        goto done;
    }
    if (rd_q15_plan_init(&rdq_plan, p.n_chirps, p.n_samples) != 0 ||
        cfar_q15_plan_init(&cfarq_plan, &p) != 0) {
        fprintf(stderr, "fixed-point plan init failed\n");
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
    enum { T_QUANT, T_RD, T_NCI, T_CFAR, T_PC, T_TOTAL, T_N };
    static const char *stage_name[T_N] =
        {"quant", "rd", "nci", "cfar", "doa_pc", "total"};
    StageTime st[T_N] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
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
        if (fixed)
            adc_to_q15(buf.adc, (int)buf.adc_n, p.adc_full_scale,
                       buf.adc_q, &sat[0]);
        double tq = now_s();
        if (fixed) {
            for (int v = 0; v < p.n_virtual; v++)
                rd_q15_process_channel(&rdq_plan, buf.adc_q + (size_t)v * ch_in,
                                       buf.rd_q + (size_t)v * ch_out, &sat[1]);
        } else {
            for (int v = 0; v < p.n_virtual; v++)
                rd_process_channel(&rd_plan, buf.adc + (size_t)v * ch_in,
                                   buf.rd + (size_t)v * ch_out);
        }
        double t1 = now_s();
        if (fixed)
            nci_q15_integrate(buf.rd_q, p.n_virtual, (int)buf.cells_n,
                              buf.power_q);
        else
            nci_integrate(buf.rd, p.n_virtual, (int)buf.cells_n, buf.power);
        double t2 = now_s();
        if (fixed)
            cfar_q15_ca_mask(&cfarq_plan, buf.power_q, &p, buf.mask);
        else
            cfar_ca_mask(&cfar_plan, buf.power, &p, buf.mask);
        double t3 = now_s();
        int n_det = fixed
            ? pointcloud_q15_build(buf.rd_q, buf.mask, &p, &doa_plan,
                                   rd_scale, buf.det, (int)buf.cells_n)
            : pointcloud_build(buf.rd, buf.mask, &p, &doa_plan,
                               buf.det, (int)buf.cells_n);
        double t4 = now_s();
        stage_add(&st[T_QUANT], tq - t0);
        stage_add(&st[T_RD], t1 - tq);
        stage_add(&st[T_NCI], t2 - t1);
        stage_add(&st[T_CFAR], t3 - t2);
        stage_add(&st[T_PC], t4 - t3);
        stage_add(&st[T_TOTAL], t4 - t0);
        n_det_total += n_det;

        if (csv && csv_write_detections(csv, i, buf.det, n_det) != 0)
            { status = 1; break; }
        if (dump_dir) {
            /* Distinct stems in fixed mode: the files carry different
             * dtypes (int16/int64 rather than complex64/float64), and a
             * directory of same-named files that cannot be read the same
             * way is a trap for whoever opens them next. */
            int bad = fixed
                ? (dump_stage(dump_dir, "rd_q15", i, buf.rd_q,
                              buf.rd_n * sizeof(cplx_q15)) != 0 ||
                   dump_stage(dump_dir, "power_q15", i, buf.power_q,
                              buf.cells_n * sizeof(int64_t)) != 0)
                : (dump_stage(dump_dir, "rd", i, buf.rd,
                              buf.rd_n * sizeof(cplx)) != 0 ||
                   dump_stage(dump_dir, "power", i, buf.power,
                              buf.cells_n * sizeof(double)) != 0);
            if (bad || dump_stage(dump_dir, "mask", i, buf.mask, buf.cells_n) != 0)
                { status = 1; break; }
        }
        n_frames++;
    }

    if (status == 0 && n_frames == 0) {
        fprintf(stderr, "no frames found in %s\n", frames_dir);
        status = 1;
    }
    if (status == 0) {
        printf("timing report (%d frames, %s):\n", n_frames,
               fixed ? "fixed-point Q15" : "float");
        printf("  %-8s %12s %12s\n", "stage", "mean us", "max us");
        for (int t = 0; t < T_N; t++) {
            if (t == T_QUANT && !fixed) continue;   /* float has no such stage */
            printf("  %-8s %12.1f %12.1f\n", stage_name[t],
                   st[t].sum / n_frames * 1e6, st[t].max * 1e6);
        }
        if (fixed) {
            /* Nonzero means an input assumption failed, not that precision
             * was merely lost — worth reading before the timings. */
            printf("saturations: %lld quantization, %lld range-Doppler\n",
                   sat[0], sat[1]);
        }
        printf("processed %d frame(s), %ld detection(s)\n", n_frames, n_det_total);
    }

done:
    if (csv) fclose(csv);
    cfar_q15_plan_free(&cfarq_plan);
    rd_q15_plan_free(&rdq_plan);
    cfar_plan_free(&cfar_plan);
    doa_plan_free(&doa_plan);
    rd_plan_free(&rd_plan);
    buffers_free(&buf);
    return status;
}
