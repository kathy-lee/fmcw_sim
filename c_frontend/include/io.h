#ifndef IO_H
#define IO_H
#include <stdio.h>
#include "fmcw.h"
#include "params.h"
#include "pointcloud.h"

/* The pipeline's working memory, allocated once at startup from the config
 * dimensions — no allocation happens inside the frame loop. */
typedef struct {
    cplx *adc;        /* input cube,  V * C * S      samples */
    cplx *rd;         /* RD cube,     V * C * (S/2)  samples */
    double *power;    /* NCI map,     C * (S/2)      cells   */
    uint8_t *mask;    /* CFAR mask,   C * (S/2)      cells   */
    Detection *det;   /* point cloud, C * (S/2) capacity (worst case: every
                         cell detected, one source each) */
    size_t adc_n, rd_n, cells_n;
} FrontendBuffers;

/* Allocate both cubes for the dimensions in *p. Returns 0 on success, -1 on
 * allocation failure (in which case the struct is zeroed and safe to free). */
int buffers_alloc(FrontendBuffers *b, const FrontendParams *p);

/* Release the cubes; idempotent (pointers are NULLed). */
void buffers_free(FrontendBuffers *b);

/* Startup memory budget: one line per buffer plus the total, computed from
 * the same sizes passed to malloc — the linker-map habit. */
void print_memory_summary(const FrontendBuffers *b, FILE *out);

/* Read exactly n cplx samples from a raw frame file into buf. Returns 0 on
 * success; -1 if the file is missing, shorter, or longer than n samples —
 * a size mismatch means config and data disagree and nothing downstream
 * can be trusted. */
int load_frame(const char *path, cplx *buf, size_t n);

/* Write the CSV header line — the column contract every consumer of the
 * detections file (pytest harness included) parses by name. Returns 0, or
 * -1 on write error. */
int csv_write_header(FILE *f);

/* Append n detections as CSV rows for the given frame index. %.9g keeps
 * full float32-derived precision without noise digits. Returns 0 or -1. */
int csv_write_detections(FILE *f, int frame, const Detection *det, int n);

/* Write nbytes of buf to <dir>/<stem>_%04d.bin — the shared layout of all
 * --dump stage files. Returns 0, or -1 with a message on stderr. */
int dump_stage(const char *dir, const char *stem, int frame,
               const void *buf, size_t nbytes);
#endif
