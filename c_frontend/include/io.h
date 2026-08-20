#ifndef IO_H
#define IO_H
#include <stdio.h>
#include "fmcw.h"
#include "params.h"

/* The pipeline's working memory, allocated once at startup from the config
 * dimensions (spec: no allocation inside the frame loop). */
typedef struct {
    cplx *adc;      /* input cube,  V * C * S      samples */
    cplx *rd;       /* RD cube,     V * C * (S/2)  samples */
    size_t adc_n, rd_n;
} FrontendBuffers;

/* Allocate both cubes for the dimensions in *p. Returns 0 on success, -1 on
 * allocation failure (in which case the struct is zeroed and safe to free). */
int buffers_alloc(FrontendBuffers *b, const FrontendParams *p);

/* Release the cubes; idempotent (pointers are NULLed). */
void buffers_free(FrontendBuffers *b);

/* Startup memory budget: one line per buffer plus the total, computed from
 * the same sizes passed to malloc — the linker-map habit (spec section 5). */
void print_memory_summary(const FrontendBuffers *b, FILE *out);

/* Read exactly n cplx samples from a raw frame file into buf. Returns 0 on
 * success; -1 if the file is missing, shorter, or longer than n samples —
 * a size mismatch means config and data disagree and nothing downstream
 * can be trusted. */
int load_frame(const char *path, cplx *buf, size_t n);
#endif
