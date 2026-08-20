/* File I/O: params.cfg parsing (this half) and frame/buffer handling.
 * Everything here is host-tool plumbing — the strict
 * "reject anything unexpected" posture exists so config mistakes fail at
 * startup, not as silent garbage in the radar math. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "params.h"

/* Parse a comma-separated double list ("1.0,2.5,...") into out[0..max).
 * Stores the element count in *count. Returns 0 on success, -1 on empty
 * list, malformed number, or overflow past max. */
static int parse_list(const char *val, double *out, int max, int *count)
{
    int n = 0;
    const char *s = val;
    while (*s != '\0' && *s != '\n') {
        char *end;
        double v = strtod(s, &end);
        if (end == s || n >= max) return -1;   /* no number parsed, or full */
        out[n++] = v;
        s = end;
        if (*s == ',') s++;
        else break;
    }
    *count = n;
    return n > 0 ? 0 : -1;
}

/* Parse one integer, rejecting trailing junk ("12abc" fails, "12" passes).
 * strtol's `end` pointer tells us where parsing stopped — a clean parse
 * stops at the end of the value (or the newline fgets kept). */
static int parse_int(const char *val, int *out)
{
    char *end;
    long v = strtol(val, &end, 10);
    if (end == val || (*end != '\0' && *end != '\n')) return -1;
    *out = (int)v;
    return 0;
}

/* Same contract as parse_int, for doubles. */
static int parse_double(const char *val, double *out)
{
    char *end;
    double v = strtod(val, &end);
    if (end == val || (*end != '\0' && *end != '\n')) return -1;
    *out = v;
    return 0;
}

/* Contract in params.h. Mechanism: read line by line, split each at '=' and
 * dispatch on the key through a strcmp chain (16 keys — a table would be
 * over-engineering); tick off each key in seen[]. After EOF, reject any
 * missing key, then cross-check the dimensions against each other. All error
 * paths funnel through `goto fail` so fclose happens exactly once. */
int params_load(const char *path, FrontendParams *p)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "params: cannot open %s\n", path); return -1; }
    memset(p, 0, sizeof *p);

    /* One flag per required key, in declaration order. */
    enum { N_KEYS = 16 };
    int seen[N_KEYS] = {0};
    char line[8192];
    int ok = 0, lineno = 0;

    while (fgets(line, sizeof line, f)) {
        lineno++;
        if (line[0] == '\n' || line[0] == '#') continue;   /* blank / comment */
        char *eq = strchr(line, '=');
        if (!eq) {
            fprintf(stderr, "params:%d: expected key=value\n", lineno);
            goto fail;
        }
        *eq = '\0';                      /* split the line in place: */
        const char *key = line, *val = eq + 1;
        int r;
        if      (!strcmp(key, "n_virtual"))       { r = parse_int(val, &p->n_virtual);       seen[0] = 1; }
        else if (!strcmp(key, "n_chirps"))        { r = parse_int(val, &p->n_chirps);        seen[1] = 1; }
        else if (!strcmp(key, "n_samples"))       { r = parse_int(val, &p->n_samples);       seen[2] = 1; }
        else if (!strcmp(key, "n_tx"))            { r = parse_int(val, &p->n_tx);            seen[3] = 1; }
        else if (!strcmp(key, "n_rx"))            { r = parse_int(val, &p->n_rx);            seen[4] = 1; }
        else if (!strcmp(key, "doa_n_sources"))   { r = parse_int(val, &p->doa_n_sources);   seen[5] = 1; }
        else if (!strcmp(key, "cfar_guard_d"))    { r = parse_int(val, &p->cfar_guard_d);    seen[6] = 1; }
        else if (!strcmp(key, "cfar_guard_r"))    { r = parse_int(val, &p->cfar_guard_r);    seen[7] = 1; }
        else if (!strcmp(key, "cfar_training_d")) { r = parse_int(val, &p->cfar_training_d); seen[8] = 1; }
        else if (!strcmp(key, "cfar_training_r")) { r = parse_int(val, &p->cfar_training_r); seen[9] = 1; }
        else if (!strcmp(key, "min_range_bins"))  { r = parse_int(val, &p->min_range_bins);  seen[10] = 1; }
        else if (!strcmp(key, "range_per_bin"))   { r = parse_double(val, &p->range_per_bin); seen[11] = 1; }
        else if (!strcmp(key, "pfa"))             { r = parse_double(val, &p->pfa);          seen[12] = 1; }
        else if (!strcmp(key, "cfar_alpha"))      { r = parse_double(val, &p->cfar_alpha);   seen[13] = 1; }
        else if (!strcmp(key, "velocity_axis"))   { r = parse_list(val, p->velocity_axis, FMCW_MAX_LIST, &p->n_velocity); seen[14] = 1; }
        else if (!strcmp(key, "pos_wl"))          { r = parse_list(val, p->pos_wl, FMCW_MAX_LIST, &p->n_pos); seen[15] = 1; }
        else { fprintf(stderr, "params:%d: unknown key '%s'\n", lineno, key); goto fail; }
        if (r != 0) { fprintf(stderr, "params:%d: bad value for '%s'\n", lineno, key); goto fail; }
    }

    for (int i = 0; i < N_KEYS; i++)
        if (!seen[i]) { fprintf(stderr, "params: missing required key #%d\n", i); goto fail; }

    if (p->n_virtual <= 0 || p->n_chirps <= 0 || p->n_samples <= 0 ||
        p->n_tx <= 0 || p->n_rx <= 0 ||
        p->n_virtual != p->n_tx * p->n_rx ||
        p->n_velocity != p->n_chirps || p->n_pos != p->n_virtual) {
        fprintf(stderr, "params: inconsistent dimensions\n");
        goto fail;
    }
    ok = 1;
fail:
    fclose(f);
    return ok ? 0 : -1;
}

/* ---- frame / buffer half (contracts in io.h) ---- */

int buffers_alloc(FrontendBuffers *b, const FrontendParams *p)
{
    size_t V = (size_t)p->n_virtual, C = (size_t)p->n_chirps, S = (size_t)p->n_samples;
    b->adc_n = V * C * S;
    b->rd_n = V * C * (S / 2);
    b->adc = malloc(b->adc_n * sizeof(cplx));
    b->rd = malloc(b->rd_n * sizeof(cplx));
    if (!b->adc || !b->rd) { buffers_free(b); return -1; }
    return 0;
}

void buffers_free(FrontendBuffers *b)
{
    free(b->adc);
    free(b->rd);
    b->adc = NULL;
    b->rd = NULL;
    b->adc_n = b->rd_n = 0;
}

void print_memory_summary(const FrontendBuffers *b, FILE *out)
{
    size_t adc_bytes = b->adc_n * sizeof(cplx);
    size_t rd_bytes = b->rd_n * sizeof(cplx);
    fprintf(out, "memory summary:\n");
    fprintf(out, "  adc_cube  %10zu bytes\n", adc_bytes);
    fprintf(out, "  rd_cube   %10zu bytes\n", rd_bytes);
    fprintf(out, "  total     %10zu bytes\n", adc_bytes + rd_bytes);
}

/* fread returns the number of complete items read; the extra fgetc probes
 * whether the file has bytes beyond the n we asked for (EOF means it ended
 * exactly where the config said it should). */
int load_frame(const char *path, cplx *buf, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(buf, sizeof(cplx), n, f);
    int extra = (fgetc(f) != EOF);
    fclose(f);
    if (got != n || extra) {
        fprintf(stderr, "frame %s: expected %zu samples\n", path, n);
        return -1;
    }
    return 0;
}
