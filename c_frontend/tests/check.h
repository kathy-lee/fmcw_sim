#ifndef CHECK_H
#define CHECK_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int checks_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        checks_failed = 1; \
    } \
} while (0)

#define CHECK_CLOSE(a, b, tol) CHECK(fabs((double)(a) - (double)(b)) <= (tol))

#define CHECK_DONE() do { \
    if (checks_failed) { fprintf(stderr, "%s: FAILED\n", __FILE__); exit(1); } \
    printf("%s: OK\n", __FILE__); \
    return 0; \
} while (0)

/* Read exactly nbytes from path into buf. Returns 0 on success, -1 on error. */
static inline int load_bin(const char *path, void *buf, size_t nbytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    size_t got = fread(buf, 1, nbytes, f);
    int extra = (fgetc(f) != EOF);
    fclose(f);
    if (got != nbytes || extra) {
        fprintf(stderr, "%s: expected %zu bytes\n", path, nbytes);
        return -1;
    }
    return 0;
}
#endif
