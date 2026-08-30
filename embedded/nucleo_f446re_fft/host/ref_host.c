/* Host reference for the Nucleo harness: the same kernel, the same input, the
 * same output format. Comparing the board against this is then a diff of
 * two identically shaped streams rather than a bespoke check.
 *
 * The twiddle table is dumped, not just the transform output, because it is
 * the one place a verbatim build can legitimately diverge: the table is
 * built in floating point, and a compiler that narrows double (avr-gcc on
 * the Uno did) can round a twiddle near a boundary one count differently
 * and then propagate that through every butterfly. arm-none-eabi-gcc keeps
 * double at 64 bits, so no divergence is expected here, but the dump stays
 * the first place to look if a mismatch ever appears.
 */
/* clock_gettime is POSIX rather than ISO C, and -std=c11 hides it. */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vectors.h"

#if ARITH_Q15
#include "fft_q15.h"
typedef FftPlanQ15 plan_t;
typedef cplx_q15 work_t;
#define ARITH_NAME "q15"
#define PLAN_INIT(p, n) fft_q15_plan_init((p), (n))
#define PLAN_FREE(p)    fft_q15_plan_free(p)
#define TRANSFORM(p, x) fft_q15_inplace((p), (x), NULL)
#else
#include "fft.h"
typedef FftPlan plan_t;
typedef cplx work_t;
#define ARITH_NAME "float"
#define PLAN_INIT(p, n) fft_plan_init((p), (n))
#define PLAN_FREE(p)    fft_plan_free(p)
#define TRANSFORM(p, x) fft_inplace((p), (x))
#endif

#define ITERS 100000

static work_t work[VEC_N];

static void load(const vec_sample *v)
{
    for (int i = 0; i < VEC_N; i++) {
        work[i].re = VEC_RE(v, i);
        work[i].im = VEC_IM(v, i);
    }
}

static void print_value(double v)
{
#if ARITH_Q15
    printf("%d", (int)v);
#else
    printf("%.6e", v);
#endif
}

static void dump_twiddles(const plan_t *p)
{
    for (int k = 0; k < VEC_N / 2; k++) {
        printf("TW %d ", k);
        print_value((double)p->twiddle[k].re);
        printf(" ");
        print_value((double)p->twiddle[k].im);
        printf("\n");
    }
}

static void run_and_dump(const plan_t *p, const vec_sample *v, const char *name)
{
    load(v);
    TRANSFORM(p, work);
    printf("VEC %s\n", name);
    for (int i = 0; i < VEC_N; i++) {
        printf("OUT %d ", i);
        print_value((double)work[i].re);
        printf(" ");
        print_value((double)work[i].im);
        printf("\n");
    }
}

static double elapsed_us(struct timespec a, struct timespec b)
{
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) / 1e3;
}

/* Two loops with the same iteration count, because the transform is
 * destructive: re-running it on its own output is a different computation
 * each pass, so every iteration must reload the input, and that reload has
 * to come back out of the number. Both raw figures are printed so the
 * subtraction can be judged rather than taken on trust. */
static void timing(const plan_t *p)
{
    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long k = 0; k < ITERS; k++) load(vec_radar);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    for (long k = 0; k < ITERS; k++) { load(vec_radar); TRANSFORM(p, work); }
    clock_gettime(CLOCK_MONOTONIC, &t3);

    double copy_us = elapsed_us(t0, t1) / ITERS;
    double total_us = elapsed_us(t2, t3) / ITERS;
    printf("TIME iters %d copy_us %.4f total_us %.4f fft_us %.4f\n",
           ITERS, copy_us, total_us, total_us - copy_us);
}

int main(void)
{
    plan_t p = {0};
    if (PLAN_INIT(&p, VEC_N) != 0) {
        fprintf(stderr, "plan init failed\n");
        return 1;
    }
    printf("# nucleo-fft 1\n");
    printf("# arith %s\n", ARITH_NAME);
    printf("# n %d\n", VEC_N);
    dump_twiddles(&p);
    run_and_dump(&p, vec_dc, "dc");
    run_and_dump(&p, vec_radar, "radar");
    timing(&p);
    /* The host has no meaningful counterpart to the board's RAM budget;
     * the fields are emitted as zero so both streams parse identically. */
    printf("MEM heap 0 stack_peak 0 free_start 0\n");
    printf("END\n");
    PLAN_FREE(&p);
    return 0;
}
