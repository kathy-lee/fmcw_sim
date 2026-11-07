/* The project's FFT beside its CMSIS-DSP counterpart on an STM32F446RE: the
 * same input through both, at 128 and 32 points, reporting cycles, memory and
 * how far the outputs differ.
 *
 * This file holds no FFT arithmetic. The project's kernel is copied in from
 * c_frontend and CMSIS-DSP is compiled unmodified, so a number measured here
 * is a number about that code.
 *
 * The configuration block comes first and is checked by the reader: a wrong
 * clock or a soft-float build produces plausible numbers that measure the
 * wrong thing.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "main.h"
#include "harness.h"

#ifndef BRINGUP
#define BRINGUP 0
#endif

#if BRINGUP
#define ARITH_NAME "bringup"
#else

#include "arm_const_structs.h"
#include "vectors.h"

#if ARITH_Q15
#include "fft_q15.h"
typedef FftPlanQ15 plan_t;
typedef cplx_q15 work_t;
typedef q15_t elem_t;
typedef arm_cfft_instance_q15 cmsis_t;
#define ARITH_NAME "q15"
#define PLAN_INIT(p, n) fft_q15_plan_init((p), (n))
#define OWN_FFT(p, x) fft_q15_inplace((p), (x), NULL)
#define CMSIS_FFT(s, x) arm_cfft_q15((s), (x), 0, 1)
#define CMSIS_128 (&arm_cfft_sR_q15_len128)
#define CMSIS_32 (&arm_cfft_sR_q15_len32)
#else
#include "fft.h"
typedef FftPlan plan_t;
typedef cplx work_t;
typedef float32_t elem_t;
typedef arm_cfft_instance_f32 cmsis_t;
#define ARITH_NAME "float"
#define PLAN_INIT(p, n) fft_plan_init((p), (n))
#define OWN_FFT(p, x) fft_inplace((p), (x))
#define CMSIS_FFT(s, x) arm_cfft_f32((s), (x), 0, 1)
#define CMSIS_128 (&arm_cfft_sR_f32_len128)
#define CMSIS_32 (&arm_cfft_sR_f32_len32)
#endif

/* CMSIS-DSP takes a flat re,im,re,im array; the project's samples are
 * structs. The cast below is only valid while these have no padding. */
_Static_assert(sizeof(work_t) == 2 * sizeof(elem_t), "work_t must be two packed elements");

#define ITERS 1000
#define CANARY 0xC5

/* From the linker script: the first byte after .bss, where the heap
 * starts, and the top of RAM, where the stack starts. */
extern char _end;
extern char _estack;

typedef struct {
    int n;
    work_t *work;
    const vec_sample *dc;
    const vec_sample *radar;
    const cmsis_t *cmsis;
    plan_t plan;
} size_case;

typedef void (*fft_fn)(const size_case *c);

static work_t work_128[VEC_N_128];
static work_t work_32[VEC_N_32];
/* Holds the project's output while CMSIS-DSP runs on the same input. */
static work_t own_result[VEC_N_128];

static size_case cases[] = {
    { VEC_N_128, work_128, vec_dc_128, vec_radar_128, CMSIS_128, {0} },
    { VEC_N_32, work_32, vec_dc_32, vec_radar_32, CMSIS_32, {0} },
};

static int plans_ok;
static unsigned heap_after_init;
static unsigned free_start;

/* Where the heap currently ends. newlib's sbrk(0) reports the break
 * without moving it. */
static char *heap_end(void)
{
    return (char *)sbrk(0);
}

static unsigned free_ram(void)
{
    char probe;
    return (unsigned)(&probe - heap_end());
}

/* Fill the unused middle of RAM with a known byte, so that afterwards the
 * lowest address the stack ever touched can be recovered by scanning for
 * the first byte still holding it. */
static void paint_stack(void)
{
    char probe;
    /* Stop short of this frame. The bound is computed as an address, not
     * as pointer arithmetic on the local, which the compiler is entitled
     * to reject. */
    uintptr_t top = (uintptr_t)&probe - 32;
    for (char *p = heap_end(); (uintptr_t)p < top; p++) *p = (char)CANARY;
}

/* Scanning starts at the heap's current end rather than at its start: by
 * the time this runs, malloc has overwritten the painted bytes below that
 * point — printf's own buffer among them — and counting them would report
 * the heap as stack. */
static unsigned stack_peak(void)
{
    char *p = heap_end();
    while (p < &_estack && *p == (char)CANARY) p++;
    return (unsigned)(&_estack - p);
}

/* Not inlined, so the reload costs the same in both timing loops. */
static __attribute__((noinline)) void load(work_t *work, const vec_sample *v, int n)
{
    for (int i = 0; i < n; i++) {
        work[i].re = v[i].re;
        work[i].im = v[i].im;
    }
}

/* Both FFTs are reached through the same kind of indirect call, so that call
 * adds the same small cost to each. */
static void run_own(const size_case *c)
{
    OWN_FFT(&c->plan, c->work);
}

static void run_cmsis(const size_case *c)
{
    CMSIS_FFT(c->cmsis, (elem_t *)c->work);
}

static void print_value(const char *key, double v)
{
#if ARITH_Q15
    printf(" %s=%d", key, (int)v);
#else
    printf(" %s=%.6e", key, v);
#endif
}

static double abs_max(double a, double b)
{
    a = fabs(a);
    b = fabs(b);
    return a > b ? a : b;
}

static void report_dc(const size_case *c, const char *impl, fft_fn fn)
{
    load(c->work, c->dc, c->n);
    fn(c);
    double other = 0.0;
    for (int i = 1; i < c->n; i++) {
        double m = abs_max(c->work[i].re, c->work[i].im);
        if (m > other) other = m;
    }
    printf("OUT n=%d impl=%s vec=dc", c->n, impl);
    print_value("bin0_re", c->work[0].re);
    print_value("bin0_im", c->work[0].im);
    print_value("other_peak", other);
    printf("\n");
}

/* Compares real and imaginary parts as separate elements. */
static void report_diff(const size_case *c)
{
    load(c->work, c->radar, c->n);
    run_own(c);
    for (int i = 0; i < c->n; i++) own_result[i] = c->work[i];

    load(c->work, c->radar, c->n);
    run_cmsis(c);

    double max_diff = 0.0, sum2 = 0.0, own_peak = 0.0, cmsis_peak = 0.0;
    for (int i = 0; i < c->n; i++) {
        double dr = (double)own_result[i].re - (double)c->work[i].re;
        double di = (double)own_result[i].im - (double)c->work[i].im;
        double d = abs_max(dr, di);
        double a = abs_max(own_result[i].re, own_result[i].im);
        double b = abs_max(c->work[i].re, c->work[i].im);
        if (d > max_diff) max_diff = d;
        if (a > own_peak) own_peak = a;
        if (b > cmsis_peak) cmsis_peak = b;
        sum2 += dr * dr + di * di;
    }
    printf("CMP n=%d vec=radar", c->n);
    print_value("own_peak", own_peak);
    print_value("cmsis_peak", cmsis_peak);
    printf(" max_abs_diff=%.6g rms_diff=%.6g\n", max_diff, sqrt(sum2 / (2.0 * c->n)));
}

/* Two loops with the same iteration count, because the transform is
 * destructive: every iteration must reload the input, and that reload has
 * to come back out of the number. Interrupts are off for the duration so
 * the millisecond SysTick does not land inside the count. */
static void report_time(const size_case *c, const char *impl, fft_fn fn)
{
    __disable_irq();
    uint32_t t0 = DWT->CYCCNT;
    for (int k = 0; k < ITERS; k++) load(c->work, c->radar, c->n);
    uint32_t t1 = DWT->CYCCNT;
    for (int k = 0; k < ITERS; k++) { load(c->work, c->radar, c->n); fn(c); }
    uint32_t t2 = DWT->CYCCNT;
    __enable_irq();

    double copy_cyc = (double)(t1 - t0) / ITERS;
    double total_cyc = (double)(t2 - t1) / ITERS;
    printf("TIME n=%d impl=%s iters=%d copy_cyc=%.1f total_cyc=%.1f fft_cyc=%.1f\n",
           c->n, impl, ITERS, copy_cyc, total_cyc, total_cyc - copy_cyc);
}

#endif /* !BRINGUP */

static void print_config(void)
{
    /* Both halves have to agree: the compiler was told to use the FPU, and
     * the runtime turned it on (CPACR CP10/CP11 full access). */
    int fpu_compiled = 0, fpu_enabled = 0;
#if defined(__FPU_USED) && (__FPU_USED == 1)
    fpu_compiled = 1;
#endif
    fpu_enabled = ((SCB->CPACR >> 20) & 0xF) == 0xF;

    int loopunroll = 0;
#ifdef ARM_MATH_LOOPUNROLL
    loopunroll = 1;
#endif

    uint32_t acr = FLASH->ACR;
    printf("# dsp_loopunroll %d\n", loopunroll);
    printf("# clk %lu\n", (unsigned long)SystemCoreClock);
    printf("# fpu %d\n", fpu_compiled && fpu_enabled);
    printf("# flash_ws %lu\n", (unsigned long)(acr & FLASH_ACR_LATENCY));
    printf("# flash_acr 0x%08lx\n", (unsigned long)acr);
}

void harness_init(void)
{
    /* The cycle counter is off after reset and reads zero until armed. */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
#if !BRINGUP
    paint_stack();
    free_start = free_ram();
    plans_ok = 1;
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++)
        if (PLAN_INIT(&cases[i].plan, cases[i].n) != 0) plans_ok = 0;
    /* Taken before the first printf, which allocates stdout's buffer;
     * this is the kernel's heap, not the C library's. */
    heap_after_init = (unsigned)(heap_end() - &_end);
#endif
}

void harness_run(void)
{
    printf("# nucleo-fft-cmsis 1\n");
    printf("# arith %s\n", ARITH_NAME);
    print_config();
#if !BRINGUP
    if (!plans_ok) {
        printf("# ERROR plan init failed\n");
        printf("END\n");
        return;
    }
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const size_case *c = &cases[i];
        report_dc(c, "own", run_own);
        report_dc(c, "cmsis", run_cmsis);
        report_diff(c);
        report_time(c, "own", run_own);
        report_time(c, "cmsis", run_cmsis);
    }
    printf("MEM heap=%u stack_peak=%u free_start=%u\n",
           heap_after_init, stack_peak(), free_start);
#endif
    printf("END\n");
}
