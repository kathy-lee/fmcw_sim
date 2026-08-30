/* Q15 and float FFT on an STM32F446RE: transform fixed input, report what
 * it cost, print everything in the format the host reference also prints.
 *
 * This file is the apparatus, not the subject. It holds no arithmetic — no
 * butterflies, no twiddles, no shifts. Those live in the kernel copied in
 * from c_frontend and compiled unmodified, which is what makes a number
 * measured here a number about the code that ships.
 *
 * The configuration block comes first and is checked by the reader: a
 * wrong clock or a soft-float build produces plausible numbers that
 * measure the wrong thing, and nothing downstream would notice.
 */
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
#define VEC_N 0
#else

#include "vectors.h"

#if ARITH_Q15
#include "fft_q15.h"
typedef FftPlanQ15 plan_t;
typedef cplx_q15 work_t;
#define ARITH_NAME "q15"
#define PLAN_INIT(p, n) fft_q15_plan_init((p), (n))
#define TRANSFORM(p, x) fft_q15_inplace((p), (x), NULL)
#else
#include "fft.h"
typedef FftPlan plan_t;
typedef cplx work_t;
#define ARITH_NAME "float"
#define PLAN_INIT(p, n) fft_plan_init((p), (n))
#define TRANSFORM(p, x) fft_inplace((p), (x))
#endif

#define ITERS 1000
#define CANARY 0xC5

/* From the linker script: the first byte after .bss, where the heap
 * starts, and the top of RAM, where the stack starts. */
extern char _end;
extern char _estack;

static work_t work[VEC_N];
static plan_t plan;
static int plan_ok;
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

/* Two loops with the same iteration count, because the transform is
 * destructive: every iteration must reload the input, and that reload has
 * to come back out of the number. Interrupts are off for the duration so
 * the millisecond SysTick does not land inside the count. Both raw figures
 * are printed so the subtraction can be judged rather than taken on trust. */
static void timing(const plan_t *p)
{
    __disable_irq();
    uint32_t t0 = DWT->CYCCNT;
    for (int k = 0; k < ITERS; k++) load(vec_radar);
    uint32_t t1 = DWT->CYCCNT;

    uint32_t t2 = DWT->CYCCNT;
    for (int k = 0; k < ITERS; k++) { load(vec_radar); TRANSFORM(p, work); }
    uint32_t t3 = DWT->CYCCNT;
    __enable_irq();

    double copy_cyc = (double)(t1 - t0) / ITERS;
    double total_cyc = (double)(t3 - t2) / ITERS;
    double us_per_cyc = 1e6 / (double)SystemCoreClock;
    printf("TIME iters %d copy_cyc %.1f total_cyc %.1f fft_cyc %.1f "
           "copy_us %.4f total_us %.4f fft_us %.4f\n",
           ITERS, copy_cyc, total_cyc, total_cyc - copy_cyc,
           copy_cyc * us_per_cyc, total_cyc * us_per_cyc,
           (total_cyc - copy_cyc) * us_per_cyc);
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

    uint32_t acr = FLASH->ACR;
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
    plan_ok = (PLAN_INIT(&plan, VEC_N) == 0);
    /* Taken before the first printf, which allocates stdout's buffer;
     * this is the kernel's heap, not the C library's. */
    heap_after_init = (unsigned)(heap_end() - &_end);
#endif
}

void harness_run(void)
{
    printf("# nucleo-fft 1\n");
    printf("# arith %s\n", ARITH_NAME);
    printf("# n %d\n", VEC_N);
    print_config();
#if !BRINGUP
    if (!plan_ok) {
        printf("# ERROR plan init failed\n");
        printf("END\n");
        return;
    }
    dump_twiddles(&plan);
    run_and_dump(&plan, vec_dc, "dc");
    run_and_dump(&plan, vec_radar, "radar");
    timing(&plan);
    printf("MEM heap %u stack_peak %u free_start %u\n",
           heap_after_init, stack_peak(), free_start);
#endif
    printf("END\n");
}
