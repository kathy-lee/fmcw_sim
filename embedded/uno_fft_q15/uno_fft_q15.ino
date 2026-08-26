/* Q15 and float FFT on an ATmega328P: transform fixed input, report what it
 * cost, print everything in the format the host reference also prints.
 *
 * This file is the apparatus, not the subject. It holds no arithmetic — no
 * butterflies, no twiddles, no shifts. Those live in the kernel copied in
 * from c_frontend and compiled unmodified, which is what makes a number
 * measured here a number about the code that ships.
 *
 * The kernel headers are C and this file is compiled as C++, so they are
 * included with C linkage; without that, name mangling would leave the
 * calls unresolved at link time.
 */
#include <Arduino.h>
#include <avr/pgmspace.h>
#include <stdint.h>
#include <stdlib.h>

#include "vectors.h"

#if ARITH_Q15
extern "C" {
#include "fft_q15.h"
}
typedef FftPlanQ15 plan_t;
typedef cplx_q15 work_t;
static const char ARITH_NAME[] = "q15";
#define PLAN_INIT(p, n) fft_q15_plan_init((p), (n))
#define TRANSFORM(p, x) fft_q15_inplace((p), (x), NULL)
#else
extern "C" {
#include "fft.h"
}
typedef FftPlan plan_t;
typedef cplx work_t;
static const char ARITH_NAME[] = "float";
#define PLAN_INIT(p, n) fft_plan_init((p), (n))
#define TRANSFORM(p, x) fft_inplace((p), (x))
#endif

#define ITERS 100
#define CANARY 0xC5

extern char __heap_start;
extern char *__brkval;

static work_t work[VEC_N];

/* Where the heap currently ends. The stack grows down from RAMEND toward
 * this point and the heap grows up to it, so it is the boundary between
 * what malloc owns and what the stack may reach. */
static char *heap_end()
{
    return __brkval ? __brkval : &__heap_start;
}

static unsigned free_ram()
{
    char probe;
    return (unsigned)(&probe - heap_end());
}

/* Fill the unused middle of RAM with a known byte, so that afterwards the
 * lowest address the stack ever touched can be recovered by scanning for
 * the first byte still holding it. Called at the top of setup(), which
 * misses whatever the C runtime used before main — a few bytes, and worth
 * not needing an .init1 hook to recover them. */
static void paint_stack()
{
    char probe;
    for (char *p = heap_end(); p < &probe - 32; p++) *p = (char)CANARY;
}

/* Scanning starts at the heap's current end rather than at its start: by
 * the time this runs, malloc has overwritten the painted bytes below that
 * point, and counting them would report the heap as stack. */
static unsigned stack_peak()
{
    char *p = heap_end();
    while (p < (char *)RAMEND && *p == (char)CANARY) p++;
    return (unsigned)(RAMEND - (unsigned)(uintptr_t)p + 1);
}

static void load(const vec_sample *v)
{
    for (int i = 0; i < VEC_N; i++) {
        work[i].re = VEC_RE(v, i);
        work[i].im = VEC_IM(v, i);
    }
}

/* Matching the host's "%d" and "%.6e" exactly, so the two streams diff
 * cleanly. Arduino's float printing cannot produce that form; avr-libc's
 * dtostre can. */
static void print_value(double v)
{
#if ARITH_Q15
    Serial.print((int)v);
#else
    char b[16];
    dtostre(v, b, 6, 0);
    Serial.print(b);
#endif
}

static void dump_twiddles(const plan_t *p)
{
    for (int k = 0; k < VEC_N / 2; k++) {
        Serial.print(F("TW "));
        Serial.print(k);
        Serial.print(' ');
        print_value((double)p->twiddle[k].re);
        Serial.print(' ');
        print_value((double)p->twiddle[k].im);
        Serial.println();
    }
}

static void run_and_dump(const plan_t *p, const vec_sample *v,
                         const __FlashStringHelper *name)
{
    load(v);
    TRANSFORM(p, work);
    Serial.print(F("VEC "));
    Serial.println(name);
    for (int i = 0; i < VEC_N; i++) {
        Serial.print(F("OUT "));
        Serial.print(i);
        Serial.print(' ');
        print_value((double)work[i].re);
        Serial.print(' ');
        print_value((double)work[i].im);
        Serial.println();
    }
}

/* The transform is destructive, so a repeat has to reload its input first;
 * timing the reload separately is what keeps it out of the answer. Both raw
 * figures are printed so the subtraction can be judged rather than taken on
 * trust. */
static void timing(const plan_t *p)
{
    unsigned long t0 = micros();
    for (int k = 0; k < ITERS; k++) load(vec_radar);
    unsigned long t1 = micros();

    unsigned long t2 = micros();
    for (int k = 0; k < ITERS; k++) { load(vec_radar); TRANSFORM(p, work); }
    unsigned long t3 = micros();

    double copy_us = (double)(t1 - t0) / ITERS;
    double total_us = (double)(t3 - t2) / ITERS;
    char b[16];
    Serial.print(F("TIME iters "));
    Serial.print(ITERS);
    Serial.print(F(" copy_us "));
    dtostre(copy_us, b, 4, 0);
    Serial.print(b);
    Serial.print(F(" total_us "));
    dtostre(total_us, b, 4, 0);
    Serial.print(b);
    Serial.print(F(" fft_us "));
    dtostre(total_us - copy_us, b, 4, 0);
    Serial.println(b);
}

void setup()
{
    Serial.begin(115200);
    paint_stack();
    unsigned free_start = free_ram();

    plan_t p = {0};
    if (PLAN_INIT(&p, VEC_N) != 0) {
        Serial.println(F("# ERROR plan init failed"));
        Serial.println(F("END"));
        return;
    }

    Serial.println(F("# uno-fft 1"));
    Serial.print(F("# arith "));
    Serial.println(ARITH_NAME);
    Serial.print(F("# n "));
    Serial.println(VEC_N);

    dump_twiddles(&p);
    run_and_dump(&p, vec_dc, F("dc"));
    run_and_dump(&p, vec_radar, F("radar"));
    timing(&p);

    Serial.print(F("MEM heap "));
    Serial.print((unsigned)(heap_end() - &__heap_start));
    Serial.print(F(" stack_peak "));
    Serial.print(stack_peak());
    Serial.print(F(" free_start "));
    Serial.println(free_start);
    Serial.println(F("END"));
    Serial.flush();
}

void loop() {}
