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
#include "main.h"
#include "harness.h"

#ifndef BRINGUP
#define BRINGUP 0
#endif

#if BRINGUP
#define ARITH_NAME "bringup"
#define VEC_N 0
#endif

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
}

void harness_run(void)
{
    printf("# nucleo-fft 1\n");
    printf("# arith %s\n", ARITH_NAME);
    printf("# n %d\n", VEC_N);
    print_config();
    printf("END\n");
}
