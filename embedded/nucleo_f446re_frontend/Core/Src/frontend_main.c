/* One-pass Q15 detection front end on the NUCLEO-F446RE: banner and idle
 * loop. The self-description prints what a reader must refuse to proceed
 * without: clock, FPU, flash timing, the compiled dimensions and detector
 * parameters — a feeder that sees a different configuration than its own
 * must not send a frame. */
#include <stdint.h>
#include <stdio.h>
#include "main.h"
#include "frontend.h"
#include "frontend_params.h"

static char stdout_buf[512];

static void print_config(void)
{
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

static void banner(void)
{
    printf("# onepass-frontend 1\n");
    print_config();
    printf("# dims %d %d %d\n", FRONTEND_V, FRONTEND_C, FRONTEND_S);
    printf("# cfar %d %d %d %d %.17g %d\n",
           FRONTEND_PARAMS.cfar_guard_d, FRONTEND_PARAMS.cfar_guard_r,
           FRONTEND_PARAMS.cfar_training_d, FRONTEND_PARAMS.cfar_training_r,
           FRONTEND_PARAMS.cfar_alpha, FRONTEND_PARAMS.min_range_bins);
    printf("# doa 181 %d\n", FRONTEND_PARAMS.doa_n_sources);
    printf("# maxdet 64\n");
    printf("RDY\n");
    fflush(stdout);
}

void frontend_init(void)
{
    /* buffer stdout through a static array: printf never touches the
     * heap, so the heap line reports only the plan tables */
    setvbuf(stdout, stdout_buf, _IOFBF, sizeof stdout_buf);
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
}

void frontend_run(void)
{
    banner();
    HAL_Delay(2000);
}
