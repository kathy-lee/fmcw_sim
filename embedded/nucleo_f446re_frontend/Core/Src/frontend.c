/* One-pass Q15 detection front end on the NUCLEO-F446RE.
 *
 * This file is the apparatus: static storage, the UART conversation, and
 * the reporting. The arithmetic lives in the kernels copied in from
 * c_frontend and compiled unmodified — a detection printed here is a
 * detection the host-tested code computed.
 *
 * The board paces the wire: it prints RDY when it can accept a frame, and
 * the feeder transmits only after seeing it. Everything is blocking
 * programmed I/O — at 921600 baud the UART, not the CPU, is the bottleneck,
 * and the row FFT (~190 us) hides entirely inside each row's ~5.6 ms
 * transfer time. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "main.h"
#include "frontend.h"
#include "frontend_params.h"
#include "onepass_q15.h"

extern UART_HandleTypeDef huart2;
extern char _end;
extern char _estack;

#define CANARY 0xC5
#define R_BINS (FRONTEND_S / 2)
#define HDR_LEN 14
#define ROW_BYTES (FRONTEND_S * 4)

/* ---- static storage: every lifetime the linker can see ---- */

static cplx_q15 planes[FRONTEND_V * FRONTEND_C * R_BINS];

/* power is dead once the CFAR walk has produced the cell list; snapshots
 * and the DoA spectrum then live in its storage (disjoint sub-ranges,
 * snapshots first) — the overlay onepass_q15_finish's contract permits */
static union {
    int64_t power[FRONTEND_C * R_BINS];
    struct {
        cplx_q15 snapshots[ONEPASS_MAX_DET * FRONTEND_V];
        double spectrum[181];
    } post;
} shared;

static CfarCellQ15 cells_list[ONEPASS_MAX_DET];
static cplx_q15 rowbuf_rx[FRONTEND_S];
static OnepassPlanQ15 plan;
static int plan_ok;
static unsigned heap_after_init;
static unsigned free_start;
static char stdout_buf[512];

/* ---- memory instruments (the harness's forms) ---- */

static char *heap_end(void) { return (char *)sbrk(0); }

static unsigned free_ram(void)
{
    char probe;
    return (unsigned)(&probe - heap_end());
}

static void paint_stack(void)
{
    char probe;
    uintptr_t top = (uintptr_t)&probe - 32;
    for (char *p = heap_end(); (uintptr_t)p < top; p++) *p = (char)CANARY;
}

static unsigned stack_peak(void)
{
    char *p = heap_end();
    while (p < &_estack && *p == (char)CANARY) p++;
    return (unsigned)(&_estack - p);
}

/* ---- self-description ---- */

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
    printf("# maxdet %d\n", ONEPASS_MAX_DET);
    if (!plan_ok) printf("# ERROR plan init failed\n");
    printf("RDY\n");
    fflush(stdout);
}

/* ---- the wire ---- */

static void drain_rx(void)
{
    uint8_t b;
    while (HAL_UART_Receive(&huart2, &b, 1, 50) == HAL_OK) { }
}

/* Detections are printed as they are computed, from V-sized locals; the
 * cycles spent printing are counted here and subtracted from the finish
 * phase so TIME reports compute only. The DET count line rides on the
 * first emission — the cell walk is complete by then, so plan.n_cells and
 * plan.overflow are already set. */
typedef struct { int first; } EmitCtx;

static uint32_t io_cyc;

static void print_det(void *vctx, const OnepassDet *d)
{
    EmitCtx *ctx = vctx;
    uint32_t t0 = DWT->CYCCNT;
    if (ctx->first) {
        printf("DET %d %d\n", plan.n_cells, plan.overflow);
        ctx->first = 0;
    }
    printf("D %d %d %d %.9e %.9e %.9e %.9e %.17g %.17g\n",
           d->range_bin, d->doppler_bin, d->az_idx,
           d->x, d->y, d->v_radial, d->snr_db,
           (double)d->power, (double)d->thresh_q16 / 65536.0);
    fflush(stdout);
    io_cyc += DWT->CYCCNT - t0;
}

static void report_mem(void)
{
    printf("MEM heap %u stack_peak %u free_start %u\n",
           heap_after_init, stack_peak(), free_start);
}

void frontend_init(void)
{
    /* buffer stdout through a static array: printf never touches the
     * heap, so the heap line reports only the plan tables */
    setvbuf(stdout, stdout_buf, _IOFBF, sizeof stdout_buf);
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
    paint_stack();
    free_start = free_ram();
    plan_ok = (onepass_q15_init(&plan, &FRONTEND_PARAMS, planes,
                                shared.power, cells_list,
                                shared.post.snapshots, shared.post.spectrum,
                                rowbuf_rx) == 0);
    /* the kernel's heap only: taken before the first printf would have
     * allocated anything (it never does — stdout is statically buffered) */
    heap_after_init = (unsigned)(heap_end() - &_end);
}

/* Called in a loop from main(); one call is one idle window or one frame. */
void frontend_run(void)
{
    banner();
    /* the banner-only cut idled here instead of listening:
     *     HAL_Delay(2000);
     * the receive timeout below waits the same two seconds, listening */
    if (!plan_ok) { HAL_Delay(2000); return; }

    uint8_t hdr[HDR_LEN];
    if (HAL_UART_Receive(&huart2, hdr, HDR_LEN, 2000) != HAL_OK)
        return;                              /* idle: banner again next call */

    uint32_t frame_id = (uint32_t)hdr[4] | ((uint32_t)hdr[5] << 8)
                      | ((uint32_t)hdr[6] << 16) | ((uint32_t)hdr[7] << 24);
    unsigned v_in = hdr[8] | (hdr[9] << 8);
    unsigned c_in = hdr[10] | (hdr[11] << 8);
    unsigned s_in = hdr[12] | (hdr[13] << 8);
    if (memcmp(hdr, "OPF1", 4) != 0 ||
        v_in != FRONTEND_V || c_in != FRONTEND_C || s_in != FRONTEND_S) {
        printf("ERR cfg %u %u %u\n", v_in, c_in, s_in);
        fflush(stdout);
        drain_rx();
        return;
    }

    plan.sat = 0;               /* rows saturate too, not just the finish */
    uint32_t row_cyc = 0;
    for (int v = 0; v < FRONTEND_V; v++) {
        for (int c = 0; c < FRONTEND_C; c++) {
            uint8_t ck[2];
            if (HAL_UART_Receive(&huart2, (uint8_t *)rowbuf_rx,
                                 ROW_BYTES, 1000) != HAL_OK ||
                HAL_UART_Receive(&huart2, ck, 2, 1000) != HAL_OK) {
                printf("ERR timeout %d %d\n", v, c);
                fflush(stdout);
                drain_rx();
                return;
            }
            unsigned sum = 0;
            const uint8_t *raw = (const uint8_t *)rowbuf_rx;
            for (int i = 0; i < ROW_BYTES; i++) sum += raw[i];
            if ((sum & 0xFFFFu) != (unsigned)(ck[0] | (ck[1] << 8))) {
                printf("ERR row %d %d\n", v, c);
                fflush(stdout);
                drain_rx();
                return;
            }
            uint32_t t0 = DWT->CYCCNT;
            onepass_q15_row(&plan, v, c);
            row_cyc += DWT->CYCCNT - t0;
        }
    }

    io_cyc = 0;
    EmitCtx ectx = { 1 };
    uint32_t t0 = DWT->CYCCNT;
    int n = onepass_q15_finish(&plan, print_det, &ectx);
    uint32_t total = DWT->CYCCNT - t0;
    if (ectx.first)                 /* zero detections: no emit ever fired */
        printf("DET %d %d\n", n, plan.overflow);
    printf("TIME row_cyc %lu finish_cyc %lu\n",
           (unsigned long)row_cyc, (unsigned long)(total - io_cyc));
    report_mem();
    printf("SAT %ld\n", (long)plan.sat);
    printf("END %lu\n", (unsigned long)frame_id);
    printf("RDY\n");
    fflush(stdout);
}
