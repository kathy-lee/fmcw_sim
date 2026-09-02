#include <stdlib.h>
#include <string.h>
#include "rd_q15.h"
#include "rd_stream_q15.h"
#include "check.h"

#define C 32
#define S 128
#define R (S / 2)

static uint32_t rng = 22222227u;
/* bounded like the quantizer's output (left-aligned by 3): |comp| <= 16384,
 * the domain in which the DC removal provably cannot clip */
static int16_t sample(void)
{
    rng = rng * 1664525u + 1013904223u;
    return (int16_t)((rng >> 16) % 32769u) - 16384;
}

int main(void)
{
    static cplx_q15 in[C * S], out_ref[C * R], plane[C * R];
    static cplx_q15 rowbuf[S], colbuf[C];

    RdPlanQ15 ref;
    RdStreamPlanQ15 st;
    CHECK(rd_q15_plan_init(&ref, C, S) == 0);
    CHECK(rd_stream_q15_plan_init(&st, C, S) == 0);

    /* Same input, two orderings: the whole channel at once through the
     * frozen stage, then the same rows one at a time through the
     * streaming one. Anything but bit-equality is an operation-order
     * divergence, not precision loss. */
    for (int trial = 0; trial < 3; trial++) {
        for (int i = 0; i < C * S; i++) { in[i].re = sample(); in[i].im = sample(); }

        long long sat_ref = 0, sat_st = 0;
        rd_q15_process_channel(&ref, in, out_ref, &sat_ref);

        for (int c = 0; c < C; c++) {
            memcpy(rowbuf, in + (size_t)c * S, sizeof rowbuf);
            rd_stream_q15_row(&st, rowbuf, c, plane, &sat_st);
        }
        rd_stream_q15_plane_finish(&st, plane, colbuf, &sat_st);

        CHECK(memcmp(plane, out_ref, sizeof out_ref) == 0);
        CHECK(sat_st == sat_ref);
    }

    rd_q15_plan_free(&ref);
    rd_stream_q15_plan_free(&st);
    CHECK_DONE();
}
