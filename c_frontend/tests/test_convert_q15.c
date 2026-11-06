#include "convert_q15.h"
#include "check.h"

/* Full scale of 2048 makes the 12-bit step exactly 1.0, so a float value
 * quantizes to its own rounded self and every expectation below can be read
 * without arithmetic. */
#define FS 2048.0

int main(void)
{
    long long sat = 0;
    cplx in[8];
    cplx_q15 q[8];
    cplx back[8];

    /* ---- forward: grid, left-alignment, clipping ---- */

    in[0] = (cplx){100.0f, -100.0f};    /* code 100 -> 100 << 3          */
    in[1] = (cplx){0.0f, 0.4f};         /* rounds to 0                   */
    in[2] = (cplx){0.5f, -0.5f};        /* half away from zero -> +/-1   */
    in[3] = (cplx){-2048.0f, 2047.0f};  /* the extreme codes, no clip    */
    in[4] = (cplx){2048.0f, 0.0f};      /* one past the top code -> clip */
    in[5] = (cplx){5000.0f, -9000.0f};  /* far outside -> clips both     */

    /* First four only: nothing here reaches the rails, so the counter must
     * stay at zero. */
    adc_to_q15(in, 4, FS, q, &sat);

    CHECK(q[0].re == 800 && q[0].im == -800);
    CHECK(q[1].re == 0 && q[1].im == 0);
    CHECK(q[2].re == 8 && q[2].im == -8);

    /* -2048 is representable, +2048 is not: the two's-complement asymmetry
     * shows up as a clip on the positive side only. Left-aligned by 3, so
     * the usable range is +/-16384 rather than +/-32768 — the top bit is
     * the guard the DC removal needs. */
    CHECK(q[3].re == -16384 && q[3].im == 16376);
    CHECK(sat == 0);

    /* Now the two that do clip, counted cumulatively. */
    adc_to_q15(&in[4], 1, FS, &q[4], &sat);
    CHECK(q[4].re == 16376 && sat == 1);      /* im is 0.0: not a clip */

    adc_to_q15(&in[5], 1, FS, &q[5], &sat);
    CHECK(q[5].re == 16376 && q[5].im == -16384 && sat == 3);

    /* NULL counter is tolerated. */
    adc_to_q15(&in[5], 1, FS, &q[5], NULL);
    CHECK(q[5].re == 16376);

    /* ---- reverse: round-trip within half a step ---- */

    double one_sample = q15_rd_scale(FS, 1, 1);
    CHECK_CLOSE(one_sample, 0.125, 1e-12);       /* FS / 16384 */

    for (int i = 0; i < 8; i++)
        in[i] = (cplx){(float)(i * 3.7 - 12.0), (float)(-i * 1.3)};
    adc_to_q15(in, 8, FS, q, &sat);
    q15_to_cplx(q, 8, one_sample, back);
    for (int i = 0; i < 8; i++) {
        CHECK_CLOSE(back[i].re, in[i].re, 0.5 + 1e-6);
        CHECK_CLOSE(back[i].im, in[i].im, 0.5 + 1e-6);
    }

    /* ---- the cube's reconstruction factor ---- */

    /* Each FFT contributes 1/length, so the cube sits at K/(S*C) of the
     * float value. At S=256, C=64 the product is 16384, exactly K's
     * numerator, so the factor reduces to adc_full_scale itself. */
    CHECK_CLOSE(q15_rd_scale(FS, 256, 64), FS, 1e-9);
    /* Halving a dimension halves the factor. */
    CHECK_CLOSE(q15_rd_scale(FS, 128, 64), FS / 2.0, 1e-9);

    CHECK_DONE();
}
