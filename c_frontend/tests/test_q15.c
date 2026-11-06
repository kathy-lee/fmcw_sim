#include "fmcw_q15.h"
#include "check.h"

int main(void)
{
    long long sat = 0;

    /* ---- q15_mul ---- */

    /* 0.5 * 0.5 = 0.25 */
    CHECK(q15_mul(16384, 16384) == 8192);

    /* Rounding is half-toward-+infinity, NOT half-away-from-zero the way
     * Python's quantize() rounds. Both products below are exactly 0.5 LSB. */
    CHECK(q15_mul(1, 16384) == 1);
    CHECK(q15_mul(-1, 16384) == 0);

    /* The one product that leaves int16: -1.0 * -1.0 would be +1.0, which
     * Q15 cannot represent (its top value is 32767/32768). */
    CHECK(q15_mul(-32768, -32768) == 32767);

    /* The most negative product lands at -32767, inside int16 — so the
     * clamp above is one-sided by design and a negative branch would be
     * dead code. (The rounding term lifts it one LSB clear of the floor:
     * -32768*32767 + 0x4000 is -32768 * 32766.5, which floors to -32767.) */
    CHECK(q15_mul(-32768, 32767) == -32767);

    /* ---- shr1_round ---- */

    CHECK(shr1_round(3) == 2);
    CHECK(shr1_round(-3) == -1);

    /* The extremes of the FFT butterfly's input range land exactly on the
     * int16 bounds — this is what makes the per-stage shift safe. */
    CHECK(shr1_round(-65536) == -32768);
    CHECK(shr1_round(65534) == 32767);

    /* ---- sat16 ---- */

    CHECK(sat16(40000, &sat) == 32767 && sat == 1);
    CHECK(sat16(-40000, &sat) == -32768 && sat == 2);
    CHECK(sat16(100, &sat) == 100 && sat == 2);   /* in range: not counted */
    CHECK(sat16(40000, NULL) == 32767);           /* NULL counter tolerated */

    /* ---- cq15_sub_sat ---- */

    cplx_q15 p = {1000, -2000}, q = {400, 500};
    cplx_q15 d = cq15_sub_sat(p, q, &sat);
    CHECK(d.re == 600 && d.im == -2500 && sat == 2);

    cplx_q15 lo = {-32768, 0}, hi = {32767, 0};
    d = cq15_sub_sat(lo, hi, &sat);
    CHECK(d.re == -32768 && sat == 3);            /* clipped and counted */

    /* ---- cq15_mul_tw ---- */

    sat = 0;
    cplx_q15 a = {1000, -2000};

    /* Twiddle k=0 is exp(0) = 1.0, which Q15 cannot hold; it clamps to
     * 32767, so this is near-identity rather than exact identity. */
    cplx_q15 one = {32767, 0};
    cplx_q15 r = cq15_mul_tw(a, one, &sat);
    CHECK(r.re == 1000 && r.im == -2000 && sat == 0);

    /* Multiplying by -i maps (re, im) -> (im, -re). */
    cplx_q15 minus_i = {0, -32768};
    r = cq15_mul_tw(a, minus_i, &sat);
    CHECK(r.re == -2000 && r.im == -1000 && sat == 0);

    /* A rotation of a value whose MAGNITUDE exceeds 32767 can push a
     * component past int16, so the result saturates and is counted. This
     * is the precondition fft_q15 has to keep: |x| <= 32767, not merely
     * |re|,|im| <= 32767. Here |a| = 46341, and a 45-degree twiddle turns
     * that into a component of 46339. */
    cplx_q15 big = {32767, 32767};
    cplx_q15 w45 = {23170, -23170};               /* exp(-i*pi/4) in Q15 */
    r = cq15_mul_tw(big, w45, &sat);
    CHECK(r.re == 32767 && sat == 1);

    /* ---- cq15_mag2 ---- */

    CHECK(cq15_mag2(a) == 1000LL * 1000 + 2000LL * 2000);

    /* Exact at the extreme, and past what int32 could hold — which is why
     * the integration accumulator is int64. */
    cplx_q15 corner = {-32768, -32768};
    CHECK(cq15_mag2(corner) == 2147483648LL);

    CHECK(sizeof(cplx_q15) == 2 * sizeof(int16_t));  /* interleaved I/Q */
    CHECK_DONE();
}
