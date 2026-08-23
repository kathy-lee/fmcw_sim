#ifndef FMCW_Q15_H
#define FMCW_Q15_H
#include <stdint.h>

/* Q15 fixed-point primitives: int16 data path, int32 registers.
 *
 * "Q15" means an int16 read as a fraction in [-1, 1): the stored integer n
 * stands for n/32768. That interpretation exists only in the shift amounts
 * below — the hardware sees plain integers either way. Note that 1.0 is NOT
 * representable; the largest value is 32767/32768.
 *
 * Rounding convention throughout: halves go toward +infinity. This is the
 * cheap form, (x + half) >> shift, and the one CMSIS-DSP uses. It carries a
 * sub-LSB DC bias — a deliberate trade, pinned by test_q15 rather than left
 * as a surprise.
 *
 * Portability note: the shifts below rely on right-shift of a negative
 * signed value being arithmetic. C11 leaves that implementation-defined;
 * every compiler this project targets defines it that way.
 */

#define Q15_MIN (-32768)
#define Q15_MAX ( 32767)
#define Q15_ONE ( 32768)   /* the scale factor: one past the largest value */

/* Interleaved Q15 I/Q sample — the integer counterpart of cplx, and half
 * its width, which is what halves the radar cube. */
typedef struct { int16_t re, im; } cplx_q15;

/* Narrow to int16, saturating. Counts each clipped value into *sat, which
 * may be NULL to ignore. Saturation is meant to be measured, not silently
 * absorbed: a nonzero count means an input assumption was violated. */
static inline int16_t sat16(int32_t x, long long *sat)
{
    if (x > Q15_MAX) { if (sat) (*sat)++; return (int16_t)Q15_MAX; }
    if (x < Q15_MIN) { if (sat) (*sat)++; return (int16_t)Q15_MIN; }
    return (int16_t)x;
}

/* Q15 multiply: (a/32768) * (b/32768) -> result/32768.
 *
 * The product of two int16 is Q30, so it shifts back down by 15. Only the
 * positive side can leave int16: the most negative product, -32768 * 32767,
 * rounds to -32767 and fits with a bit to spare, while -32768 * -32768
 * gives +32768, one past the top. Hence the one-sided clamp — a negative
 * branch here would be unreachable. */
static inline int16_t q15_mul(int16_t a, int16_t b)
{
    int32_t p = (((int32_t)a * (int32_t)b) + 0x4000) >> 15;
    return (int16_t)(p > Q15_MAX ? Q15_MAX : p);
}

/* Halve with rounding. Domain is the FFT butterfly's sum or difference of
 * two int16 values, [-65536, 65534], which maps exactly onto [-32768,
 * 32767] — so no saturation is needed here, and the shift is what keeps the
 * transform in range stage after stage. */
static inline int16_t shr1_round(int32_t x)
{
    return (int16_t)((x + 1) >> 1);
}

/* Componentwise difference, saturating. Used for the DC and slow-time mean
 * removals, where the subtraction can double a magnitude. */
static inline cplx_q15 cq15_sub_sat(cplx_q15 a, cplx_q15 b, long long *sat)
{
    cplx_q15 r;
    r.re = sat16((int32_t)a.re - (int32_t)b.re, sat);
    r.im = sat16((int32_t)a.im - (int32_t)b.im, sat);
    return r;
}

/* Complex Q15 multiply where w is a UNIT-MAGNITUDE twiddle, meaning
 * w.re^2 + w.im^2 == 32768^2. The precondition is in the name because it is
 * what keeps the int32 intermediates safe:
 *
 *   real part  a.re*w.re - a.im*w.im reaches 2147450880, plus the rounding
 *              term 2147467264 — just inside INT32_MAX. Safe for any
 *              operands at all.
 *   imag part  a.re*w.im + a.im*w.re would reach 2^31, one past INT32_MAX,
 *              if both components of both operands were -32768. A
 *              unit-magnitude w cannot have both components at -32768
 *              (that vector has magnitude 46341), so it cannot happen here.
 *
 * The RESULT can still leave int16 and saturate. Rotation preserves
 * magnitude, so the output component is bounded by |a|, not by max(|a.re|,
 * |a.im|) — a value with both components near full scale has magnitude
 * 46341 and will clip at some rotation. Callers wanting saturation-free
 * operation must keep |a| <= 32767, which is a bound on the magnitude, not
 * on the components. */
static inline cplx_q15 cq15_mul_tw(cplx_q15 a, cplx_q15 w, long long *sat)
{
    int32_t re = (((int32_t)a.re * w.re) - ((int32_t)a.im * w.im) + 0x4000) >> 15;
    int32_t im = (((int32_t)a.re * w.im) + ((int32_t)a.im * w.re) + 0x4000) >> 15;
    cplx_q15 r = { sat16(re, sat), sat16(im, sat) };
    return r;
}

/* Squared magnitude, exact: integer multiply-add with no rounding anywhere.
 * One sample reaches 2^31, past what int32 holds, which is why the
 * integration accumulator downstream is int64. */
static inline int64_t cq15_mag2(cplx_q15 a)
{
    return (int64_t)a.re * a.re + (int64_t)a.im * a.im;
}
#endif
