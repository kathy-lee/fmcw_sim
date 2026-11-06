#ifndef FMCW_H
#define FMCW_H
#include <stddef.h>

#define FMCW_PI 3.14159265358979323846

/* Interleaved float I/Q sample — memory-layout-identical to the exported
 * frame format (float32 re, float32 im). */
typedef struct { float re, im; } cplx;

static inline cplx cadd(cplx a, cplx b) { return (cplx){a.re + b.re, a.im + b.im}; }
static inline cplx csub(cplx a, cplx b) { return (cplx){a.re - b.re, a.im - b.im}; }
static inline cplx cmul(cplx a, cplx b)
{
    return (cplx){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline double cmag2(cplx a)
{
    return (double)a.re * a.re + (double)a.im * a.im;
}
#endif
