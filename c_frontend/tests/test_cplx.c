#include "fmcw.h"
#include "check.h"

int main(void)
{
    cplx a = {1.0f, 2.0f};
    cplx b = {3.0f, -1.0f};

    cplx s = cadd(a, b);
    CHECK_CLOSE(s.re, 4.0, 1e-9);
    CHECK_CLOSE(s.im, 1.0, 1e-9);

    cplx d = csub(a, b);
    CHECK_CLOSE(d.re, -2.0, 1e-9);
    CHECK_CLOSE(d.im, 3.0, 1e-9);

    /* (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 5 + 5i */
    cplx m = cmul(a, b);
    CHECK_CLOSE(m.re, 5.0, 1e-9);
    CHECK_CLOSE(m.im, 5.0, 1e-9);

    CHECK_CLOSE(cmag2(a), 5.0, 1e-9);
    CHECK_CLOSE(FMCW_PI, 3.14159265358979, 1e-12);

    CHECK(sizeof(cplx) == 2 * sizeof(float)); /* interleaved I/Q layout */
    CHECK_DONE();
}
