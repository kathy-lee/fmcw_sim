#include "fmcw.h"
#include "fft.h"
#include "check.h"

/* Largest |a[i] - b[i]| over two complex arrays — the error metric for
 * comparing our FFT output against the NumPy golden vectors. */
static double max_err(const cplx *a, const cplx *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double dre = (double)a[i].re - b[i].re;
        double dim = (double)a[i].im - b[i].im;
        double e = sqrt(dre * dre + dim * dim);
        if (e > m) m = e;
    }
    return m;
}

/* Largest |a[i]| — the scale we make the error tolerance relative to, so the
 * check is meaningful regardless of the input's magnitude. */
static double max_mag(const cplx *a, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double e = sqrt(cmag2(a[i]));
        if (e > m) m = e;
    }
    return m;
}

/* Golden-vector check: run our size-n FFT on the same input NumPy saw and demand
 * the result match NumPy's output within 1e-5 of the spectrum peak (float32
 * rounding leaves errors well below that; anything above is a real bug). */
static void check_against_numpy(int n)
{
    cplx in[64], ref[64];
    char path[64];
    snprintf(path, sizeof path, "tests/vectors/fft%d_in.bin", n);
    CHECK(load_bin(path, in, (size_t)n * sizeof(cplx)) == 0);
    snprintf(path, sizeof path, "tests/vectors/fft%d_out.bin", n);
    CHECK(load_bin(path, ref, (size_t)n * sizeof(cplx)) == 0);

    FftPlan plan;
    CHECK(fft_plan_init(&plan, n) == 0);
    fft_inplace(&plan, in);              /* in-place: `in` becomes the output */
    CHECK(max_err(in, ref, n) <= 1e-5 * max_mag(ref, n));
    fft_plan_free(&plan);
}

int main(void)
{
    /* Analytic sanity case, checkable by hand: the DFT of a delta impulse
     * [1, 0, 0, ...] is all-ones (every bin sums exactly one sample). */
    cplx delta[8] = {{1.0f, 0.0f}};      /* rest zero-initialized by C rules */
    FftPlan p8;
    CHECK(fft_plan_init(&p8, 8) == 0);
    fft_inplace(&p8, delta);
    for (int i = 0; i < 8; i++) {
        CHECK_CLOSE(delta[i].re, 1.0, 1e-6);
        CHECK_CLOSE(delta[i].im, 0.0, 1e-6);
    }
    fft_plan_free(&p8);

    /* Against NumPy for both sizes the pipeline will use plan sizes for. */
    check_against_numpy(16);
    check_against_numpy(64);

    /* fftshift on even n swaps halves: [0 1 | 2 3] -> [2 3 | 0 1]. */
    cplx v[4] = {{0,0},{1,0},{2,0},{3,0}};
    fftshift_inplace(v, 4);
    CHECK_CLOSE(v[0].re, 2.0, 1e-9);
    CHECK_CLOSE(v[1].re, 3.0, 1e-9);
    CHECK_CLOSE(v[2].re, 0.0, 1e-9);
    CHECK_CLOSE(v[3].re, 1.0, 1e-9);

    /* Invalid sizes rejected: zero and non-power-of-two. */
    FftPlan bad;
    CHECK(fft_plan_init(&bad, 0) == -1);
    CHECK(fft_plan_init(&bad, 12) == -1);
    CHECK_DONE();
}
