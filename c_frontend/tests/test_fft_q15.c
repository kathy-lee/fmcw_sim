#include <math.h>
#include "fmcw.h"
#include "fft.h"
#include "fft_q15.h"
#include "check.h"

#define N 64

/* Deterministic pseudo-random components; unsigned arithmetic so the
 * wraparound is defined. */
static uint32_t rng = 12345u;
static int rnd(int lo, int hi)
{
    rng = rng * 1103515245u + 12345u;
    return lo + (int)((rng >> 16) % (uint32_t)(hi - lo + 1));
}

static void fill_tone(cplx_q15 *x, int n, int k, double amp)
{
    for (int j = 0; j < n; j++) {
        double ang = 2.0 * FMCW_PI * k * j / (double)n;
        x[j].re = (int16_t)lround(amp * cos(ang));
        x[j].im = (int16_t)lround(amp * sin(ang));
    }
}

int main(void)
{
    FftPlanQ15 p = {0};
    cplx_q15 x[N];
    long long sat;

    /* Invalid sizes are rejected, and free is safe on the failed plan. */
    CHECK(fft_q15_plan_init(&p, 6) == -1);
    fft_q15_plan_free(&p);
    CHECK(fft_q15_plan_init(&p, N) == 0);

    /* ---- 1. a constant transforms to bin 0, exactly ----
     *
     * The DFT of a constant A is n*A in bin 0 and zero elsewhere, so with
     * the 1/n scaling the answer is A itself. Getting it back to the last
     * count is the sharpest available check that the per-stage halvings
     * compose to exactly 1/n — one shift too many or too few and this bin
     * is off by a factor of two. */
    for (int i = 0; i < N; i++) x[i] = (cplx_q15){1000, 0};
    sat = 0;
    fft_q15_inplace(&p, x, &sat);
    CHECK(x[0].re == 1000 && x[0].im == 0);
    for (int i = 1; i < N; i++) CHECK(x[i].re == 0 && x[i].im == 0);
    CHECK(sat == 0);

    /* ---- 2. a tone lands in its own bin, at its own amplitude ---- */
    const int K = 7;
    const double A = 8000.0;
    fill_tone(x, N, K, A);
    sat = 0;
    fft_q15_inplace(&p, x, &sat);
    CHECK(sat == 0);

    int peak = 0;
    int64_t best = -1;
    for (int i = 0; i < N; i++) {
        int64_t m = cq15_mag2(x[i]);
        if (m > best) { best = m; peak = i; }
    }
    CHECK(peak == K);
    CHECK_CLOSE(sqrt((double)best), A, 0.02 * A);

    /* Everything else is rounding noise: a couple of counts, not a lobe.
     * The tone sits exactly on a bin, so there is no leakage to hide it. */
    for (int i = 0; i < N; i++)
        if (i != K) CHECK(cq15_mag2(x[i]) < 2500);      /* magnitude < 50 */

    /* ---- 3. range safety at the magnitude precondition ----
     *
     * A full-scale tone has |x| = amp for every sample, so this sits just
     * under the |x| <= 32767 bound with every component in play. No
     * saturation may occur. (The counter's mechanism itself is exercised
     * in test_q15, on cq15_mul_tw directly: reaching it through the
     * transform is hard, because a DIT FFT applies its off-axis twiddles
     * to values that have already been combined, and single-bin signals
     * concentrate before the rotating stages ever see them.) */
    fill_tone(x, N, 3, 32000.0);
    sat = 0;
    fft_q15_inplace(&p, x, &sat);
    CHECK(sat == 0);

    /* ---- 4. agreement with the float transform ---- */
    {
        FftPlan pf = {0};
        cplx_q15 xq[N];
        cplx xf[N];
        CHECK(fft_plan_init(&pf, N) == 0);
        for (int i = 0; i < N; i++) {
            int re = rnd(-8000, 8000), im = rnd(-8000, 8000);
            xq[i] = (cplx_q15){(int16_t)re, (int16_t)im};
            xf[i] = (cplx){(float)re, (float)im};
        }
        sat = 0;
        fft_q15_inplace(&p, xq, &sat);
        fft_inplace(&pf, xf);
        CHECK(sat == 0);

        /* The integer result carries the 1/n the float one does not, so
         * scale it back before comparing. */
        double peak_f = 0.0, err = 0.0;
        for (int i = 0; i < N; i++) {
            double qr = (double)xq[i].re * N, qi = (double)xq[i].im * N;
            peak_f = fmax(peak_f, fmax(fabs(xf[i].re), fabs(xf[i].im)));
            err = fmax(err, fmax(fabs(qr - xf[i].re), fabs(qi - xf[i].im)));
        }
        CHECK(err <= 0.01 * peak_f);
        fft_plan_free(&pf);
    }

    /* ---- 5. fftshift swaps halves ---- */
    {
        cplx_q15 v[4] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
        fftshift_q15_inplace(v, 4);
        CHECK(v[0].re == 3 && v[1].re == 4 && v[2].re == 1 && v[3].re == 2);
    }

    fft_q15_plan_free(&p);
    CHECK_DONE();
}
