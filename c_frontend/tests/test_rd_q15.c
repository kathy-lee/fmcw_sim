#include <math.h>
#include "fmcw.h"
#include "rd.h"
#include "rd_q15.h"
#include "check.h"

#define C 16
#define S 32
#define R (S / 2)

int main(void)
{
    /* ---- the Q15 window matches the float one ---- */
    {
        int16_t wq[S];
        double wf[S];
        hann_fill_q15(wq, S);
        hann_fill(wf, S);
        for (int i = 0; i < S; i++)
            CHECK_CLOSE((double)wq[i] / Q15_ONE, wf[i], 1.0 / Q15_ONE);
        /* Even length, so the highest tap falls beside the peak rather than
         * on it and stays under 1.0 — representable without the clamp
         * engaging. How far under depends on n: at S=32 the top tap is
         * 0.997, where a 256-point window would reach 0.99996. */
        int16_t top = 0;
        for (int i = 0; i < S; i++) if (wq[i] > top) top = wq[i];
        CHECK(top > 32000 && top < Q15_MAX);
    }

    RdPlanQ15 p = {0};

    /* Non-power-of-two dims are rejected; free is safe on the failed plan. */
    CHECK(rd_q15_plan_init(&p, C, 30) == -1);
    rd_q15_plan_free(&p);
    CHECK(rd_q15_plan_init(&p, C, S) == 0);

    /* ---- a synthetic target lands in its own cell ----
     *
     * A tone at range bin K_R in fast time and Doppler bin K_D in slow
     * time. fftshift moves Doppler bin k to index (k + C/2) mod C, so that
     * is where the peak must appear. */
    const int K_R = 5, K_D = 3;
    const double A = 8000.0;
    static cplx_q15 in[C * S];
    static cplx_q15 out[C * R];

    for (int c = 0; c < C; c++) {
        for (int s = 0; s < S; s++) {
            double ang = 2.0 * FMCW_PI * ((double)K_R * s / S + (double)K_D * c / C);
            in[c * S + s].re = (int16_t)lround(A * cos(ang));
            in[c * S + s].im = (int16_t)lround(A * sin(ang));
        }
    }

    long long sat = 0;
    rd_q15_process_channel(&p, in, out, &sat);

    /* A moderate input, well inside every bound: nothing may clip. The DC
     * removal cannot clip by construction, so a nonzero count here would
     * mean the clutter removal or the FFT overflowed. */
    CHECK(sat == 0);

    int peak_c = -1, peak_r = -1;
    int64_t best = -1;
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            int64_t m = cq15_mag2(out[c * R + r]);
            if (m > best) { best = m; peak_c = c; peak_r = r; }
        }
    }
    CHECK(peak_r == K_R);
    CHECK(peak_c == (K_D + C / 2) % C);

    /* Amplitude sanity: each Hann contributes a coherent gain of about 0.5
     * and each FFT's 1/length cancels its own coherent sum, so the peak
     * should land near A/4. A wide band still catches a lost or doubled
     * shift, which is what could plausibly go wrong. */
    CHECK_CLOSE(sqrt((double)best), A / 4.0, 0.6 * A / 4.0);

    /* ---- clutter removal actually notches zero Doppler ----
     *
     * A stationary return (no slow-time modulation) must be suppressed
     * relative to the moving target above: its energy sits at Doppler bin
     * 0, which the across-chirp mean removes. */
    for (int c = 0; c < C; c++) {
        for (int s = 0; s < S; s++) {
            double ang = 2.0 * FMCW_PI * (double)K_R * s / S;
            in[c * S + s].re = (int16_t)lround(A * cos(ang));
            in[c * S + s].im = (int16_t)lround(A * sin(ang));
        }
    }
    sat = 0;
    rd_q15_process_channel(&p, in, out, &sat);
    CHECK(sat == 0);

    int64_t worst = 0;
    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++)
            if (cq15_mag2(out[c * R + r]) > worst) worst = cq15_mag2(out[c * R + r]);
    /* Whatever survives is residue, far below the moving target's peak. */
    CHECK(worst < best / 100);

    rd_q15_plan_free(&p);
    CHECK_DONE();
}
