#include "fmcw.h"
#include "rd.h"
#include "check.h"

int main(void)
{
    /* Hann windows must match np.hanning to double precision — the window
     * shapes every RD value, so an off-by-one in the denominator (N vs N-1,
     * the symmetric-vs-periodic trap) would corrupt everything downstream. */
    double w64[64], w256[256], ref64[64], ref256[256];
    hann_fill(w64, 64);
    hann_fill(w256, 256);
    CHECK(load_bin("tests/vectors/hann64.bin", ref64, sizeof ref64) == 0);
    CHECK(load_bin("tests/vectors/hann256.bin", ref256, sizeof ref256) == 0);
    for (int i = 0; i < 64; i++) CHECK_CLOSE(w64[i], ref64[i], 1e-12);
    for (int i = 0; i < 256; i++) CHECK_CLOSE(w256[i], ref256[i], 1e-12);
    CHECK_CLOSE(w64[0], 0.0, 1e-15);     /* symmetric window: zero endpoints */
    CHECK_CLOSE(w64[63], 0.0, 1e-15);

    /* Full RD chain vs the Python range_doppler() golden vector: an 8-chirp
     * x 16-sample frame in, 8x8 complex RD map out. This is the end-to-end
     * oracle for DC removal -> window -> range FFT -> clutter removal ->
     * window -> Doppler FFT -> fftshift, all in one check. */
    enum { C = 8, S = 16, R = S / 2 };
    cplx in[C * S], ref[C * R], out[C * R];
    CHECK(load_bin("tests/vectors/rd_in.bin", in, sizeof in) == 0);
    CHECK(load_bin("tests/vectors/rd_out.bin", ref, sizeof ref) == 0);

    RdPlan plan;
    CHECK(rd_plan_init(&plan, C, S) == 0);
    rd_process_channel(&plan, in, out);

    double maxmag = 0.0, maxerr = 0.0;
    for (int i = 0; i < C * R; i++) {
        double m = sqrt(cmag2(ref[i]));
        if (m > maxmag) maxmag = m;
        double dre = (double)out[i].re - ref[i].re;
        double dim = (double)out[i].im - ref[i].im;
        double e = sqrt(dre * dre + dim * dim);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr <= 1e-4 * maxmag);
    rd_plan_free(&plan);
    CHECK_DONE();
}
