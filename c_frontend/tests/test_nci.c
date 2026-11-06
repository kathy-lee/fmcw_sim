#include "fmcw.h"
#include "nci.h"
#include "check.h"

int main(void)
{
    /* Golden check vs Python integrate(): a 3x4x6 stack whose per-cell
     * power sum was computed in float64. C accumulates the same float32
     * inputs in double, in the same v-order, so agreement is essentially
     * bit-exact — 1e-12 relative is a formality. */
    enum { V = 3, C = 4, R = 6, N = C * R };
    cplx in[V * N];
    double ref[N], out[N];
    CHECK(load_bin("tests/vectors/nci_in.bin", in, sizeof in) == 0);
    CHECK(load_bin("tests/vectors/nci_out.bin", ref, sizeof ref) == 0);

    nci_integrate(in, V, N, out);

    double maxref = 0.0;
    for (int i = 0; i < N; i++) if (ref[i] > maxref) maxref = ref[i];
    for (int i = 0; i < N; i++) CHECK_CLOSE(out[i], ref[i], 1e-12 * maxref);
    CHECK_DONE();
}
