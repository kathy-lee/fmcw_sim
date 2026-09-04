#include <stdint.h>
#include <string.h>
#include "doa.h"
#include "doa_cell.h"
#include "check.h"

#define V 12

static uint32_t rng = 77777777u;
static float frand(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

int main(void)
{
    double pos_wl[V];
    for (int e = 0; e < V; e++) pos_wl[e] = 0.5 * e;   /* the TDM virtual array */

    DoaPlan ref = {0};
    CHECK(doa_plan_init(&ref, pos_wl, V) == 0);

    static double spectrum[181];
    cplx snap[V];
    double az;

    /* Random snapshots against the frozen precomputed-matrix path: the
     * tableless form runs the same libm calls with the same float casts in
     * the same order, so the spectra must agree to the LAST BIT, and the
     * selected grid index exactly — not approximately. */
    for (int trial = 0; trial < 50; trial++) {
        for (int e = 0; e < V; e++) snap[e] = (cplx){frand(), frand()};

        int k = doa_bartlett(&ref, snap, 1, &az);
        CHECK(k == 1);
        int idx = doa_bartlett_cell(snap, pos_wl, V, spectrum);

        CHECK(idx == (int)az + 90);
        /* the frozen plan keeps its last spectrum in ref.spectrum */
        CHECK(memcmp(spectrum, ref.spectrum, sizeof spectrum) == 0);
    }

    doa_plan_free(&ref);
    CHECK_DONE();
}
