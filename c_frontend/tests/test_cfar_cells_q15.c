#include <string.h>
#include "cfar_q15.h"
#include "cfar_cells_q15.h"
#include "check.h"

#define D 16
#define R 16
#define MAXC 64

static uint32_t rng = 987654321u;
static int64_t noise_cell(void)
{
    rng = rng * 1103515245u + 12345u;
    return 1000000 + (int64_t)((rng >> 16) % 100001u) - 50000;   /* 1e6 +/- 5% */
}

int main(void)
{
    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_chirps = D;
    p.n_samples = 2 * R;
    p.cfar_guard_d = 1;
    p.cfar_guard_r = 1;
    p.cfar_training_d = 2;
    p.cfar_training_r = 2;
    p.min_range_bins = 2;
    p.cfar_alpha = 4.0;

    static int64_t power[D * R];
    static uint8_t mask[D * R];
    static CfarCellQ15 cells[MAXC];
    for (int i = 0; i < D * R; i++) power[i] = noise_cell();
    power[8 * R + 8] = 1000000000;
    power[3 * R + 12] = 900000000;

    int overflow = -1;
    int n = cfar_cells_q15(power, &p, cells, MAXC, &overflow);
    cfar_q15_ca_mask_ref(power, &p, mask);

    /* ---- the cell set IS the reference mask's 1-set, in scan order ----
     *
     * The walk and the mask are the same detector with different output
     * representations, so this is an equality, not a tolerance. */
    int m = 0;
    for (int d = 0; d < D; d++) {
        for (int r = 0; r < R; r++) {
            if (!mask[d * R + r]) continue;
            CHECK(m < n);
            CHECK(cells[m].doppler_bin == d && cells[m].range_bin == r);
            CHECK(cells[m].power == power[d * R + r]);
            /* the stored threshold reproduces the reference decision */
            CHECK((cells[m].power << CFAR_Q15_ALPHA_SHIFT)
                  > cells[m].thresh_q16);
            m++;
        }
    }
    CHECK(m == n);
    CHECK(overflow == 0);
    CHECK(n == 2);

    /* min-range gate: no cell below the gate ever appears */
    for (int i = 0; i < n; i++) CHECK(cells[i].range_bin >= p.min_range_bins);

    /* ---- ranked cap ----
     *
     * alpha 0 fires every ungated cell (14 x 16 = 224 of them); a cap of 8
     * keeps the strongest by margin and reports the spill. With a zero
     * threshold the margin ordering is the power ordering. */
    p.cfar_alpha = 0.0;
    int n2 = cfar_cells_q15(power, &p, cells, 8, &overflow);
    CHECK(n2 == 8);
    CHECK(overflow == 1);

    /* kept cells stay in (d, r) scan order even after replacements */
    for (int i = 1; i < n2; i++)
        CHECK(cells[i - 1].doppler_bin < cells[i].doppler_bin ||
              (cells[i - 1].doppler_bin == cells[i].doppler_bin &&
               cells[i - 1].range_bin < cells[i].range_bin));

    /* every kept margin >= every discarded margin: against the 9th-largest
     * power among the ungated cells */
    static int64_t all[D * R];
    int na = 0;
    for (int d = 0; d < D; d++)
        for (int r = p.min_range_bins; r < R; r++)
            all[na++] = power[d * R + r];
    for (int i = 1; i < na; i++) {          /* insertion sort, descending */
        int64_t key = all[i];
        int j = i - 1;
        while (j >= 0 && all[j] < key) { all[j + 1] = all[j]; j--; }
        all[j + 1] = key;
    }
    int64_t kept_min = cells[0].power;
    for (int i = 1; i < n2; i++)
        if (cells[i].power < kept_min) kept_min = cells[i].power;
    CHECK(kept_min >= all[8]);

    CHECK_DONE();
}
