#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "doa_cell.h"
#include "doa_cell_f32.h"
#include "check.h"

#define V 12
#define N_GRID 181
#define SPACING_WL 0.5

static uint32_t rng = 77777777u;
static float frand(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

static cplx w_grid[N_GRID];
static double pos_wl[V];
static float spectrum[N_GRID];

/* The table as the board's parameter generator writes it: computed in
 * double, rounded to float. */
static void make_table(void)
{
    for (int g = 0; g < N_GRID; g++) {
        double ph = 2.0 * FMCW_PI * SPACING_WL
                    * sin((g - 90) * FMCW_PI / 180.0);
        w_grid[g] = (cplx){ (float)cos(ph), (float)sin(ph) };
    }
    for (int e = 0; e < V; e++) pos_wl[e] = SPACING_WL * e;
}

static void test_steering_snapshot_found(void)
{
    cplx snap[V];
    for (int idx = 30; idx <= 150; idx++) {
        double s = sin((idx - 90) * FMCW_PI / 180.0);
        for (int e = 0; e < V; e++) {
            double ph = 2.0 * FMCW_PI * pos_wl[e] * s;
            snap[e] = (cplx){ (float)cos(ph), (float)sin(ph) };
        }
        CHECK(doa_bartlett_cell_f32(snap, w_grid, V, spectrum) == idx);
    }
}

static void test_close_to_double_path(void)
{
    static double spectrum_f64[N_GRID];
    cplx snap[V];
    int differ = 0;
    for (int trial = 0; trial < 1000; trial++) {
        for (int e = 0; e < V; e++) snap[e] = (cplx){frand(), frand()};
        int ref = doa_bartlett_cell(snap, pos_wl, V, spectrum_f64);
        int got = doa_bartlett_cell_f32(snap, w_grid, V, spectrum);
        CHECK(got >= ref - 1 && got <= ref + 1);
        if (got != ref) differ++;
    }
    printf("doa f32 vs f64: %d of 1000 indices differ\n", differ);
}

static void test_flat_spectrum_argmax_fallback(void)
{
    cplx snap[V] = {0};
    CHECK(doa_bartlett_cell_f32(snap, w_grid, V, spectrum) == 0);
}

int main(void)
{
    make_table();
    test_steering_snapshot_found();
    test_close_to_double_path();
    test_flat_spectrum_argmax_fallback();
    CHECK_DONE();
}
