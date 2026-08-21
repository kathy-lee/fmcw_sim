/* Bartlett DoA. The spectrum accumulates conj(a)*x in double from float32
 * factors — same effective precision as NumPy's complex128 dot on our
 * float32 data, so 1e-5-relative agreement is expected, and peak indices
 * (grid cells) agree exactly on non-degenerate spectra. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "doa.h"

#define DOA_N_GRID 181

int doa_plan_init(DoaPlan *p, const double *pos_wl, int n_elem)
{
    memset(p, 0, sizeof *p);
    p->n_grid = DOA_N_GRID;
    p->n_elem = n_elem;
    p->steer = malloc((size_t)DOA_N_GRID * n_elem * sizeof(cplx));
    p->spectrum = malloc(DOA_N_GRID * sizeof(double));
    if (!p->steer || !p->spectrum) { doa_plan_free(p); return -1; }
    for (int g = 0; g < DOA_N_GRID; g++) {
        double s = sin((g - 90) * FMCW_PI / 180.0);
        for (int e = 0; e < n_elem; e++) {
            double ph = 2.0 * FMCW_PI * pos_wl[e] * s;
            p->steer[(size_t)g * n_elem + e] = (cplx){(float)cos(ph), (float)sin(ph)};
        }
    }
    return 0;
}

void doa_plan_free(DoaPlan *p)
{
    free(p->steer);
    free(p->spectrum);
    p->steer = NULL;
    p->spectrum = NULL;
}

int doa_bartlett(DoaPlan *p, const cplx *snap, int n_sources, double *az_deg)
{
    int n = p->n_grid;
    double *s = p->spectrum;
    for (int g = 0; g < n; g++) {
        const cplx *a = p->steer + (size_t)g * p->n_elem;
        double re = 0.0, im = 0.0;
        for (int e = 0; e < p->n_elem; e++) {   /* conj(a[e]) * snap[e] */
            re += (double)a[e].re * snap[e].re + (double)a[e].im * snap[e].im;
            im += (double)a[e].re * snap[e].im - (double)a[e].im * snap[e].re;
        }
        s[g] = re * re + im * im;
    }

    /* _peaks: strict local maxima with the one-sided edge rule. */
    int idx[DOA_N_GRID], m = 0;
    for (int i = 0; i < n; i++) {
        int left = (i == 0) || (s[i] > s[i - 1]);
        int right = (i == n - 1) || (s[i] > s[i + 1]);
        if (left && right) idx[m++] = i;
    }
    if (m == 0) {                       /* flat spectrum: argmax fallback */
        int best = 0;
        for (int i = 1; i < n; i++) if (s[i] > s[best]) best = i;
        idx[0] = best; m = 1;
    }

    int k = n_sources < m ? n_sources : m;
    for (int a = 0; a < k; a++) {       /* top-k by power (selection) */
        int best = a;
        for (int b = a + 1; b < m; b++)
            if (s[idx[b]] > s[idx[best]]) best = b;
        int t = idx[a]; idx[a] = idx[best]; idx[best] = t;
    }
    for (int a = 1; a < k; a++) {       /* chosen k: ascending by angle */
        int key = idx[a], b = a - 1;
        while (b >= 0 && idx[b] > key) { idx[b + 1] = idx[b]; b--; }
        idx[b + 1] = key;
    }
    for (int a = 0; a < k; a++) az_deg[a] = (double)(idx[a] - 90);
    return k;
}
