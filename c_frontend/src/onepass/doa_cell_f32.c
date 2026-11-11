#include "doa_cell_f32.h"

#define DOA_N_GRID 181

int doa_bartlett_cell_f32(const cplx *snap, const cplx *w_grid, int n_elem,
                          float *spectrum)
{
    for (int g = 0; g < DOA_N_GRID; g++) {
        cplx w = w_grid[g];
        cplx a = { 1.0f, 0.0f };             /* w^e, starting at e = 0 */
        float re = 0.0f, im = 0.0f;
        for (int e = 0; e < n_elem; e++) {   /* conj(a) * snap[e] */
            re += a.re * snap[e].re + a.im * snap[e].im;
            im += a.re * snap[e].im - a.im * snap[e].re;
            a = cmul(a, w);
        }
        spectrum[g] = re * re + im * im;
    }

    /* spectral-maxima selection: strict local maxima, one-sided edge rule,
     * then the strongest; argmax fallback if the spectrum is flat */
    int best = -1;
    for (int i = 0; i < DOA_N_GRID; i++) {
        int left = (i == 0) || (spectrum[i] > spectrum[i - 1]);
        int right = (i == DOA_N_GRID - 1) || (spectrum[i] > spectrum[i + 1]);
        if (left && right && (best < 0 || spectrum[i] > spectrum[best]))
            best = i;
    }
    if (best < 0) {
        best = 0;
        for (int i = 1; i < DOA_N_GRID; i++)
            if (spectrum[i] > spectrum[best]) best = i;
    }
    return best;
}
