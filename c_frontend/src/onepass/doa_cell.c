/* The steering element the frozen path reads from its table is
 * ((float)cos(ph), (float)sin(ph)); computing that value here at the point
 * of use, with the identical expression, is what makes the spectra
 * bit-identical without any table — the table was only ever a cache. Cost
 * moves from 181 x V x 8 bytes of RAM to 181 x V sin/cos pairs per call,
 * the right trade where detections are few and RAM is the wall. */
#include <math.h>
#include "doa_cell.h"

#define DOA_N_GRID 181

int doa_bartlett_cell(const cplx *snap, const double *pos_wl, int n_elem,
                      double *spectrum)
{
    for (int g = 0; g < DOA_N_GRID; g++) {
        double s = sin((g - 90) * FMCW_PI / 180.0);
        double re = 0.0, im = 0.0;
        for (int e = 0; e < n_elem; e++) {   /* conj(a[e]) * snap[e] */
            double ph = 2.0 * FMCW_PI * pos_wl[e] * s;
            cplx a = { (float)cos(ph), (float)sin(ph) };
            re += (double)a.re * snap[e].re + (double)a.im * snap[e].im;
            im += (double)a.re * snap[e].im - (double)a.im * snap[e].re;
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
