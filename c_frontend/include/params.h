#ifndef PARAMS_H
#define PARAMS_H

/* Parsed contents of params.cfg — the flat key=value contract written by
 * scripts/export_for_c.py. All 17 keys are required; params_load validates
 * presence, syntax, and dimensional consistency, so downstream code can
 * trust every field without re-checking. */

#define FMCW_MAX_LIST 512

typedef struct {
    int n_virtual, n_chirps, n_samples, n_tx, n_rx;
    int doa_n_sources;
    int cfar_guard_d, cfar_guard_r, cfar_training_d, cfar_training_r;
    int min_range_bins;
    double range_per_bin, pfa, cfar_alpha;
    /* The ADC's fixed input range, in the same units as the frame samples.
     * Only the fixed-point path reads it: it sets the quantizer's full
     * scale, so a signal beyond it clips rather than rescaling the grid. */
    double adc_full_scale;
    double velocity_axis[FMCW_MAX_LIST]; int n_velocity;
    double pos_wl[FMCW_MAX_LIST];        int n_pos;
} FrontendParams;

/* Parse `path` into *p. Returns 0 on success; -1 (with a message on stderr)
 * for: unreadable file, line without '=', unknown key, unparseable value,
 * missing required key, or inconsistent dimensions (n_virtual != n_tx*n_rx,
 * list lengths not matching n_chirps / n_virtual, non-positive dims). */
int params_load(const char *path, FrontendParams *p);
#endif
