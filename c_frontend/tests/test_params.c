#include <stdio.h>
#include <string.h>
#include "params.h"
#include "check.h"

/* A minimal but complete params.cfg: all 16 required keys, tiny dimensions
 * (V=6=2x3, C=4, S=8) so expected values are easy to eyeball. */
static const char *GOOD =
    "n_virtual=6\n"
    "n_chirps=4\n"
    "n_samples=8\n"
    "n_tx=2\n"
    "n_rx=3\n"
    "doa_n_sources=1\n"
    "cfar_guard_d=2\n"
    "cfar_guard_r=4\n"
    "cfar_training_d=4\n"
    "cfar_training_r=8\n"
    "min_range_bins=6\n"
    "range_per_bin=0.5859375\n"
    "pfa=0.0001\n"
    "cfar_alpha=14.83\n"
    "velocity_axis=-2.0,-1.0,0.0,1.0\n"
    "pos_wl=0.0,0.5,1.0,1.5,2.0,2.5\n";

/* Write `text` to a scratch file the parser can open (tmp_* is gitignored). */
static void write_tmp(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    fputs(text, f);
    fclose(f);
}

int main(void)
{
    /* Happy path: every field lands in the struct with the right value. */
    FrontendParams p;
    write_tmp("tests/tmp_good.cfg", GOOD);
    CHECK(params_load("tests/tmp_good.cfg", &p) == 0);
    CHECK(p.n_virtual == 6 && p.n_chirps == 4 && p.n_samples == 8);
    CHECK(p.n_tx == 2 && p.n_rx == 3);
    CHECK(p.min_range_bins == 6);
    CHECK_CLOSE(p.range_per_bin, 0.5859375, 1e-12);
    CHECK_CLOSE(p.cfar_alpha, 14.83, 1e-12);
    CHECK(p.n_velocity == 4);
    CHECK_CLOSE(p.velocity_axis[0], -2.0, 1e-12);
    CHECK_CLOSE(p.velocity_axis[3], 1.0, 1e-12);
    CHECK(p.n_pos == 6);
    CHECK_CLOSE(p.pos_wl[5], 2.5, 1e-12);

    /* Missing key: truncate GOOD before its last line (pos_wl). */
    char buf[2048];
    snprintf(buf, sizeof buf, "%s", GOOD);
    char *last = strstr(buf, "pos_wl=");
    *last = '\0';
    write_tmp("tests/tmp_missing.cfg", buf);
    CHECK(params_load("tests/tmp_missing.cfg", &p) == -1);

    /* Malformed line (no '='). */
    snprintf(buf, sizeof buf, "%sgarbage_line\n", GOOD);
    write_tmp("tests/tmp_bad.cfg", buf);
    CHECK(params_load("tests/tmp_bad.cfg", &p) == -1);

    /* Unknown key: strict contract — typos must fail, not be skipped. */
    snprintf(buf, sizeof buf, "%snot_a_key=1\n", GOOD);
    write_tmp("tests/tmp_unknown.cfg", buf);
    CHECK(params_load("tests/tmp_unknown.cfg", &p) == -1);

    /* List length inconsistent with dims: rebuild GOOD with a 2-entry
     * velocity_axis (n_chirps says 4) by splicing around that line. */
    snprintf(buf, sizeof buf, "%s", GOOD);
    char *va = strstr(buf, "velocity_axis=");
    const char *rest = strstr(va, "\n");
    char tail[1024];
    snprintf(tail, sizeof tail, "%s", rest);
    snprintf(va, sizeof buf - (size_t)(va - buf), "velocity_axis=-2.0,-1.0%s", tail);
    write_tmp("tests/tmp_shortlist.cfg", buf);
    CHECK(params_load("tests/tmp_shortlist.cfg", &p) == -1);

    /* Nonexistent file. */
    CHECK(params_load("tests/tmp_nope.cfg", &p) == -1);
    CHECK_DONE();
}
