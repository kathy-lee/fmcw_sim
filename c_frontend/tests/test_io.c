#include <stdio.h>
#include <string.h>
#include "params.h"
#include "io.h"
#include "check.h"

int main(void)
{
    /* Round-trip a tiny frame file: what fwrite wrote, load_frame must read
     * back bit-identically (the binary contract the exporter relies on). */
    cplx w[6] = {{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}};
    FILE *f = fopen("tests/tmp_frame.bin", "wb");
    CHECK(f != NULL);
    CHECK(fwrite(w, sizeof(cplx), 6, f) == 6);
    fclose(f);

    cplx r[6];
    CHECK(load_frame("tests/tmp_frame.bin", r, 6) == 0);
    CHECK(memcmp(w, r, sizeof w) == 0);

    /* Wrong expected count -> error, both directions (file too big for 5,
     * too small for 7): silent truncation would corrupt every later stage. */
    CHECK(load_frame("tests/tmp_frame.bin", r, 5) == -1);
    CHECK(load_frame("tests/tmp_frame.bin", r, 7) == -1);
    CHECK(load_frame("tests/tmp_nofile.bin", r, 6) == -1);

    /* Buffer sizing from params: V=6, C=4, S=8 -> adc 192, rd 96 samples,
     * 16 cells for the power map / mask / worst-case detection list. */
    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_virtual = 6; p.n_chirps = 4; p.n_samples = 8;
    FrontendBuffers b;
    CHECK(buffers_alloc(&b, &p) == 0);
    CHECK(b.adc_n == 6u * 4u * 8u);
    CHECK(b.rd_n == 6u * 4u * 4u);
    CHECK(b.cells_n == 4u * 4u);
    CHECK(b.adc != NULL && b.rd != NULL);
    CHECK(b.power != NULL && b.mask != NULL && b.det != NULL);
    print_memory_summary(&b, stdout);   /* eyeball in test output */
    buffers_free(&b);
    CHECK(b.adc == NULL && b.rd == NULL);
    CHECK(b.power == NULL && b.mask == NULL && b.det == NULL);

    /* CSV: header + one row, byte-exact — this string IS the file format
     * the pytest harness parses. */
    Detection d1 = {5, 3, 2.0, 1.0, 0.5, -0.25, 12.5};
    f = fopen("tests/tmp_det.csv", "w");
    CHECK(f != NULL);
    CHECK(csv_write_header(f) == 0);
    CHECK(csv_write_detections(f, 0, &d1, 1) == 0);
    fclose(f);
    char csv[256] = {0};
    f = fopen("tests/tmp_det.csv", "r");
    CHECK(f != NULL);
    size_t got = fread(csv, 1, sizeof csv - 1, f);
    fclose(f);
    CHECK(got > 0);
    CHECK(strcmp(csv,
        "frame,x,y,v_radial,snr_db,range_bin,doppler_bin,azimuth_deg\n"
        "0,1,0.5,-0.25,12.5,5,3,2\n") == 0);

    /* dump_stage round-trip via load_bin (strict-size read). */
    double blob[4] = {1.0, 2.0, 3.0, 4.0}, back[4];
    CHECK(dump_stage("tests", "tmp_dump", 7, blob, sizeof blob) == 0);
    CHECK(load_bin("tests/tmp_dump_0007.bin", back, sizeof back) == 0);
    CHECK(memcmp(blob, back, sizeof blob) == 0);
    CHECK_DONE();
}
