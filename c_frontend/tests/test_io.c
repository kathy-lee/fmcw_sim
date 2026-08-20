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

    /* Buffer sizing from params: V=6, C=4, S=8 -> adc 192, rd 96 samples. */
    FrontendParams p;
    memset(&p, 0, sizeof p);
    p.n_virtual = 6; p.n_chirps = 4; p.n_samples = 8;
    FrontendBuffers b;
    CHECK(buffers_alloc(&b, &p) == 0);
    CHECK(b.adc_n == 6u * 4u * 8u);
    CHECK(b.rd_n == 6u * 4u * 4u);
    CHECK(b.adc != NULL && b.rd != NULL);
    print_memory_summary(&b, stdout);   /* eyeball in test output */
    buffers_free(&b);
    CHECK(b.adc == NULL && b.rd == NULL);
    CHECK_DONE();
}
