#include "nci_q15.h"
#include "check.h"

#define V 3
#define CELLS 4

int main(void)
{
    /* Hand-computable: three channels, four cells. */
    cplx_q15 rd[V * CELLS] = {
        /* channel 0 */ {3, 4},   {0, 0},      {-1, 0},   {100, -100},
        /* channel 1 */ {0, 5},   {-2, -2},    {1, 1},    {0, 0},
        /* channel 2 */ {1, 0},   {32767, 0},  {0, -3},   {-32768, -32768},
    };
    int64_t power[CELLS];

    nci_q15_integrate(rd, V, CELLS, power);

    /* Every one of these is exact: integer multiply-add does not round, so
     * this stage costs the fixed-point port precisely nothing. */
    CHECK(power[0] == (9 + 16) + (0 + 25) + (1 + 0));
    CHECK(power[1] == 0 + (4 + 4) + (32767LL * 32767));
    CHECK(power[2] == 1 + (1 + 1) + 9);

    /* The extreme cell: two channels contribute nothing, the third holds a
     * value whose squared magnitude alone is 2^31 — past int32, which is
     * why the accumulator is int64. */
    CHECK(power[3] == (100LL * 100 + 100LL * 100) + 0 + 2147483648LL);

    /* Zero channels leave a zeroed map rather than stale contents. */
    for (int i = 0; i < CELLS; i++) power[i] = 12345;
    nci_q15_integrate(rd, 0, CELLS, power);
    for (int i = 0; i < CELLS; i++) CHECK(power[i] == 0);

    CHECK_DONE();
}
