#include <stdio.h>
#include "fmcw.h"

int main(void)
{
    fprintf(stderr,
            "usage: fmcw_frontend --config params.cfg --frames <dir> [--dump <dir>]\n"
            "(pipeline not wired yet)\n");
    return 1;
}
