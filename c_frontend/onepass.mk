# Build additions for the one-pass streaming front end.
#
# The frozen kernel set lives directly under src/ and include/; the
# one-pass additions live under src/onepass/ and include/onepass/, and
# every build variable they need is appended here — the main Makefile
# gains only the include of this file. The sources join the same SRC list
# because they belong in the same library and link into every test.

CFLAGS += -Iinclude/onepass
DBGFLAGS += -Iinclude/onepass
HDRS += $(wildcard include/onepass/*.h)

SRC += src/onepass/cfar_cells_q15.c
TESTS += tests/test_cfar_cells_q15
