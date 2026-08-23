CC ?= gcc
CFLAGS  = -std=c11 -Wpedantic -Wall -Wextra -Werror -O2 -Iinclude
DBGFLAGS = -std=c11 -Wpedantic -Wall -Wextra -Werror -g -fsanitize=address,undefined -Iinclude
LDLIBS  = -lm

SRC = src/fft.c src/io.c src/rd.c src/nci.c src/cfar.c src/doa.c src/pointcloud.c src/fmcw_api.c src/convert_q15.c src/fft_q15.c src/rd_q15.c src/nci_q15.c src/cfar_q15.c src/pointcloud_q15.c
TESTS = tests/test_cplx tests/test_q15 tests/test_convert_q15 tests/test_fft tests/test_fft_q15 tests/test_params tests/test_io tests/test_rd tests/test_rd_q15 tests/test_nci tests/test_nci_q15 tests/test_cfar tests/test_cfar_q15 tests/test_doa tests/test_pointcloud tests/test_api

# Every header, not just fmcw.h: much of the fixed-point path is inline in
# its headers, so listing one by name would let a change to the hot code
# leave the binaries silently stale.
HDRS = $(wildcard include/*.h)

all: fmcw_frontend

fmcw_frontend: src/main.c $(SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ src/main.c $(SRC) $(LDLIBS)

debug: src/main.c $(SRC) $(HDRS)
	$(CC) $(DBGFLAGS) -o fmcw_frontend_debug src/main.c $(SRC) $(LDLIBS)

lib: $(SRC) $(HDRS)
	$(CC) $(CFLAGS) -fPIC -shared -o libfmcw.so $(SRC) $(LDLIBS)

tests/%: tests/%.c $(SRC) $(HDRS) tests/check.h
	$(CC) $(CFLAGS) -Itests -o $@ $< $(SRC) $(LDLIBS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

clean:
	rm -f fmcw_frontend fmcw_frontend_debug libfmcw.so $(TESTS)

.PHONY: all debug test lib clean
