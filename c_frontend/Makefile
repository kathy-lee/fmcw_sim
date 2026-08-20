CC ?= gcc
CFLAGS  = -std=c11 -Wpedantic -Wall -Wextra -Werror -O2 -Iinclude
DBGFLAGS = -std=c11 -Wpedantic -Wall -Wextra -Werror -g -fsanitize=address,undefined -Iinclude
LDLIBS  = -lm

SRC = src/fft.c src/io.c
TESTS = tests/test_cplx tests/test_fft tests/test_params tests/test_io

all: fmcw_frontend

fmcw_frontend: src/main.c $(SRC) include/fmcw.h
	$(CC) $(CFLAGS) -o $@ src/main.c $(SRC) $(LDLIBS)

debug: src/main.c $(SRC) include/fmcw.h
	$(CC) $(DBGFLAGS) -o fmcw_frontend_debug src/main.c $(SRC) $(LDLIBS)

tests/%: tests/%.c $(SRC) include/fmcw.h tests/check.h
	$(CC) $(CFLAGS) -Itests -o $@ $< $(SRC) $(LDLIBS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

clean:
	rm -f fmcw_frontend fmcw_frontend_debug $(TESTS)

.PHONY: all debug test clean
