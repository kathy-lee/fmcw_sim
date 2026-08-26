# Build, flash and capture for the Uno FFT harness.
#
# The kernel sources are copied in from c_frontend before every build: the
# Arduino toolchain compiles only what is inside the sketch folder, and it
# will not reach across the tree. Copying rather than symlinking keeps the
# behaviour identical on any filesystem, and syncing unconditionally means
# the copies cannot go stale.

ARITH   ?= q15
PORT    ?= /dev/ttyACM0
FQBN     = arduino:avr:uno
BAUD     = 115200
KERNEL   = ../../c_frontend

ifeq ($(ARITH),q15)
SYNC_SRC = $(KERNEL)/src/fft_q15.c
SYNC_HDR = $(KERNEL)/include/fft_q15.h $(KERNEL)/include/fmcw_q15.h $(KERNEL)/include/fmcw.h
DEFINE   = -DARITH_Q15=1
STALE    = fft.c fft.h
else
SYNC_SRC = $(KERNEL)/src/fft.c
SYNC_HDR = $(KERNEL)/include/fft.h $(KERNEL)/include/fmcw.h
DEFINE   = -DARITH_FLOAT=1
STALE    = fft_q15.c fft_q15.h fmcw_q15.h
endif

EXTRA    = -O2 $(DEFINE)
PROPS    = --build-property "compiler.c.extra_flags=$(EXTRA)" \
           --build-property "compiler.cpp.extra_flags=$(EXTRA)"

.PHONY: sync vectors host check-host build flash capture report ports clean

build/.dir:
	@mkdir -p build && touch build/.dir

# Only the selected arithmetic's sources are present, so the other one is
# neither compiled into the image nor able to shadow an include.
sync:
	@rm -f $(STALE)
	@cp $(SYNC_SRC) $(SYNC_HDR) .
	@echo "synced: $(notdir $(SYNC_SRC) $(SYNC_HDR))"

vectors:
	python gen_vectors.py --self-check

# The host reference lives in host/ rather than beside the sketch: the
# Arduino toolchain compiles every source in the sketch root and in src/,
# and would try to build this laptop-only file for the AVR.
host: sync
	gcc -std=c11 -Wall -Wextra -Wpedantic -O2 -I. $(DEFINE) \
	    -o ref_host host/ref_host.c $(notdir $(SYNC_SRC)) -lm

# The constant vector transforms to exactly 1000 in bin 0 and exactly zero
# elsewhere. That pins the per-stage halvings to exactly 1/n and catches a
# bad sync or a corrupt vectors.h before anything is flashed.
check-host: host build/.dir
	@./ref_host > build/stream_$(ARITH)_host.txt || exit 1
	@awk '/^VEC dc$$/{f=1;next} /^VEC /{f=0} f&&/^OUT/{ \
	    if ($$2==0) { if ($$3!=1000 || $$4!=0) { print "FAIL bin0", $$3, $$4; exit 1 } } \
	    else if ($$3!=0 || $$4!=0) { print "FAIL bin", $$2, $$3, $$4; exit 1 } } \
	    END{print "check-host passed"}' build/stream_$(ARITH)_host.txt

build: sync build/.dir
	arduino-cli compile --fqbn $(FQBN) $(PROPS) --output-dir build . \
	    2>&1 | tee build/size_$(ARITH).txt

flash: build
	arduino-cli upload -p $(PORT) --fqbn $(FQBN) --input-dir build .

# Reading the port with cat rather than through arduino-cli monitor, which
# emits nothing when its output is redirected to a file. The sketch reports
# every few seconds, so the window only has to be long enough to contain one
# whole run; whether opening the port resets the board is not relied on.
capture: build/.dir
	@stty -F $(PORT) $(BAUD) raw -echo
	-timeout 20 cat $(PORT) > build/stream_$(ARITH)_uno.txt
	@grep -q '^END' build/stream_$(ARITH)_uno.txt \
	    && echo "captured build/stream_$(ARITH)_uno.txt" \
	    || (echo "stream incomplete — no END line"; exit 1)

report:
	python read_results.py --build-dir build

ports:
	arduino-cli board list

clean:
	rm -f ref_host fft.c fft.h fft_q15.c fft_q15.h fmcw_q15.h fmcw.h
	rm -rf build
