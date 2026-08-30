# Build, flash and capture for the NUCLEO-F446RE FFT harness.
#
# Makefile.cubemx is what STM32CubeMX generated and is never edited, so the
# project can be regenerated from the .ioc without losing anything here.
# It is included, and the harness's additions are appended after it: the
# synced kernel sources, the arithmetic define, and -O2 in place of
# CubeMX's -Og. Every variable touched is one CubeMX expands lazily.
#
# The kernel sources are copied in from c_frontend before every build,
# unconditionally, so the copies cannot go stale.

ARITH   ?= q15
PORT    ?= /dev/ttyACM0
PYTHON  ?= python
BAUD     = 115200
KERNEL   = ../../c_frontend
BRINGUP ?= 0

ifeq ($(ARITH),q15)
SYNC_SRC = $(KERNEL)/src/fft_q15.c
SYNC_HDR = $(KERNEL)/include/fft_q15.h $(KERNEL)/include/fmcw_q15.h $(KERNEL)/include/fmcw.h
DEFINE   = -DARITH_Q15=1
STALE    = Core/Src/fft.c Core/Inc/fft.h
else
SYNC_SRC = $(KERNEL)/src/fft.c
SYNC_HDR = $(KERNEL)/include/fft.h $(KERNEL)/include/fmcw.h
DEFINE   = -DARITH_FLOAT=1
STALE    = Core/Src/fft_q15.c Core/Inc/fft_q15.h Core/Inc/fmcw_q15.h
endif

# Each arithmetic builds into its own directory: the two builds share
# object names compiled under different defines, and a stale harness.o
# from the other one would link silently. override wins over the plain
# assignment inside Makefile.cubemx.
override BUILD_DIR = build/$(ARITH)

include Makefile.cubemx

KERNEL_C   = $(patsubst $(KERNEL)/src/%,Core/Src/%,$(SYNC_SRC))
C_SOURCES += $(KERNEL_C) Core/Src/harness.c Core/Src/retarget.c
C_DEFS    += $(DEFINE) -DBRINGUP=$(BRINGUP)
C_INCLUDES += -I.
OPT        = -O2
# newlib-nano leaves float formatting out of printf unless asked for it.
LDFLAGS   += -u _printf_float

ELF = $(BUILD_DIR)/$(TARGET).elf

# Makefile.cubemx expands the ELF's prerequisite list at the point it is
# read, before the sources above were appended; the extra objects would
# be linked but never compiled. Restating the dependency here, after the
# append, adds them.
$(ELF): $(OBJECTS)

.DEFAULT_GOAL := help
.PHONY: help sync vectors host check-host build flash capture report distclean

help:
	@echo "targets: sync vectors host check-host build flash capture report distclean"
	@echo "         ARITH=q15|float (default q15), BRINGUP=1 for the configuration-only image"

# build/ is created here because Makefile.cubemx makes its own directory
# with a plain mkdir, which fails when the parent does not exist.
sync:
	@mkdir -p build
	@rm -f $(STALE) ref_host
	@cp $(SYNC_SRC) Core/Src/
	@cp $(SYNC_HDR) Core/Inc/
	@echo "synced: $(notdir $(SYNC_SRC) $(SYNC_HDR))"

vectors:
	$(PYTHON) gen_vectors.py --n 256 --self-check

# The host reference lives in host/ so the layout matches the Uno harness.
host: sync
	gcc -std=c11 -Wall -Wextra -Wpedantic -O2 -I. -ICore/Inc $(DEFINE) \
	    -o ref_host host/ref_host.c $(KERNEL_C) -lm

# The constant vector transforms to exactly 1000 in bin 0 and exactly zero
# elsewhere. That pins the per-stage halvings to exactly 1/n and catches a
# bad sync or a corrupt vectors.h before anything is flashed.
check-host: host
	@./ref_host > build/stream_$(ARITH)_host.txt || exit 1
	@awk '/^VEC dc$$/{f=1;next} /^VEC /{f=0} f&&/^OUT/{ \
	    if ($$2==0) { if ($$3!=1000 || $$4!=0) { print "FAIL bin0", $$3, $$4; exit 1 } } \
	    else if ($$3!=0 || $$4!=0) { print "FAIL bin", $$2, $$3, $$4; exit 1 } } \
	    END{print "check-host passed"}' build/stream_$(ARITH)_host.txt

# The ELF is built by a second make so that sync has finished copying
# before Makefile.cubemx's rules look at the sources.
build: sync
	@$(MAKE) --no-print-directory ARITH=$(ARITH) BRINGUP=$(BRINGUP) $(ELF)
	arm-none-eabi-size $(ELF) | tee build/size_$(ARITH).txt

flash: build
	openocd -f board/st_nucleo_f4.cfg -c "program $(ELF) verify reset exit"

# Reading the port with stty and cat. The board reports every few seconds,
# so the window only has to be long enough to contain one whole run.
capture:
	@mkdir -p build
	@stty -F $(PORT) $(BAUD) raw -echo
	-timeout 20 cat $(PORT) > build/stream_$(ARITH)_nucleo.txt
	@grep -q '^END' build/stream_$(ARITH)_nucleo.txt \
	    && echo "captured build/stream_$(ARITH)_nucleo.txt" \
	    || (echo "stream incomplete — no END line"; exit 1)

report:
	$(PYTHON) read_results.py --build-dir build

# Makefile.cubemx already defines clean (its own build dir); this removes
# everything the harness adds as well.
distclean:
	rm -rf build ref_host \
	    Core/Src/fft.c Core/Src/fft_q15.c \
	    Core/Inc/fft.h Core/Inc/fft_q15.h Core/Inc/fmcw.h Core/Inc/fmcw_q15.h
