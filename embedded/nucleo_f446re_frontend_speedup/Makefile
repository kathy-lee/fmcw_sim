# Build, flash and capture for the NUCLEO-F446RE one-pass front end.
#
# Makefile.cubemx is what STM32CubeMX generated and is never edited. The
# kernel sources are copied in from c_frontend before every build,
# unconditionally, so the copies cannot go stale; the parameters header is
# rendered from the exported params.cfg the same way.

PORT   ?= /dev/ttyACM0
PYTHON ?= python
BAUD    = 921600
KERNEL  = ../../c_frontend
CFG    ?= ../../data/c_export_stream128/params.cfg

SYNC_SRC = $(KERNEL)/src/fft_q15.c $(KERNEL)/src/rd_q15.c \
           $(KERNEL)/src/onepass/rd_stream_q15.c $(KERNEL)/src/onepass/cfar_cells_q15.c \
           $(KERNEL)/src/onepass/doa_cell.c $(KERNEL)/src/onepass/onepass_q15.c \
           $(KERNEL)/src/pointcloud.c $(KERNEL)/src/doa.c \
           $(KERNEL)/src/convert_q15.c
SYNC_HDR = $(KERNEL)/include/fmcw.h $(KERNEL)/include/fmcw_q15.h \
           $(KERNEL)/include/fft_q15.h $(KERNEL)/include/params.h \
           $(KERNEL)/include/rd_q15.h $(KERNEL)/include/onepass/rd_stream_q15.h \
           $(KERNEL)/include/cfar_q15.h $(KERNEL)/include/onepass/cfar_cells_q15.h \
           $(KERNEL)/include/doa.h $(KERNEL)/include/onepass/doa_cell.h \
           $(KERNEL)/include/pointcloud.h $(KERNEL)/include/convert_q15.h \
           $(KERNEL)/include/onepass/onepass_q15.h

# The output directory must not be named plain "build": the phony build
# target would shadow it in the object rules' order-only prerequisite and
# make would recurse forever. Namespace it the way the harness does.
override BUILD_DIR = build/q15

include Makefile.cubemx

# notdir: the copies land flat in Core/Src whatever subdirectory of the
# kernel tree they came from (frozen kernels in src/, additions in
# src/onepass/)
KERNEL_C   = $(addprefix Core/Src/,$(notdir $(SYNC_SRC)))
C_SOURCES += $(KERNEL_C) Core/Src/frontend.c Core/Src/retarget.c
C_INCLUDES += -I.
OPT        = -O2
LDFLAGS   += -u _printf_float

ELF = $(BUILD_DIR)/$(TARGET).elf

$(ELF): $(OBJECTS)

.DEFAULT_GOAL := help
.PHONY: help sync build flash map distclean

help:
	@echo "targets: sync build flash map distclean   (CFG=path/to/params.cfg)"

sync:
	@mkdir -p $(BUILD_DIR)
	@cp $(SYNC_SRC) Core/Src/
	@cp $(SYNC_HDR) Core/Inc/
	@$(PYTHON) gen_frontend_params.py --cfg $(CFG) --out Core/Inc/frontend_params.h
	@echo "synced: $(notdir $(SYNC_SRC) $(SYNC_HDR)) + frontend_params.h"

build: sync
	@$(MAKE) --no-print-directory $(ELF)
	arm-none-eabi-size $(ELF) | tee build/size.txt

flash: build
	openocd -f board/st_nucleo_f4.cfg -c "program $(ELF) verify reset exit"

# The static regions, largest first, against the linker map.
map: build
	@grep -E "^\.(bss|data)|planes|shared|cells_|rx_buf|_user_heap_stack" \
	    $(BUILD_DIR)/$(TARGET).map | head -40
	@arm-none-eabi-nm --size-sort --radix=d $(ELF) | tail -15

distclean:
	rm -rf build Core/Inc/frontend_params.h \
	    $(KERNEL_C) $(addprefix Core/Inc/,$(notdir $(SYNC_HDR)))
