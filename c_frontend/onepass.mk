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

SRC += src/onepass/cfar_cells_q15.c src/onepass/rd_stream_q15.c \
       src/onepass/doa_cell.c src/onepass/onepass_q15.c \
       src/onepass/fmcw_onepass_api.c src/onepass/onepass_stages_q15.c \
       src/onepass/doa_cell_f32.c
TESTS += tests/test_cfar_cells_q15 tests/test_rd_stream_q15 \
         tests/test_doa_cell tests/test_onepass_q15 \
         tests/test_onepass_stages_q15 tests/test_doa_cell_f32

# The same library on the CMSIS-DSP FFT: the laptop reference for a board
# built with FFT=cmsis. fft_q15.c gives way to the file the board compiles
# (fft_q15_cmsis.c) and to CMSIS-DSP's FFT in the version the board runs
# (src/onepass/cfft_q15_board_path.c). libfmcw.so is not affected.
#
# CMSIS-DSP is not written for -Wpedantic -Werror, so this target has its
# own flags. __GNUC_PYTHON__ is CMSIS-DSP's switch for a compiler without
# the CMSIS-Core headers. -Bsymbolic keeps every call inside this library
# when both libraries are loaded into one process. -ffp-contract=off keeps
# the float steps as written, as -std=c11 does for libfmcw.so: gnu11 would
# otherwise let a compiler with fused multiply-add merge them.
CMSIS_DSP   = ../embedded/third_party/CMSIS-DSP
CMSIS_TF    = $(CMSIS_DSP)/Source/TransformFunctions
CMSIS_SRC   = ../embedded/nucleo_f446re_frontend_speedup/Core/Src/fft_q15_cmsis.c \
              src/onepass/cfft_q15_board_path.c \
              $(CMSIS_TF)/arm_bitreversal.c $(CMSIS_TF)/arm_bitreversal2.c \
              $(CMSIS_DSP)/Source/CommonTables/arm_const_structs.c \
              $(CMSIS_DSP)/Source/CommonTables/arm_common_tables.c
CMSIS_FLAGS = -std=gnu11 -Wall -O2 -ffp-contract=off -Iinclude -Iinclude/onepass \
              -I$(CMSIS_DSP)/Include -I$(CMSIS_DSP)/PrivateInclude \
              -I$(CMSIS_TF) -D__GNUC_PYTHON__ -DARM_MATH_LOOPUNROLL

lib-cmsis: libfmcw_cmsis.so

libfmcw_cmsis.so: $(SRC) $(CMSIS_SRC) $(HDRS)
	$(CC) $(CMSIS_FLAGS) -fPIC -shared -Wl,-Bsymbolic -Wl,--no-undefined \
	    -o $@ $(filter-out src/fft_q15.c,$(SRC)) $(CMSIS_SRC) $(LDLIBS)

# The board's FFT file with its block scaling, tested on the laptop, and a
# check that the FFT underneath is the board's version: the test also links
# CMSIS-DSP's plain C version under other names (tests/cfft_q15_plain_path.c).
CMSIS_TEST = tests/test_fft_q15_cmsis

$(CMSIS_TEST): tests/test_fft_q15_cmsis.c tests/cfft_q15_plain_path.c \
               $(CMSIS_SRC) $(HDRS) tests/check.h
	$(CC) $(CMSIS_FLAGS) -Itests -o $@ tests/test_fft_q15_cmsis.c \
	    tests/cfft_q15_plain_path.c $(CMSIS_SRC) $(LDLIBS)

test-cmsis: $(CMSIS_TEST)
	./$(CMSIS_TEST)

clean: clean-cmsis
clean-cmsis:
	rm -f libfmcw_cmsis.so $(CMSIS_TEST)

.PHONY: lib-cmsis test-cmsis clean-cmsis
# this file is included above the main Makefile's first rule
.DEFAULT_GOAL := all
