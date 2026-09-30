# saphira-llm — native C11 inference runtime for Saphira Linux.
#
# House style: plain POSIX make, C11, musl-compatible, no glibc-only calls,
# no mandatory BLAS, no Python anywhere in the build or the runtime.

CC      ?= cc
PREFIX  ?= /usr/local
BINDIR  ?= $(PREFIX)/bin

# x86-64-v3 is the baseline, not a fallback tier. The whole binary is compiled
# for it, so v3 instructions are legitimate anywhere. Anything ABOVE v3 must
# live in an SLLM_ISA_EXT section and be reached only through the dispatcher;
# `make check-isa` proves that mechanically.
BASELINE ?= -march=x86-64-v3

# ARITHMETIC CONTRACT. Regression results must not depend on which compiler
# built the tree.
#
# Measured on this source: with FMA contraction enabled, gcc and clang differ
# by 0.0034 in mean_nll (perplexity 145.0836 against 145.5773, a 0.34% gap) on
# the in-tree fixture. With contraction disabled they agree BIT-EXACTLY. The
# whole difference is the forward pass fusing multiply-adds, and it is large
# enough to be mistaken for a model change.
#
# So contraction is off by default, for the library, the tests, the evaluator
# and the training tooling alike. One arithmetic for everything: a test suite
# that built the library differently from the shipped binary would be testing
# something nobody runs.
#
# Overriding this is the deliberate production-FMA decision, which is a separate
# performance benchmark and is NOT the quality baseline. See docs/DECISIONS.md
# decision 1. v0.0.1 at tag 7dbcc67 predates this and is unaffected: the tag
# names a tree, not a build.
FPFLAGS ?= -ffp-contract=off

# _POSIX_C_SOURCE is required because we target musl/POSIX, not bare ISO C:
# mmap, setenv and friends are POSIX, and -std=c11 alone hides them.
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CFLAGS  += $(BASELINE) $(FPFLAGS) -Iinclude -D_POSIX_C_SOURCE=200809L
LDFLAGS ?=
LDLIBS   = -lm -lpthread

ifeq ($(DEBUG),1)
CFLAGS  += -g -O0 -fno-omit-frame-pointer
endif

# Separate trees so the sanitiser build never contaminates the release objects.
OBJDIR   = build/obj
BIN      = saphira-llm
TESTBIN  = saphira-llm-test

CORE_SRC = src/status.c src/log.c src/isa.c src/gguf.c src/kernel_probe.c \
           src/eval.c \
           src/topology.c src/thread.c src/quant.c src/ops.c src/unicode_data.c src/i2s_gemm.c src/tokenizer.c src/forward.c
MAIN_SRC = src/main.c
TEST_SRC = tests/main.c tests/test_isa.c tests/test_gguf.c tests/test_thread.c \
            tests/test_eval.c \
            tests/test_ops.c tests/test_i2s.c tests/test_tokenizer.c tests/test_forward.c \
            tests/test_phase5.c tests/test_dot_f16.c

CORE_OBJ = $(CORE_SRC:%.c=$(OBJDIR)/%.o)
MAIN_OBJ = $(MAIN_SRC:%.c=$(OBJDIR)/%.o)
TEST_OBJ = $(TEST_SRC:%.c=$(OBJDIR)/%.o)

ALL_OBJ  = $(CORE_OBJ) $(MAIN_OBJ) $(TEST_OBJ)
LIBOBJ   = $(CORE_OBJ)

BENCH_SRC = bench/sched_bench.c
BENCH_OBJ = $(BENCH_SRC:%.c=$(OBJDIR)/%.o)
BENCH_BIN = saphira-llm-schedbench

.PHONY: all clean install test check check-isa san asan ubsan bench format-check help

all: $(BIN)

$(BIN): $(CORE_OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(TESTBIN): $(CORE_OBJ) $(TEST_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BENCH_BIN): $(CORE_OBJ) $(BENCH_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Model-quality measurement. Separate executable from saphira-llm on purpose:
# this is a measurement, not a gate, and folding it into the inference binary
# would invite treating a number as a pass/fail.
EVAL_SRC = src/eval_main.c
EVAL_OBJ = $(EVAL_SRC:%.c=$(OBJDIR)/%.o)
EVAL_BIN = saphira-llm-eval

.PHONY: eval
eval: $(EVAL_BIN)

$(EVAL_BIN): $(EVAL_OBJ) $(LIBOBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm -lpthread

# ---------------------------------------------------------------- tests

test: $(TESTBIN)
	./$(TESTBIN)

# Scheduler benchmark. Measures the Phase 2 gate: no full-occupancy collapse.
bench: $(BENCH_BIN)
	./$(BENCH_BIN) $(BENCH_ARGS)

# The mechanical baseline proof. Reads the disassembled binary and fails if an
# instruction above x86-64-v3 appears anywhere except the guarded section.
check-isa: $(BIN) $(TESTBIN) $(CORE_OBJ)
	@SLLM_BASELINE_ISA="$(SLLM_BASELINE_ISA)" sh scripts/check-baseline-isa.sh

check: test check-isa

# ---------------------------------------------------------------- sanitizers

# Sanitizers.
#
# These use clang, not the default cc, because this musl target has no
# libasan/libubsan for the Saphira gcc: -fsanitize fails at link with
# "cannot find libasan_preinit.o". clang ships its own compiler-rt, and
# ASan+UBSan work here dynamically. A static clang link fails on _DYNAMIC
# under musl, so the sanitizer builds are dynamic. The release build is
# unaffected and is what Saphira ships.
SAN_CC ?= clang

san:
	$(MAKE) clean
	$(MAKE) test CC="$(SAN_CC)" DEBUG=1 \
		CFLAGS="-O1 -g -std=c11 -Wall -Wextra -Wpedantic $(BASELINE) $(FPFLAGS) \
			-Iinclude -D_POSIX_C_SOURCE=200809L \
			-fsanitize=address,undefined -fno-omit-frame-pointer \
			-fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=address,undefined"

asan: san
ubsan: san

# ---------------------------------------------------------------- install

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)$(BINDIR)/$(BIN)

clean:
	rm -rf build $(BIN) $(TESTBIN) $(BENCH_BIN)

help:
	@echo 'targets: all test check check-isa san asan ubsan install clean'
	@echo '         eval tgbench kernelbench topoprobe'
	@echo 'vars   : CC CFLAGS BASELINE FPFLAGS PREFIX BINDIR DEBUG=1 SAN_CC=clang'
	@echo '  BASELINE defaults to $(BASELINE)'

# Phase 6 decode-throughput harness. Loads the model once and sweeps thread
# counts, so the numbers are not mixed up with page faults and cold cache.
TGBENCH_SRC = bench/tg_bench.c bench/prof.c
TGBENCH_OBJ = $(TGBENCH_SRC:%.c=$(OBJDIR)/%.o)
TGBENCH_BIN = saphira-llm-tgbench

.PHONY: tgbench
tgbench: $(TGBENCH_BIN)

$(TGBENCH_BIN): $(TGBENCH_OBJ) $(LIBOBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm -lpthread

# Isolated timing for the ternary dot kernels. Interleaves the two paths inside
# one process, because end-to-end drift is larger than the effect being measured.
KERNELBENCH_SRC = bench/kernel_bench.c
KERNELBENCH_OBJ = $(KERNELBENCH_SRC:%.c=$(OBJDIR)/%.o)
KERNELBENCH_BIN = saphira-llm-kernelbench

.PHONY: kernelbench
kernelbench: $(KERNELBENCH_BIN)

$(KERNELBENCH_BIN): $(KERNELBENCH_OBJ) $(LIBOBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm -lpthread

# Topology classification diagnostic. Prints per-CPU scores, within-CPU spread
# and the margin to the best, plus a threshold sweep, so the P/E classification
# rule can be chosen from evidence instead of guessed.
TOPOPROBE_SRC = bench/topo_probe.c
TOPOPROBE_OBJ = $(TOPOPROBE_SRC:%.c=$(OBJDIR)/%.o)
TOPOPROBE_BIN = saphira-llm-topoprobe

.PHONY: topoprobe
topoprobe: $(TOPOPROBE_BIN)

$(TOPOPROBE_BIN): $(TOPOPROBE_OBJ) $(LIBOBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm -lpthread
