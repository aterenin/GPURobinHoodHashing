# Makefile for gpurhh — header-only GPU Robin Hood hash table.
#
# Builds tests, examples, and benchmarks against the public header in
# include/. There is no library to compile (header-only). Each .cu file
# under tests/, examples/, and benchmarks/ becomes its own executable
# under build/.
#
# OPT controls optimization / debug flags. Default behavior:
#   - tests / examples: empty (nvcc's default; fastest to compile)
#   - benchmarks:       -O3 (unoptimized throughput numbers are meaningless)
#
# Override OPT to apply to every target uniformly:
#
#   make tests                          unoptimized
#   make tests     OPT=-O3              optimized tests
#   make benchmark                      sweep at -O3 (default for benchmarks)
#   make benchmark OPT=-O0              sweep unoptimized (rarely useful)
#   make test      OPT="-O0 -G -g"      debuggable test build
#
# EXTRA is appended to the nvcc command line on top of OPT. Useful for
# one-off additions (e.g. `make tests EXTRA=-lineinfo`).
#
# Common overrides:
#   make NVCC=/path/to/nvcc
#   make ARCH=sm_75
#   make CXX_STD=c++17

NVCC    ?= nvcc
CXX_STD ?= c++20
ARCH    ?= sm_89

OPT     ?=
EXTRA   ?=

# Benchmark-specific OPT: -O3 if the user didn't set OPT explicitly, else
# whatever they passed. Lives in its own variable so the pattern rule
# below stays straightforward.
ifeq ($(strip $(OPT)),)
    BENCHMARK_OPT := -O3
else
    BENCHMARK_OPT := $(OPT)
endif

INCLUDES   := -Iinclude

NVCC_FLAGS       := -std=$(CXX_STD) -arch=$(ARCH) $(OPT)       $(INCLUDES) $(EXTRA)
NVCC_FLAGS_BENCHMARK := -std=$(CXX_STD) -arch=$(ARCH) $(BENCHMARK_OPT) $(INCLUDES) $(EXTRA)

BUILD := build

TEST_SRCS    := $(wildcard tests/test_*.cu)
TEST_BINS    := $(patsubst tests/%.cu,$(BUILD)/tests/%,$(TEST_SRCS))

EXAMPLE_SRCS := $(wildcard examples/*.cu)
EXAMPLE_BINS := $(patsubst examples/%.cu,$(BUILD)/examples/%,$(EXAMPLE_SRCS))

BENCHMARK_SRCS   := $(wildcard benchmarks/benchmark_*.cu)
BENCHMARK_BINS   := $(patsubst benchmarks/%.cu,$(BUILD)/benchmarks/%,$(BENCHMARK_SRCS))

# Treat any header change as a reason to rebuild everything. Adequate for a
# small project; revisit if compile times become an issue.
ALL_HEADERS := $(shell find include tests benchmarks -name "*.cuh" 2>/dev/null)

.PHONY: all tests examples benchmarks test benchmark clean

all: tests examples

tests:      $(TEST_BINS)
examples:   $(EXAMPLE_BINS)
benchmarks: $(BENCHMARK_BINS)

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "==> $$t"; "$$t"; done

# Run the full benchmark sweep. Benchmark binaries default to -O3 (see
# BENCHMARK_OPT above); override via OPT=... if needed.
benchmark: $(BENCHMARK_BINS)
	@bash scripts/benchmark.sh

$(BUILD)/tests/%: tests/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/tests
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/examples/%: examples/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/examples
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/benchmarks/%: benchmarks/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/benchmarks
	$(NVCC) $(NVCC_FLAGS_BENCHMARK) $< -o $@

clean:
	rm -rf $(BUILD)
