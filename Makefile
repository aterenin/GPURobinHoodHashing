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
#   make benchmarks OPT=-O0             benchmarks unoptimized (rarely useful)
#   make test      OPT="-O0 -G -g"      debuggable test build
#
# Tests run via `make test`; benchmarks run via the standalone driver
# scripts/benchmark.sh (optionally with --with-baselines). The
# benchmark sweep is heavy enough that keeping its invocation out of
# the Makefile makes the inner-loop test target faster and clearer.
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
# --extended-lambda allows __device__-annotated lambdas in benchmark code:
# benchmarks/benchmarks.cuh uses one inside count_occupied_slots, and the
# cuco baseline uses one to synthesize cuco::pair<Key, Value> on the fly.
NVCC_FLAGS_BENCHMARK := -std=$(CXX_STD) -arch=$(ARCH) $(BENCHMARK_OPT) $(INCLUDES) $(EXTRA) --extended-lambda

BUILD := build

TEST_SRCS    := $(wildcard tests/test_*.cu)
TEST_BINS    := $(patsubst tests/%.cu,$(BUILD)/tests/%,$(TEST_SRCS))

EXAMPLE_SRCS := $(wildcard examples/*.cu)
EXAMPLE_BINS := $(patsubst examples/%.cu,$(BUILD)/examples/%,$(EXAMPLE_SRCS))

# Two benchmark studies under benchmarks/: timing (apples-to-apples kernel
# throughput vs. cuco / warpcore baselines) and memory_bandwidth (the
# counter-instrumented gpurhh study plus the memcpy ceiling reference,
# whose ratio gives the bandwidth story).
BENCHMARK_SRCS := \
    $(wildcard benchmarks/timing/benchmark_*.cu) \
    $(wildcard benchmarks/memory_bandwidth/benchmark_*.cu)

# Optional baseline libraries. Each is included only if the corresponding
# header subtree is present under external/include/ (which is gitignored —
# run scripts/setup-baselines.sh to populate, see that script for the
# pinned commits). When absent, the baselines silently drop out and
# `make benchmarks` produces just the gpurhh binaries.
EXTERNAL_INCLUDE := external/include

HAVE_CUCO     := $(wildcard $(EXTERNAL_INCLUDE)/cuco)
HAVE_WARPCORE := $(wildcard $(EXTERNAL_INCLUDE)/warpcore)

ifneq ($(HAVE_CUCO),)
    # cuco/      — static_map with its default linear_probing<4>.
    # cuco_dh/   — same headers, static_map specialized with
    #              double_hashing<8> so we can isolate the probing-scheme
    #              variable from everything else cuco does.
    BENCHMARK_SRCS += $(wildcard benchmarks/timing/baselines/cuco/benchmark_*.cu)
    BENCHMARK_SRCS += $(wildcard benchmarks/timing/baselines/cuco_dh/benchmark_*.cu)
endif
ifneq ($(HAVE_WARPCORE),)
    BENCHMARK_SRCS += $(wildcard benchmarks/timing/baselines/warpcore/benchmark_*.cu)
endif

BENCHMARK_BINS   := $(patsubst benchmarks/%.cu,$(BUILD)/benchmarks/%,$(BENCHMARK_SRCS))

# Treat any header change as a reason to rebuild everything. Adequate for a
# small project; revisit if compile times become an issue.
ALL_HEADERS := $(shell find include tests benchmarks -name "*.cuh" 2>/dev/null)

.PHONY: all tests examples benchmarks test docs clean

all: tests examples benchmarks

tests:      $(TEST_BINS)
examples:   $(EXAMPLE_BINS)
benchmarks: $(BENCHMARK_BINS)

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "==> $$t"; "$$t"; done

# HTML documentation. Combines the hand-written design pages under
# docs/ with the API reference extracted from include/gpurhh/ by
# Doxygen. Output lands in build/docs/html/; open index.html.
docs:
	doxygen Doxyfile

$(BUILD)/tests/%: tests/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/tests
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/examples/%: examples/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/examples
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/benchmarks/%: benchmarks/%.cu $(ALL_HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS_BENCHMARK) $< -o $@ -lcurand

# Baseline pattern rule — more specific than the catch-all benchmark
# rule above, so it wins for targets under build/benchmarks/timing/baselines/.
# Both cuco and warpcore share -Iexternal/include since the script
# consolidates them under that single tree. --expt-relaxed-constexpr is
# required by cuCollections's public API (and harmless for warpcore);
# --extended-lambda comes from NVCC_FLAGS_BENCHMARK above.
$(BUILD)/benchmarks/timing/baselines/%: benchmarks/timing/baselines/%.cu $(ALL_HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS_BENCHMARK) -I$(EXTERNAL_INCLUDE) --expt-relaxed-constexpr $< -o $@ -lcurand

clean:
	rm -rf $(BUILD)
