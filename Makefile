# Makefile for gpurhh — header-only GPU Robin Hood hash table.
#
# Builds tests and examples against the public header in include/. There is
# no library to compile (header-only). Each .cu file under tests/ and
# examples/ becomes its own executable in build/.
#
# Build modes (pick at invocation time):
#   make            no optimization flags passed (nvcc's default); the default
#   make release    optimized build (-O2)
#   make debug      unoptimized + host/device debug info (-O0 -G -g)
#
# Common overrides:
#   make NVCC=/path/to/nvcc
#   make ARCH=sm_75
#   make CXX_STD=c++17
#   make OPT="-O3 -lineinfo"   # bypass the mode selection entirely

NVCC    ?= nvcc
CXX_STD ?= c++20
ARCH    ?= sm_89

# Pick optimization flags based on the goal that was passed on the command
# line. Falls through to -O0 for plain `make` or any non-mode goal.
ifneq (,$(filter release,$(MAKECMDGOALS)))
    OPT ?= -O2
else ifneq (,$(filter debug,$(MAKECMDGOALS)))
    OPT ?= -O0 -G -g
else
    OPT ?=
endif

INCLUDES   := -Iinclude
NVCC_FLAGS := -std=$(CXX_STD) -arch=$(ARCH) $(OPT) $(INCLUDES)

BUILD := build

TEST_SRCS    := $(wildcard tests/test_*.cu)
TEST_BINS    := $(patsubst tests/%.cu,$(BUILD)/tests/%,$(TEST_SRCS))

EXAMPLE_SRCS := $(wildcard examples/*.cu)
EXAMPLE_BINS := $(patsubst examples/%.cu,$(BUILD)/examples/%,$(EXAMPLE_SRCS))

# Treat any header change as a reason to rebuild everything. Adequate for a
# small project; revisit if compile times become an issue.
ALL_HEADERS := $(shell find include tests -name "*.cuh" 2>/dev/null)

.PHONY: all tests examples test release debug clean

all: tests examples

tests:    $(TEST_BINS)
examples: $(EXAMPLE_BINS)

# Build-mode aliases. The optimization flags are picked above based on the
# goal name; these targets just funnel work to `all`.
release: all
debug:   all

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "==> $$t"; "$$t"; done

$(BUILD)/tests/%: tests/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/tests
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/examples/%: examples/%.cu $(ALL_HEADERS)
	@mkdir -p $(BUILD)/examples
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD)
