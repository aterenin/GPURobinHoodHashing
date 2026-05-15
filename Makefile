# Makefile for gpurhh — header-only GPU Robin Hood hash table.
#
# Builds tests and examples against the public header in include/. There is
# no library to compile (header-only). Each .cu file under tests/ and
# examples/ becomes its own executable in build/.
#
# Common overrides:
#   make NVCC=/path/to/nvcc
#   make ARCH=sm_75
#   make CXX_STD=c++17
#   make OPT="-O0 -g"

NVCC    ?= nvcc
CXX_STD ?= c++20
ARCH    ?= sm_80
OPT     ?= -O2

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

.PHONY: all tests examples test clean

all: tests examples

tests:    $(TEST_BINS)
examples: $(EXAMPLE_BINS)

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
