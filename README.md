```
  ____   ____    _   _   ____    _   _   _   _
 / ___| |  _ \  | | | | |  _ \  | | | | | | | |
| |  _  | |_) | | | | | | |_) | | |_| | | |_| |
| |_| | |  __/  | |_| | |  _ <  |  _  | |  _  |
 \____| |_|      \___/  |_| \_\ |_| |_| |_| |_|
P                       o       o       a
U                       b       o       s
                        i       d       h
                        n               i
                                        n
                                        g
```

[![Docs](https://img.shields.io/badge/docs-design%20notes-blue)](docs/index.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A header-only CUDA library implementing a parallel hash table for NVIDIA GPUs, using Robin Hood open-addressing with bucketed (cache-line-aligned), sub-warp-cooperative probing.
The design target is bandwidth-bound performance on large tables.

# Status

Insert and lookup are implemented, tested, and benchmarked against cuCollections and WarpCore — on a 4090, lookups perform favorably at sufficiently high load factors.
See [docs/](docs/index.md) for design rationale and scope.

# Requirements

- NVIDIA GPU, sm_70 or newer (64-bit slots run natively on all supported arches; 128-bit slots run natively on sm_90+ and via libcu++ lock-table fallback otherwise).
- CUDA Toolkit 13.x (for the C++20 + libcu++ `bit_cast` combination this library relies on).
- A C++20 host compiler compatible with that nvcc.

# Layout

- `include/gpurhh/` — the library headers. Add this to your include path; everything ships from here.
- `tests/` — per-topic test executables built against the public header.
- `examples/` — small standalone programs. `examples/basic.cu` is a self-contained construct + bulk insert + bulk lookup walkthrough.
- `benchmarks/` — apples-to-apples timing benchmarks (with optional cuCollections and WarpCore baselines) and a bandwidth experiment.
- `docs/` — design documentation (the canonical record of *why* the implementation looks the way it does).

# Quick start

```cpp
#include <gpurhh/hash_table.cuh>

using Table = gpurhh::HashTable<std::uint32_t, std::uint32_t>;

Table table(1u << 20);          // host-side; capacity rounded up to next pow-2

__global__ void insert_kernel(Table::View view, const std::uint32_t* keys,
                              const std::uint32_t* values, std::size_t n) {
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tile_id     = (blockIdx.x * blockDim.x + threadIdx.x) / Table::tile_size;
    const std::size_t total_tiles = (gridDim.x  * blockDim.x)               / Table::tile_size;

    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        view.insert(tile, keys[i], values[i]);
    }
}
```

See [examples/basic.cu](examples/basic.cu) for the complete walkthrough and [docs/usage.md](docs/usage.md) for the API surface and the `View` pattern.

# Building

The repository ships a single `Makefile`:

- `make` — build tests, examples, and benchmarks.
- `make tests` / `make examples` / `make benchmarks` — build one group.
- `make test` — build and run all tests.
- `make clean` — wipe `build/`.

Overrides via env vars: `make ARCH=sm_90 CXX_STD=c++20 OPT=-O3`.
Benchmark binaries always build at `-O3`; unoptimized throughput numbers are meaningless.

Optional baselines under `external/include/` are picked up automatically; populate them with `scripts/setup-baselines.sh` if you want the cuCollections / WarpCore comparisons.

# Benchmark script

```sh
scripts/benchmark.sh --timing
scripts/benchmark.sh --memory-bandwidth
scripts/benchmark.sh --all          # everything available
```

Output lands under `output/<timestamp>/`, with `run_info.txt` capturing `nvcc --version`, `uname`, and `nvidia-smi` for each run.

# Documentation

Design and rationale live under [docs/](docs/), starting with [docs/index.md](docs/index.md).
The C++ API reference is produced separately by Doxygen from the headers themselves.

# License

[MIT](LICENSE).
