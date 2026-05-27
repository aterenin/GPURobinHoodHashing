# GPU Robin Hood Hashing

`gpurhh` is a header-only CUDA library implementing a parallel Robin Hood hash table for NVIDIA GPUs.
It targets bandwidth-bound performance on large tables by combining cache-line-aligned bucketed probing with sub-warp-cooperative tiles.

These Markdown design pages and the C++ API reference (extracted from the source) are bundled into a unified HTML site by Doxygen.

## Goals

- **Header-only CUDA library.**
  A single `.cuh` that users `#include` and template-instantiate for their key/value types.
  No separate compilation of the table itself.
- **Massively parallel insertion and retrieval.**
  A batch of `N` keys is inserted or retrieved by `N` (or more) GPU threads cooperatively.
- **Memory-bandwidth-oriented design.**
  On large tables, the table is much bigger than the L2 cache, so performance is determined by how many useful bytes each DRAM transaction delivers.
  The aim is to come as close to peak HBM bandwidth as the access pattern allows.

## Navigating these pages

- [Usage](usage.md) — the `View` pattern, host and device API, lossless insert failure, and preprocessor-gated benchmark counters.
- [Architecture](architecture.md) — why Robin Hood, the slot layout, the bandwidth-saturation argument, and how work decomposes onto the GPU hierarchy.
- [Algorithm](algorithm.md) — parallel insertion with potential reduction operators, and lookup.
- [Implementation](implementation.md) — language and toolchain choices, the power-of-two assumption on capacity, and hash function.
- [Testing](testing.md) — the test suite, shared infrastructure, and what the `IdentityHash` fixture buys us.
- [Benchmarks](benchmarks.md) — running the benchmarks, code structure, workload, experiment phases, and per-tile counters.
- [Limitations](limitations.md) — known limitations and potential future work.
