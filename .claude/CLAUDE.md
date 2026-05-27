# CLAUDE.md

Project context for Claude Code.
This file is loaded automatically at the start of every conversation in this repository.

## What this project is

A header-only CUDA library implementing a parallel hash table for NVIDIA GPUs, using Robin Hood open-addressing with bucketed (cache-line-aligned), sub-warp-cooperative probing.
The design target is bandwidth-bound performance on large tables.

## Repository layout

- `include/gpurhh/` — public headers.
  This is what users `#include` after adding `include/` to their include path.
  The library is header-only; everything ships from here.
- `tests/` — tests built against the public header.
  Each `test_*.cu` file is compiled as its own executable with its own `main`, scoped to one topic (a public method, a trait, etc.).
  Shared driver kernels, table aliases, and host-side bulk helpers live in `tests/kernels.cuh`; the CUDA error-check operator lives in `tests/tests.cuh`; `tests/isolated.cuh` provides the `IdentityHash`-based fixture that lets tests hand-build and verify Robin Hood states.
- `examples/` — small standalone programs demonstrating use of the library.
  `examples/basic.cu` is a self-contained construct + bulk insert + bulk lookup + pretty-print walkthrough.
- `benchmarks/` — split into `timing/` (apples-to-apples library throughput vs. cuCollections and WarpCore) and `memory_bandwidth/` (counter-instrumented gpurhh study + memcpy ceiling reference).
  Run via `scripts/benchmark.sh`; output lands under `output/<timestamp>/`.
  `external/` (gitignored) holds the sparse-checkout of baseline headers; populate with `scripts/setup-baselines.sh`.
- `docs/` — design documentation, written for humans.
  These files are the canonical record of *why* the implementation looks the way it does, split into thematic pages (`index.md`, `usage.md`, `architecture.md`, `algorithm.md`, `implementation.md`, `testing.md`, `benchmarks.md`, `limitations.md`).
  The C++ API reference is produced separately by Doxygen from the headers themselves.
  Read these before touching code.
- `Makefile` — builds tests, examples, and benchmarks with `nvcc`. Defaults to C++20, `sm_89`, and no optimization for tests/examples; `-O3` for benchmarks (since unoptimized throughput numbers are meaningless).
  Targets: `make` (= `make all` — tests + examples + benchmarks), `make tests`, `make examples`, `make benchmarks`, `make test` (build + run all tests), `make clean`.
  Override flags via env vars (`make ARCH=sm_90 CXX_STD=c++17 OPT=-O3`); benchmark binaries always build at `-O3` regardless of `OPT`.

When new design decisions are made or revised, update the relevant page in `docs/` rather than scattering rationale across commit messages or code comments.

## Conventions for working in this repo

- **Notes are working documents that will become documentation.**
  Keep them prose-first and structured.
  Avoid jargon shorthand that only makes sense inside one conversation, since these will eventually be read by people who weren't in the room.
- **Architecture-derived constants are derived, not chosen.**
  `BucketSize`, `TileSize`, `TilesPerWarp` are all functions of `SlotBytes`, `CacheLineBytes`, and `WarpSize`.
  The latter two should be table-level template parameters, not literals embedded in the implementation.

## Out of scope (for now)

- Deletion (insert + lookup first).
- Resize / dynamic growth.
- Multi-GPU.
- AMD / non-NVIDIA backends — but the abstractions should not preclude them.
