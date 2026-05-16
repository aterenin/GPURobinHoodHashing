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
  Shared driver kernels, table aliases, and host-side bulk helpers live in `tests/kernels.cuh`; the CUDA error-check operator and other test-wide infrastructure (including the macro that exposes `HashTable::data()`) live in `tests/tests.cuh`.
- `examples/` — small standalone programs demonstrating use of the library.
  Currently a placeholder; to be filled in with a real construct + bulk insert + bulk lookup demo.
- `notes/` — design notes, written for humans.
  These files are the canonical record of *why* the implementation looks the way it does.
  They are intended to eventually become the project's documentation, but they are not documentation yet — they are working notes that capture decisions as we make them.
  Read these before touching code.
- `Makefile` — builds tests and examples with `nvcc`. Defaults to C++20, `sm_89`, and no optimization.
  Build modes: `make` (no `-O`), `make release` (`-O3`), `make debug` (`-O0 -G -g`).
  Other useful targets: `make test` builds and runs all tests, `make clean` wipes `build/`. Override flags via env vars (`make ARCH=sm_90 CXX_STD=c++17`).

When new design decisions are made or revised, update the relevant note in `notes/` rather than scattering rationale across commit messages or code comments.

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
