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
  Shared driver kernels, CUDA error-check operator, and host-side bulk helpers live in `tests/common.cuh`; small CUDA utilities like `CUDA_CHECK` live in `tests/utils.cuh`.
- `examples/` — small standalone programs demonstrating use of the library.
  Currently a placeholder; will be populated as the API stabilises.
- `notes/` — design notes, written for humans.
  These files are the canonical record of *why* the implementation looks the way it does.
  They are intended to eventually become the project's documentation, but they are not documentation yet — they are working notes that capture decisions as we make them.
  Read these before touching code.
- `Makefile` — build entry point (placeholder; not yet populated).

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
