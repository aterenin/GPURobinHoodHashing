# Usage

`gpurhh` is built around a host-side `HashTable` resource and a device-side `View` handle.
The host object owns the bucket storage; the `View` is a trivially-copyable handle that user kernels take by value and operate on through cooperative tiles.

## The View pattern

The library is deliberately header-only and lean on host-side helpers.
Bulk insert / bulk lookup driver kernels are *not* part of the table — callers manage their device buffers and write their own driver kernels.
A user kernel that already has its keys in registers calls `view.insert(tile, ...)` or `view.get(tile, ...)` directly, and that is the canonical extension point.

The tests in `tests/kernels.cuh` and `tests/isolated.cuh` show the canonical pattern; [examples/basic.cu](../examples/basic.cu) is a minimal standalone walkthrough.

## Host API

The headers under `include/gpurhh/` define, on the host side:

- `gpurhh::HashTable<Key, Value, ...>` — the host-side owner.
  Non-copyable, movable.
  Constructed from a `min_capacity` (in slots) that is rounded up to the next power of two; provides `clear(stream)`, `capacity()`, and `data()` accessors, plus `view()` which produces the device handle.
- `gpurhh::replace_op`, `gpurhh::sum_op` — built-in reduction functors for the `Reduction` template parameter (see [Algorithm: Reduction operators](algorithm.md#reduction-operators)).
- `gpurhh::print_slots(table, start, stop)` — a diagnostic helper, lives in `<gpurhh/print.cuh>` as a separate header so the core library stays independent of `<cstdio>` and `<vector>`.

The `HashTable` template parameters are `<Key, Value, Hash = default_hash<Key>, EmptyKey = default_empty_key<Key>::key, Reduction = replace_op, CacheLineBytes = 128, WarpSize = 32, MaxProbeBuckets = 8>`.
The cache-line and warp-size parameters are present specifically so the table is not portability-pinned to NVIDIA constants (see [Could this scheme also work for AMD GPUs?](architecture.md#could-this-scheme-also-work-for-amd-gpus)).

## Device API

The device-side API is designed around the `View` handle, obtained host-side via `table.view()` and passed by value into user kernels.
`View` is trivially copyable and carries only the bucket pointer and the derived capacity/mask constants — no ownership.

It exposes two cooperative-tile methods, both of which take a `cooperative_groups::thread_block_tile<TileSize>` as their first argument (the `B` lanes that share each bucket load):

- `__device__ cuda::std::optional<Slot> insert(tile, key, value)` — insert a `(key, value)` pair under the table's `Reduction`.
  Returns an empty optional on success; on probe-cap failure returns the leftover slot that could not be placed — see the next section on lossless failure.
- `__device__ cuda::std::optional<Value> get(tile, key) const` — look up the value associated with `key`.
  Returns the value on hit, empty optional on miss.

Both methods are designed to be called from inside a user-written grid-stride loop where each tile owns one in-flight key per iteration.
The tests in `tests/kernels.cuh` and the example in `examples/basic.cu` show the canonical loop shape.

The full C++ signatures, parameter lists, and per-method semantics belong in the Doxygen-generated reference rather than being maintained by hand here.

## Lossless insert failure via returned slots

`View::insert` returns `cuda::std::optional<Slot>`.
On success it is empty; on probe-cap failure it holds the leftover `(key, value)` pair that could not be placed.

Crucially, this leftover may *not* be the original input.
If Robin Hood displacements happened during the probe, the originally-passed-in pair is already in the table and the leftover is the most-recently-evicted victim.
Returning it is what makes the operation *lossless* under failure — callers can buffer leftovers and replay them into a rebuilt larger table without losing any of the keys they handed in.

## Preprocessor-gated benchmark counters

One macro gates functionality that the core library doesn't expose by default.
It must be defined in the translation unit *before* any `<gpurhh/...>` header is included:

- **`GPURHH_BENCHMARK_COUNTERS`** adds a trailing `std::uint32_t& tile_counter` parameter to `View::insert` and `View::get`.
  Lane 0 of the tile increments it on every cache-line-sized memory event: each bucket-load at the top of a probe-loop iteration (including CAS-retry re-reads of the same bucket), *and* each attempted CAS — every CAS requires the line in an exclusive state, which is an additional memory transaction beyond the cooperative load.
  `View::get` is read-only and only bumps on bucket loads.
  The caller maintains a per-tile register accumulator and writes it out once at end of kernel; no atomics fire in the hot loop.
  Downstream bandwidth = `counter × sizeof(Bucket) / time`.
  The memory-bandwidth benchmark binaries enable this; the timing benchmarks, tests, and the example do not.
