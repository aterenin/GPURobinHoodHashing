# Implementation

## Language and toolchain

- **Target language: C++20.**
- **Target CUDA toolkit: CUDA 13.x.**
  Driven by two requirements: `-std=c++20` support in nvcc (CUDA 12.0+) and `cuda::std` namespaced operations in libcu++ (CUDA 13.0 was the first version where this combination worked cleanly on the test system).

Within the implementation, we stick to C++17-shape idioms by default and only reach for C++20 features where they demonstrably improve the code.
The C++20 uses today are:

- `std::bit_width` in `next_pow2` (host-side, compile-time).
- Host-side compile-time `std::bit_cast` for the `empty_key_byte` extraction and the memsettability `static_assert` lambda.
- Device-side `cuda::std::bit_cast` (libcu++) for the runtime bit-reinterpret in `default_hash`.
  The libcu++ version is mandatory here because `std::bit_cast` is host-only `constexpr` in libstdc++; calling it from a `__device__` function is a hard error in CUDA 13.x.

Concepts on the `Hash` template parameter, `std::source_location` in place of `__FILE__` / `__LINE__` macros, ranges, and coroutines are all deferred until they pay for themselves.

## Capacity, hash function, load factor

- We require that capacity is a power of two so that `mod` becomes a mask.
  The constructor takes `min_capacity` (in slots) and rounds up to the next power of two, never below `bucket_size`.
  Users targeting a specific load factor compute the slot count themselves (e.g. `min_capacity = ceil(N / target_load_factor)` for an expected `N` insertions).
- Hash function is a template parameter; default is a finalizer-style mixer (e.g. the splitmix64/Murmur3 finalizer) that is cheap on GPU and decorrelates most structured keys reasonably well.
- The design targets load factors of 0.85–0.95. Robin Hood's tight, low-variance probe-distance distribution keeps the warp's tail latency in check even at these high loads, and the benchmarks show this is precisely the range where the bandwidth advantage over linear probing and double hashing is most pronounced.

### Where the power-of-two assumption is used

The "capacity is a power of two" constraint is not just a convenience — it's wired into multiple operations.
Anyone touching these sites should know that breaking the invariant would require switching every `& mask` back to a `% capacity`, which could be slower on certain GPU (multi-cycle integer divmod vs. a single AND).

In [include/gpurhh/hash_table.cuh](../include/gpurhh/hash_table.cuh):

- **`HashTable` constructor**: `next_pow2(min_capacity)` rounds the user-supplied capacity up; `capacity_mask_ = capacity_ - 1` is cached for use across the hot paths.
  Capacity can grow by up to 2× over what the user asked for.
  The trade-off is a one-cycle mask vs. a multi-cycle modulus on every probe step.
- **Home-slot computation** (insert and get): `home_slot = hash(key) & capacity_mask_` (and `home_bucket = home_slot / bucket_size`).
  Replacing `&` with `%` would also require `capacity_mask_` to be replaced by the raw capacity, doubling the mod cost on every probe.
- **Probe wrap-around**: `bucket_idx = (home_bucket + probe) & bucket_mask`, where `bucket_mask = num_buckets - 1`.
  Since `num_buckets = capacity / bucket_size` and both are powers of two, the bucket count is also pow-2 and admits a mask.
- **Probe-distance derivation** (used inside the Robin Hood swap check): `resident_probe = (bucket_idx - resident_home) & bucket_mask` — this handles wrap-around correctly precisely because `bucket_mask = num_buckets - 1`.
- **Tile size and warp size**: independently asserted pow-2 via `static_assert((tile_size & (tile_size - 1)) == 0, ...)`.
  Used by `tiles_per_block = block_size / tile_size` and by `cg::tiled_partition<tile_size>`, which itself requires a pow-2 tile size.

In benchmarks ([benchmarks/benchmarks.cuh](../benchmarks/benchmarks.cuh)):

- **Key generation** (`map_to_range`): folds cuRAND bits to `[0, range)` via `x & (range - 1)` when range is a power of two, falling back to `x % range` otherwise.
  The benchmark script always uses `key_range = capacity`, so the mask path is taken in practice.

A subtler dependency lives in the **default hash function**.
With a pow-2 mask, only the low bits of `hash(key)` reach `capacity_mask_` — the high bits are thrown away.
If a user supplied an identity hash on structured keys (e.g. dense integer IDs), low-bit clustering would map directly into slot clustering and probe lengths would explode.
The default `default_hash<uint32_t>` is `fmix32`, a finalizer-style mixer that spreads bits well, so the low bits of `hash(k)` are pseudo-uniform even when keys are not.
Users supplying their own `Hash` should preserve this property; passing a non-mixing hash with a pow-2 capacity is the textbook way to break a fast hash table.
