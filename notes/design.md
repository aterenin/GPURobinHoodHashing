# GPU Robin Hood Hash Table — Design Notes

This document records the design and decisions behind `gpurhh`, a header-only CUDA library implementing a parallel Robin Hood hash table for NVIDIA GPUs.
It covers what the library does, why it looks the way it does, the public API it exposes, how the implementation is tested, and what is deliberately out of scope.

## Goals

- **Header-only CUDA library.**
  A single `.cuh` (or a small set) that users `#include` and template-instantiate for their key/value types.
  No separate compilation of the table itself.
- **Massively parallel insertion.**
  A batch of `N` keys is inserted by `N` (or more) GPU threads cooperatively.
  Worst-case behavior must remain bounded even under heavy contention.
- **Memory-bandwidth bound.**
  On large tables, the table is much bigger than the L2 cache, so performance is determined by how many useful bytes each DRAM transaction delivers.
  The aim is to come as close to peak HBM bandwidth as the access pattern allows.

## Language and toolchain

- **Target language: C++20.**
- **Target CUDA toolkit: CUDA 13.x.**
  Driven by two requirements: `-std=c++20` support in nvcc (CUDA 12.0+) and `cuda::std::bit_cast` in libcu++ (CUDA 13.0 was the first version where this combination worked cleanly on the test system).

Within the implementation, we stick to C++17-shape idioms by default and only reach for C++20 features where they demonstrably improve the code.
The C++20 uses today are:

- `std::bit_width` in `next_pow2` (host-side, compile-time).
- Host-side compile-time `std::bit_cast` for the `empty_key_byte` extraction and the memsettability `static_assert` lambda.
- Device-side `cuda::std::bit_cast` (libcu++) for the runtime bit-reinterpret in `default_hash`.
  The libcu++ version is mandatory here because `std::bit_cast` is host-only `constexpr` in libstdc++; calling it from a `__device__` function is a hard error in CUDA 13.x.

Concepts on the `Hash` template parameter, `std::source_location` in place of `__FILE__` / `__LINE__` macros, ranges, and coroutines are all deferred until they pay for themselves.

The C++20 baseline is conservative future-proofing: it lets us reach for the cleaner spelling when we want it without revisiting the build settings each time.

## Why Robin Hood?

Open-addressing with the invariant *"the entry in slot `i` whose probe distance is smaller gets evicted by an entry whose probe distance is larger"* gives:

- A tight, low-variance distribution of probe distances → small worst-case lookup length even at load factors of 0.85–0.9.
- A natural early-exit rule for lookups (stop as soon as the slot's probe distance is smaller than the probe distance of the key we're searching for).
- Backward-shift deletion that avoids tombstones.

These properties matter even more on a GPU than on a CPU because the warp's tail latency is set by the longest probe sequence among its 32 lanes.

## Slot layout

A slot is the unit of an atomic update.
The simplest design that admits lock-free CAS on a single machine word is:

```
struct alignas(sizeof(Key) + sizeof(Value)) Slot {
    Key   key;       // typically uint32_t or uint64_t
    Value value;     // typically the same width as Key
};
```

Empty slots use a reserved sentinel key (default: all-bits-set for unsigned integer Key); users are forbidden from inserting that key.
The packed slot supports two widths:

1. **64-bit slots** (4-byte key + 4-byte value): atomic CAS via `cuda::atomic_ref<Slot>`, which on every NVIDIA arch compiles down to a 64-bit `atom.cas`. Fast.
2. **128-bit slots** (8-byte key + 8-byte value): atomic CAS still goes through `cuda::atomic_ref<Slot>`; on sm_90+ this is the single-instruction `atom.cas.b128`, on sm_70–sm_89 libcu++ emulates with a lock-table fallback. Functional everywhere, but the emulated path is significantly slower than 64-bit CAS under contention.

We use the same control flow for both widths — the only thing that changes is which atomic instruction `cuda::atomic_ref::compare_exchange_strong` resolves to. A parallel-arrays layout (separate keys[] and values[]) is rejected: it halves the per-CAS bytes but at the cost of an awkward second-read-the-key step for finders that complicates correctness.

**Probe distance.**
We do *not* store probe distance in the slot.
It is derivable as `(slot_index - hash(key)) mod capacity`.
Storing it would cost bits we'd rather give to the payload, and the derivation is a single subtract once we've read the slot.

## Capacity, hash function, load factor

- Capacity is a power of two so `mod` becomes a mask.
  Users specify a target load factor (default 0.7); we round capacity up to `next_pow2(N / load_factor)`.
- Hash function is a template parameter; default is a finalizer-style mixer (e.g. the splitmix64/Murmur3 finalizer) that is cheap on GPU and decorrelates structured keys well.
- Robin Hood tolerates higher load than linear probing, but on a GPU the warp tail-latency argument pushes us back toward 0.7–0.8 rather than 0.9.

### Where the power-of-two assumption is load-bearing

The "capacity is a power of two" constraint is not just a convenience — it's wired into multiple hot-path operations. Anyone touching these sites should know that breaking the invariant would require switching every `& mask` back to a `% capacity`, which is materially slower on GPU (multi-cycle integer divmod vs. a single AND).

In `include/gpurhh/hash_table.cuh`:

- **`HashTable` constructor**: `next_pow2(min_capacity)` rounds the user-supplied capacity up; `capacity_mask_ = capacity_ - 1` is cached for use across the hot paths. Capacity can grow by up to 2× over what the user asked for. The trade-off is a one-cycle mask vs. a multi-cycle modulus on every probe step.
- **Home-slot computation** (insert and get): `home_slot = hash(key) & capacity_mask_` (and `home_bucket = home_slot / bucket_size`). Replacing `&` with `%` would also require capacity_mask_ to be replaced by the raw capacity, doubling the mod cost on every probe.
- **Probe wrap-around**: `bucket_idx = (home_bucket + probe) & bucket_mask`, where `bucket_mask = num_buckets - 1`. Since `num_buckets = capacity / bucket_size` and both are powers of two, the bucket count is also pow-2 and admits a mask.
- **Probe-distance derivation** (used inside the Robin Hood swap check): `resident_probe = (bucket_idx - resident_home) & bucket_mask` — this handles wrap-around correctly precisely because `bucket_mask = num_buckets - 1`.
- **Tile size and warp size**: independently asserted pow-2 via `static_assert((tile_size & (tile_size - 1)) == 0, ...)`. Used by `tiles_per_block = block_size / tile_size` and by `cg::tiled_partition<tile_size>`, which itself requires a pow-2 tile size.

In benchmarks (`benchmarks/benchmarks.cuh`):

- **Key generation** (`map_to_range`): folds cuRAND bits to `[0, range)` via `x & (range - 1)` when range is a power of two, falling back to `x % range` otherwise. The sweep driver always uses `key_range = capacity`, so the mask path is taken in practice.

A subtler dependency lives in the **default hash function**. With a pow-2 mask, only the low bits of `hash(key)` reach `capacity_mask_` — the high bits are thrown away. If a user supplied an identity hash on structured keys (e.g. dense integer IDs), low-bit clustering would map directly into slot clustering and probe lengths would explode. The default `default_hash<uint32_t>` is `fmix32`, a finalizer-style mixer that spreads bits well, so the low bits of `hash(k)` are pseudo-uniform even when keys are not. Users supplying their own `Hash` should preserve this property; passing a non-mixing hash with a pow-2 capacity is the textbook way to break a fast hash table.

## Reduction operators

When `View::insert` encounters an existing entry for the same key, the new value is combined with the existing one through a `Reduction` functor — a template parameter of `HashTable`, defaulting to `gpurhh::replace_op`.
Stateless and trivially copyable, the functor sits inside the `View` by value:

```cpp
struct replace_op {
    template <class T>
    __device__ T operator()(T /*existing*/, T incoming) const noexcept { return incoming; }
};

struct sum_op {
    template <class T>
    __device__ T operator()(T existing, T incoming) const noexcept { return existing + incoming; }
};
```

`replace_op` gives last-writer-wins semantics. `sum_op` accumulates values for repeated-key inserts — useful as a parallel histogram or counter primitive. Users can define their own functor with the same signature.

For commutative and associative reductions like `sum_op`, the final stored value is fully deterministic regardless of how the CAS retries interleave across tiles: the result is just the reduction of all inserted values for that key. For non-commutative reductions, the outcome depends on which writers win their CAS races.

## Parallel insertion algorithm

Each thread owns one `(key, value)` pair.
The thread walks the probe sequence and, at every slot, performs one of three actions:

1. **Empty slot, CAS succeeds.** Done.
2. **Same key, CAS succeeds.** This is an update; done.
3. **Occupied by another key.**
   Compare probe distances.
   If the resident is "richer" (its probe distance from *its* home is smaller than ours from *our* home), we attempt to CAS our pair into the slot.
   On success, we adopt the displaced pair and continue probing from this slot.
   If the resident is "poorer" or equal, we move on.

A CAS failure is treated as "the world changed under us": we re-read the slot and re-evaluate without advancing the probe index.
This is the standard lock-free retry.

### Why this is correct under contention

The Robin Hood invariant is maintained by every individual CAS because each CAS only swaps in a pair with probe distance strictly greater than the one already in the slot (or the slot was empty).
Two concurrent inserts targeting the same slot serialize through CAS, and whichever loses retries with the freshest state.
There is no global ordering requirement: as long as every write respects the local invariant, the final table is a valid Robin Hood layout for *some* permutation of the input.

Caveat: the *value* portion of an update (case 2) needs care.
If two threads insert the same key concurrently and the user wants "last write wins" semantics, we need an additional CAS loop on the value, or we tolerate non-deterministic resolution.

## Achieving bandwidth saturation

The 64-bit slot is small relative to a 128-byte L2 cache line: one transaction brings 16 slots.
The bandwidth strategy follows from this:

- **Warp-cooperative probing.**
  Within a warp, threads whose home slots land in the same cache line should service their probes from one load, not 32 loads.
  We have the warp ballot/shuffle primitives to do this cheaply.
- **Bucketed probing.**
  Treat the table as an array of *buckets* of 16 slots (one cache line).
  A probe inspects an entire bucket at a time — one coalesced 128-byte load — and uses warp intrinsics (`__ballot_sync`, `__match_any_sync`) to find the candidate slot.
  Insertion still uses CAS on a single 8-byte slot inside the bucket, but the read traffic per probe is exactly one cache line.
  This is the same idea as `bucketed_cuckoo` and Bucketized Cuckoo Hashing.
- **Probe length capping.**
  Exposed as a `MaxProbeBuckets` template parameter (default 8).
  An insert that would place a key past the cap fails (the caller is expected to rehash); a get that probes the cap without finding the key returns "not found" (correct because of the insert-side cap).
  With Robin Hood at load factor ~0.85 and `bucket_size` 16, the expected longest probe is well under the default cap; the cap protects against adversarial inputs and very high load factors.
  Empirically, on uniform-random keys with α=1, the default cap-8 budget handles input load factor F ≤ 2 (≈ 80% slot occupancy) with zero failures, and starts losing about 0.1% of inserts at F = 3 (≈ 95% occupancy) — a real but small number that callers can either tolerate, drop (cf. lossy-storage workloads), or recover by buffering the leftover Slots and replaying into a larger table.
  Crucially, a failing insert hands the leftover `(key, value)` back to the caller via the return value rather than dropping it on the floor.
  This matters because the leftover may not be the original input: if Robin Hood displacements happened during the probe, the originally-passed-in pair is already in the table and the leftover is the most-recently-evicted victim.
  Returning it is what makes the operation lossless under failure — callers can buffer leftovers and replay them into a rebuilt larger table.
- **Avoid divergence.**
  All threads in a warp execute the same probe loop.
  The "I'm done, others aren't" case is handled with masked operations rather than early-return-then-divergence.

If we get this right, an insert of `N` keys at load factor 0.7 should cost approximately `(1/(1-α)) * sizeof(slot)` bytes per insert in expectation, which is the information-theoretic floor for an open-addressed table with this layout.

## Work decomposition: how insertions map onto the GPU hierarchy

Before discussing parameters, we need to commit to a unit of work — what GPU resource cooperates on a single key insertion.
There are four candidates:

| Mapping              | Threads / key | Probe is one coalesced load? | Keys in flight / warp | Wasted lanes / probe |
| -------------------- | ------------- | ---------------------------- | --------------------- | -------------------- |
| Thread per key       | 1             | No — 32 unrelated lines      | 32                    | 0                    |
| **Sub-warp tile**    | **B**         | **Yes — exactly one line**   | **32 / B**            | **0 (if B ∣ 32)**    |
| Warp per key         | 32            | Yes, but ≤ B lanes used      | 1                     | 32 − B               |
| Block per key        | 128–1024      | Yes, but lanes mostly idle   | ≪ 1                   | huge                 |

- **Thread-per-key** is the naive choice and exactly the case our bandwidth strategy is designed to avoid.
  Each warp issues 32 cache-line loads per probe step instead of one, and we lose any coalescing benefit.
  Fatal for our stated goal.
- **Block-per-key** is too coarse: there is no useful work for 128+ threads on a single key.
- **Warp-per-key** is the classic "warp-cooperative hashing" design but wastes lanes whenever the bucket is narrower than the warp.
  With 8-byte slots and a 128-byte line on NVIDIA, the bucket holds 16 slots, so half of every warp sits idle.
- **Sub-warp tile per key**, with tile width equal to the bucket width, is the one design where *one tile-probe = one cache-line transaction* with no idle lanes.
  This is the design we adopt.

### The structure that follows

1. A **bucket** is a contiguous, cache-line-aligned run of `B` slots.
2. A **cooperative tile** of `B` threads (`cooperative_groups::thread_block_tile<B>`) owns one in-flight key at a time.
   Each tile lane holds one slot of the current bucket in a register; one bucket probe is one coalesced load.
3. **Tiles per warp** = `WarpSize / B`.
   For NVIDIA with 8-byte slots that is 2.
4. **Block size** is purely an occupancy knob — some multiple of `WarpSize`, tuned empirically.
5. **Work assignment** is a tile-strided loop: tile index `t` (across the whole grid) handles input keys `t, t+T, t+2T, …` where `T` is the total number of resident tiles.
   Naturally load-balanced and robust to input skew.

### Per-tile probe step

On each probe:

1. The `B` lanes cooperatively load one bucket — each lane reads one slot, fully coalesced.
2. `tile.ballot(...)` over a predicate identifies empty slots, key matches, or Robin-Hood-displaceable slots in a single warp intrinsic.
3. The lane corresponding to the chosen slot performs the CAS; the outcome is broadcast to the rest of the tile with `tile.shfl(...)`.
4. On displacement, the evicted `(key, value)` becomes the tile's new in-flight pair and the tile advances by one bucket.

Every step is a warp-intrinsic operation on a single cache line, which is what makes the bandwidth argument hold.

### Parameters: what is forced by the architecture, what is free

Once we commit to the sub-warp tile design, almost every constant is *derived* from the hardware, not chosen:

| Parameter          | Value (default)              | Source                                   |
| ------------------ | ---------------------------- | ---------------------------------------- |
| `SlotBytes`        | 8 (4+4) or 16 (8+8)          | Key/value packing choice                 |
| `CacheLineBytes`   | 128 (NVIDIA)                 | Hardware                                 |
| `BucketSize`       | `CacheLineBytes / SlotBytes` | **Derived**                              |
| `WarpSize`         | 32 (NVIDIA)                  | Hardware                                 |
| `TileSize`         | `BucketSize`                 | **Derived** (the central design choice)  |
| `TilesPerWarp`     | `WarpSize / BucketSize`      | **Derived**                              |
| `BlockSize`        | tuned (e.g. 128 or 256)      | Occupancy — empirical                    |
| `TargetLoadFactor` | tuned (e.g. 0.7)             | Robin Hood quality — empirical           |
| `MaxProbeBuckets`  | 8                            | Worst-case time vs. insert-failure rate  |

So the only genuinely free knobs are `BlockSize`, `TargetLoadFactor`, and `MaxProbeBuckets`.
The rest should be expressed in the code as compile-time constants derived from `SlotBytes`, `CacheLineBytes`, and `WarpSize`, with the latter two being template parameters of the table (with sensible defaults) rather than hard-coded.

### Why the abstraction matters — an AMD aside (out of scope for v1)

AMD GPUs (CDNA, RDNA) have a 64-byte cache line and a 64-thread wavefront.
With 8-byte slots:

- NVIDIA: `BucketSize` = 128 / 8 = 16, `TilesPerWarp` = 32 / 16 = **2**.
- AMD:    `BucketSize` =  64 / 8 =  8, `TilesPerWarp` = 64 /  8 = **8**.

The same algorithm runs unchanged; only the derived constants differ.
We are not targeting AMD, but it is the reason we want `CacheLineBytes` and `WarpSize` to be parameters of the table type rather than hard-coded `128` and `32`.
Anything that ends up baked in as a literal in the implementation is a portability bug waiting to happen.

## Lookup

`View::get` follows the same probing rule as insert, read-only, returning `cuda::std::optional<Value>` — populated on hit, empty on miss. The probe terminates as soon as it sees one of:

- A key match → return `{value}`.
- An empty slot → return `nullopt`. Robin Hood would have placed the key before this point.
- A resident "richer" than ours (smaller probe distance) → return `nullopt`. Our key would have evicted that resident if it were in the table, contradiction.
- The probe cap → return `nullopt`. The insert-side cap means the key cannot live further than this.

Lookups are read-only and trivially parallel: concurrent lookups never race each other.

**Concurrent get during insert is not linearizable.** A get running simultaneously with an insert can return `nullopt` for a key that is about to be inserted, or for a key in the middle of being displaced. Linearizability would require per-slot versioning. Most workloads sync between insert and get phases.

## Deletion

Not supported. The natural design is backward-shift deletion, but a correct lock-free implementation is significantly harder than insert/lookup, and the use case (this is an append-only table for most consumers) hasn't pushed us to do it.

## Public API

```cpp
namespace gpurhh {

// Named alias for the CUDA runtime's default stream. Used as the
// default argument for host-side methods that issue async work.
inline constexpr cudaStream_t default_stream = 0;

template <class Key,
          class Value,
          class Hash             = default_hash<Key>,
          Key   EmptyKey         = default_empty_key<Key>::key,
          class Reduction        = replace_op,
          int   CacheLineBytes   = 128,
          int   WarpSize         = 32,
          int   MaxProbeBuckets  = 8>
class HashTable {
public:
    // Host-side resource management. Non-copyable, movable.
    explicit HashTable(std::size_t min_capacity);
    ~HashTable();
    HashTable(HashTable&&) noexcept;
    HashTable& operator=(HashTable&&) noexcept;

    // Empty the table — every slot reset to (empty_key, _). Async on
    // the given stream; caller is responsible for synchronizing before
    // any operation that depends on the cleared state.
    void clear(cudaStream_t stream = default_stream);

    // Device-side handle. Trivially copyable; pass by value into kernels.
    class View {
    public:
        // Returns nullopt on success; on probe-cap failure returns the
        // leftover Slot that could not be placed (may be the original
        // pair, or a Robin Hood victim displaced before the failure).
        //
        // The `tile_counter` parameter appears only when
        // GPURHH_BENCHMARK_COUNTERS is defined. Lane 0 of the tile
        // increments it on each bucket-load; the caller is expected
        // to pass a per-tile register accumulator and flush it to a
        // per-tile global slot at end of kernel.
        template <class Tile>
        __device__ cuda::std::optional<Slot>
        insert(const Tile& tile, Key key, Value value
#ifdef GPURHH_BENCHMARK_COUNTERS
            , std::uint32_t& tile_counter
#endif
        );

        template <class Tile>
        __device__ cuda::std::optional<Value>
        get(const Tile& tile, Key key
#ifdef GPURHH_BENCHMARK_COUNTERS
            , std::uint32_t& tile_counter
#endif
        ) const;
    };

    View view() const noexcept;

    // Actual capacity, in slots, rounded up to the next power of two and
    // never less than `bucket_size`.
    std::size_t capacity() const noexcept;

    // Direct pointer access to the device-resident bucket array.
    // Provided for tests, diagnostics, and benchmark instrumentation —
    // the `data()` name (matching the `std::vector::data` convention)
    // and the comment in the header are enough warning that callers
    // are bypassing the table's concurrency contract.
    Bucket*       data() noexcept;
    const Bucket* data() const noexcept;
};

// Diagnostic helper. Lives in <gpurhh/print.cuh>, a separate header so
// the core library stays independent of <cstdio> and <vector>. Reads
// the bucket array via HashTable::data().
template <class Table>
void print_slots(const Table& table, std::size_t start, std::size_t stop);

}  // namespace gpurhh
```

### Opt-in macros

One macro gates functionality that the core library doesn't expose by default. It must be defined in the translation unit *before* any `<gpurhh/...>` header is included:

- **`GPURHH_BENCHMARK_COUNTERS`** adds a trailing `std::uint32_t& tile_counter` parameter to `View::insert` and `View::get`. Lane 0 of the tile increments it on every bucket-load (a CAS-retry re-read counts too — it's a real DRAM transaction). The caller maintains a per-tile register accumulator and writes it out once at end of kernel; no atomics are involved in the probing hot loop. The memory-bandwidth benchmark binaries enable this; the timing benchmarks, tests, and the example do not.

The `View` is the centerpiece for header-only use: a user kernel that already has its keys in registers calls `view.insert(tile, ...)` or `view.get(tile, ...)` directly.
Bulk host-side operations are deliberately *not* part of the table — callers manage their device buffers and write their own driver kernels.
The tests in `tests/kernels.cuh` and `tests/isolated.cuh` show the canonical pattern; `examples/basic.cu` is a minimal standalone walkthrough.

## Testing

The test suite lives in `tests/`, with each `test_*.cu` compiled into its own executable.

- **`test_construction.cu`** — construct / destruct, move semantics, edge sizes (capacity == bucket_size, capacity below bucket_size), moved-from state behavior, and `clear()` returning the bucket array to the empty-byte pattern.
- **`test_combined.cu`** — end-to-end insert + get round-trips through the public API.
- **`test_get.cu`** — `View::get` in isolation. Pre-states are hand-built with `IdentityHash` so the probe sequence and home buckets are deterministic; the test then verifies `get`'s match / empty / richer-resident / wrap-around behavior.
- **`test_insert.cu`** — `View::insert` in isolation. Some tests use hand-computed expected layouts; all run a Robin Hood invariant scanner over the final state as a safety net. Covers the probe-cap failure path and concurrent same-key inserts.
- **`test_concurrency.cu`** — heavy parallel contention: adversarial displacement chains, many concurrent gets, mixed new / duplicate inserts, parallel saturation of the probe cap.
- **`test_randomized.cu`** — randomized stress. Uniform-key inserts at small (4 KB table) and gigabyte (1 GB table) scales; a sum-reduction stress test with random keys and deliberate duplicates whose expected per-key sum is computed on the host.

Three shared infrastructure headers:

- **`tests/tests.cuh`** — the `CUDA_CHECK` postfix error-check operator (`expr >> CUDA_CHECK`).
- **`tests/kernels.cuh`** — the `Table` alias (with `default_hash`), bulk insert/get host helpers, and end-to-end driver kernels. Used by the "combined" and construction tests.
- **`tests/isolated.cuh`** — the `TestTable` alias (with `IdentityHash` for predictable home buckets), templated `run_insert` / `run_get` / `run_get_one` / `run_insert_with_outcomes`, the `assert_robin_hood_invariant` scanner, and `set_state` / `read_state` for direct table-memory manipulation via `HashTable::data()`.

The `IdentityHash` is the enabling choice for the isolated tests: with `hash(K) = K`, the home bucket of a key is fully predictable from its value, so we can construct any valid Robin Hood state with one `cudaMemcpy` and hand-predict the outcome of small insert sequences. The randomized tests apply a bit-mixing function (the same `fmix32` finalizer used by `default_hash`) as a permutation on sequential integers — this gives distinct random-looking keys without the host-side dedup overhead that an `unordered_set` would require, which is what makes the gigabyte-scale tests fit in memory.

The test suite runs about 35 tests across ~1100 lines of code, against ~350 lines of library code. The high ratio is deliberate: this is a concurrent data structure, and the failure modes for concurrent CAS retries, displacement chains, and probe-cap interactions are subtle enough that we put significant effort into checking each path.

## Benchmarking

Throughput and instrumentation measurements live in `benchmarks/`, split into two studies with their own directory subtree and their own CSV output tree.

- **`benchmarks/timing/`** — apples-to-apples kernel throughput against optional baselines (`cuCollections::static_map`, `WarpCore::SingleValueHashTable`). gpurhh's binaries here use `replace_op` reduction and `MaxProbeBuckets = 1 << 20` (effectively uncapped), matching the baselines' unbounded-probe semantics. Two executables: `benchmark_insert`, `benchmark_get`. Baseline equivalents live under `benchmarks/timing/baselines/{cuco,cuco_dh,warpcore}/` and append to the same per-workload CSVs; the `library` column distinguishes their rows. The `cuco/` baseline uses cuCollections's default linear probing (CG=4); the `cuco_dh/` baseline reuses the same headers but specializes `static_map` with `double_hashing<8, ...>`, isolating the probing-scheme variable within a single library so the family-vs-implementation axis is testable separately.
- **`benchmarks/memory_bandwidth/`** — the bandwidth study. `benchmark_memcpy` provides the empirical DRAM ceiling; `benchmark_insert` and `benchmark_get` are gpurhh's design under instrumentation (`sum_op` reduction, default `MaxProbeBuckets = 8`, per-tile probe / failure / hit counters via `GPURHH_BENCHMARK_COUNTERS`). The CSVs from this study let you compute precise per-op DRAM traffic — `total_probes × sizeof(Bucket) / time_ms` — and compare it against memcpy's ceiling. No library baselines, since neither cuco nor warpcore exposes the per-probe counters needed for the bandwidth analysis.

The split exists because the two studies want fundamentally different configurations: timing needs the most permissive probe behavior to make per-library performance comparable; the bandwidth study wants the design's actual default (capped probes, real reduction work) so we can read the probe / failure / hit characteristics back out and convert them into bandwidth numbers.

`benchmark_memcpy` is the peak-bandwidth reference. Times `cudaMemcpyAsync(... D2D)` and a trivial `uint4`-vectorized copy kernel on a buffer of configurable size. The CSV stores `dram_bytes = 2 * payload_bytes` (one read of src + one write of dst), so the downstream bandwidth computation is simply `dram_bytes / time_ms` — and this is directly comparable to the hash-table benchmarks' `total_probes × sizeof(Bucket) / time_ms`, since both metrics are bytes through the DRAM controller per second. Recording the 2× factor at write time rather than relying on downstream scripts to apply it keeps the convention impossible to misuse in analysis.

### Workload knobs

- **`capacity`** is held at 1 GiB by default — well past the L2 cache on any sm_89 device, putting the kernel firmly in the DRAM-bound regime.
- **Key distribution** is raw `Uniform(0, 2^32)` — every uint32 is equally likely, no clamping. With capacity = 2^27, this gives `α = key_range / capacity ≈ 32`: keys are effectively unique. This is the collision-resolution regime where table designs actually differ; at much smaller α the table fills very slowly with duplicates and linear probing's pathological clustering never engages.
- **`F = n_ops / capacity`** is the swept axis. The grid is `F ∈ {0.5, 0.7, 0.85, 0.95, 1.0, 1.5, 2.0, 3.0}`. With α ≈ 32, effective occupancy ≈ F up to ~0.97, so F = 0.9–1.0 is the high-occupancy regime where Robin Hood's variance reduction matters; F > 1 over-subscribes the table (more unique keys than slots) and the sweep measures how each library degrades — both throughput (`time_ms`) and how many of those keys actually landed (`n_unique` and `drops`).
- **`block_size ∈ {64, 128, 256, 512, 1024}`** is swept for the gpurhh binaries to spot-check that launch shape doesn't dominate. cuco / warpcore pick their own internal launch shapes.

### Experiment phases

Every insert benchmark structures its work via `run_benchmark_loop`, which carves each rep into three callbacks. **Only the middle phase (`launch`) sits inside the timing window** — `EventTimer::begin()` / `end_ms()` bracket exactly that callback's content, nothing more.

**Once at startup (untimed, before the loop).**
- Allocate `d_keys`, `d_values`, and `d_sorted` device buffers, each `n_ops × uint32`.
- `cudaMemset(d_values, 0, ...)` once. Under `replace_op` the value content doesn't affect kernel work, so a zero buffer suffices; gpurhh, cuco, and warpcore all do this so the kernel pays an apples-to-apples `d_values` read.
- Construct the table (`gpurhh::HashTable`, `cuco::static_map`, or `warpcore::SingleValueHashTable`).
- Construct a single `UniformKeyGenerator` (cuRAND Philox4_32_10) seeded with `--seed`.
- For cuco: build a `thrust::counting_iterator → cuco::pair{d_keys[i], d_values[i]}` transform iterator for the bulk-insert call.

**`setup` callback (untimed, runs before each warmup and each timed rep).**
- Empty the table (`table.clear()` / `map.clear_async()` / `table.init()`).
- Generate fresh input keys: `fill_uniform_keys(d_keys, n_ops, gen)` advances the cuRAND stream by `n_ops` draws. Across reps the generator state continues, so the 16 reps see iid draws rather than 16 timings of one frozen input — statistically equivalent to 16 independent seeds but with CUDA / table / generator setup paid once.
- Count distinct inputs for drop accounting: copy `d_keys → d_sorted`, `thrust::sort`, `thrust::unique`, record `n_unique`. This is the ground-truth denominator for the drops measurement: at α ≈ 32 most inputs are unique but ~1–2% are birthday-collision duplicates, and we don't want input duplicates to show up as drops.

**`launch` callback (timed window).**
- One bulk insert: gpurhh's grid-stride insert kernel, cuco's `map.insert_async(pair_begin, pair_begin + n_ops)`, or warpcore's `table.insert(d_keys, d_values, n_ops)`. That's it.

**`after` callback (untimed, runs after each timed rep).**
- Count occupied slots in the table:
  - gpurhh: `count_occupied_slots(reinterpret_cast<Slot*>(table.data()), capacity, empty_key)` — a `thrust::count_if` scan in `benchmarks.cuh`. We use the table's `data()` accessor to view the bucket storage as a flat slot array.
  - cuco: `map.size()` (synchronous, built-in).
  - warpcore: `table.size()` (synchronous, built-in).
- Compute `drops = n_unique - occupied`. This is the count of keys the library tried to insert and couldn't fit, with input duplicates already factored out via `n_unique`.
- Emit one CSV row with `time_ms`, `n_unique`, `drops`, and the configuration columns.

The get benchmark follows the same phase structure with one difference: the pre-fill is one-shot (untimed, before the loop) rather than per-rep, since pre-fill input variability is a second-order effect for the get throughput we're measuring. Only the get-key stream is refreshed in `setup` per rep.

### Per-tile probe / failure / hit counters

The `GPURHH_BENCHMARK_COUNTERS` macro (see ## Public API → Opt-in macros) adds a trailing `std::uint32_t&` parameter to `View::insert` / `View::get`. The memory-benchmark kernels declare a per-tile register accumulator, pass it through every `view.insert` / `view.get` call in the grid-stride loop, and flush one value per tile to a global array at end of kernel. No atomics fire in the probe hot loop. The host sums per-tile values into a grand total per rep.

The memory-bandwidth CSVs (`memory_bandwidth/insert.csv`, `memory_bandwidth/get.csv`) add `total_probes` and `total_failures` (insert) or `total_probes`, `total_hits`, `total_misses` (get) on top of the timing CSV schema. Downstream analysis derives metrics like average probe length (`total_probes / n_ops`) and bandwidth-corrected throughput (`total_probes × sizeof(Bucket) / time_ms`), the latter being directly comparable to the memcpy ceiling in the same directory.

At low F the corrected and uncorrected throughput numbers converge (average probe length ≈ 1). At high F the corrected number diverges upward — that gap is itself a useful signal about how hard Robin Hood is working.

### Sweep driver

`scripts/benchmark.sh` runs the full grid. Flags select which study to run:

- `--timing` — gpurhh's timing sweep (insert + get).
- `--memory-bandwidth` — the bandwidth sweep (memcpy ceiling + counter-instrumented gpurhh insert + get).
- `--cuco` — cuCollections baseline, default linear probing (timing only).
- `--cuco-dh` — cuCollections baseline re-specialized with double hashing (timing only).
- `--warpcore` — WarpCore baseline (timing only).
- `--all` — every flag above; missing binaries are skipped with a notice rather than erroring.

Output goes to `output/<timestamp>/timing/{insert,get}.csv` and `output/<timestamp>/memory_bandwidth/{memcpy,insert,get}.csv`, with a `run_info.txt` sidecar (`nvcc --version`, `uname`, `nvidia-smi` GPU summary) and a `benchmark.log` transcript at the top of the output dir. The seed is fixed at 1 across the sweep; `reps = 16` per invocation, with the cuRAND stream refreshing keys before every rep so each rep is an independent input draw.

`make benchmarks` builds the binaries at `-O3` (the default for the benchmark pattern rule). The sweep is launched separately from `scripts/benchmark.sh` because runs are heavy. Numbers from unoptimized builds are meaningless, so the `-O3` default is effectively non-overridable.

## Known limitations

- **Deletion** is not supported. Users that need it allocate a fresh table and re-insert.
- **Resize** is not supported. Users that need growth allocate a bigger table and re-insert.
- **Multi-GPU** is not supported.
- **Concurrent get during insert is not linearizable.** A get running simultaneously with insert can return `nullopt` for a key in flight. Workloads should synchronize between insert and get phases.
- **128-bit slot performance on pre-Hopper hardware** (sm_70–sm_89) is bottlenecked by libcu++'s lock-table emulation of 128-bit CAS. Functionally correct, but considerably slower than the native single-instruction path that sm_90+ has.

## Out of scope

- **Non-NVIDIA backends.** The abstractions (`CacheLineBytes` and `WarpSize` as table-level template parameters) don't preclude AMD support, but we haven't written or tested it. The same algorithm with AMD's 64-byte cache line and 64-thread wavefront would give `BucketSize = 8` and `TilesPerWarp = 8`.
- **Full stream support.** `clear()` takes a `cudaStream_t` (defaulting to `gpurhh::default_stream`). The constructor does not — `cudaMalloc` is synchronous and dominates the construction cost, so threading a stream through the slot-init memset would buy nothing while leaving the constructor's overall behavior confusingly half-async. The destructor doesn't take a stream either; destruction is a host-side lifecycle event, and the caller is expected to synchronize relevant streams before letting the table go out of scope.

## Future work

- Validate the 128-bit slot path on a Hopper (sm_90+) system; benchmark it against the 64-bit path.
- Benchmark against `cuCollections` and `WarpCore` on the same workloads to validate the bandwidth-bound claim. Pinned commits for both are inlined in `scripts/setup-baselines.sh`, which clones each upstream with a sparse-checkout of just the `include/` subtree into `external/<lib>/`. `external/` itself is gitignored, so downstream users of `gpurhh` who recursively clone aren't forced to pull benchmark-only dependencies.
- Add a CPU reference Robin Hood for randomized differential testing if test signal demands it (currently considered low priority — the invariant scanner plus the sum-reduction equivalence check cover most of what differential testing would add).
- Implement deletion (backward-shift, lock-free) if a use case appears.
