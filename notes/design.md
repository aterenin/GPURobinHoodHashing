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

    // Device-side handle. Trivially copyable; pass by value into kernels.
    class View {
    public:
        template <class Tile>
        __device__ bool insert(const Tile& tile, Key key, Value value);

        template <class Tile>
        __device__ cuda::std::optional<Value> get(const Tile& tile, Key key) const;
    };

    View view() const noexcept;

    // Actual capacity, in slots, rounded up to the next power of two and
    // never less than `bucket_size`.
    std::size_t capacity() const noexcept;

    // Diagnostic: pretty-print a slot range to stdout. Bucket boundaries
    // are shown; empty slots are labeled "empty".
    void print_slots(std::size_t start, std::size_t stop) const;

#ifdef GPURHH_ENABLE_INTERNAL_ACCESS
    // Direct pointer access to the device-resident bucket array. Gated
    // behind a macro so users must opt in explicitly; the test suite does.
    Bucket*       data() noexcept;
    const Bucket* data() const noexcept;
#endif
};

}  // namespace gpurhh
```

The `View` is the centerpiece for header-only use: a user kernel that already has its keys in registers calls `view.insert(tile, ...)` or `view.get(tile, ...)` directly.
Bulk host-side operations are deliberately *not* part of the table — callers manage their device buffers and write their own driver kernels.
The tests in `tests/kernels.cuh` and `tests/isolated.cuh` show the canonical pattern; `examples/basic.cu` is a minimal standalone walkthrough.

## Testing

The test suite lives in `tests/`, with each `test_*.cu` compiled into its own executable.

- **`test_construction.cu`** — construct / destruct, move semantics, edge sizes (capacity == bucket_size, capacity below bucket_size), moved-from state behavior.
- **`test_combined.cu`** — end-to-end insert + get round-trips through the public API.
- **`test_get.cu`** — `View::get` in isolation. Pre-states are hand-built with `IdentityHash` so the probe sequence and home buckets are deterministic; the test then verifies `get`'s match / empty / richer-resident / wrap-around behavior.
- **`test_insert.cu`** — `View::insert` in isolation. Some tests use hand-computed expected layouts; all run a Robin Hood invariant scanner over the final state as a safety net. Covers the probe-cap failure path and concurrent same-key inserts.
- **`test_concurrency.cu`** — heavy parallel contention: adversarial displacement chains, many concurrent gets, mixed new / duplicate inserts, parallel saturation of the probe cap.
- **`test_randomized.cu`** — randomized stress. Uniform-key inserts at small (4 KB table) and gigabyte (1 GB table) scales; a sum-reduction stress test with random keys and deliberate duplicates whose expected per-key sum is computed on the host.

Three shared infrastructure headers:

- **`tests/tests.cuh`** — the `CUDA_CHECK` postfix error-check operator (`expr >> CUDA_CHECK`) and the `GPURHH_ENABLE_INTERNAL_ACCESS` macro, which gates `HashTable::data()`.
- **`tests/kernels.cuh`** — the `Table` alias (with `default_hash`), bulk insert/get host helpers, and end-to-end driver kernels. Used by the "combined" and construction tests.
- **`tests/isolated.cuh`** — the `TestTable` alias (with `IdentityHash` for predictable home buckets), templated `do_insert` / `do_get` / `do_insert_with_outcomes`, the `assert_robin_hood_invariant` scanner, and `set_state` / `read_state` for direct table-memory manipulation via the `data()` back-door.

The `IdentityHash` is the enabling choice for the isolated tests: with `hash(K) = K`, the home bucket of a key is fully predictable from its value, so we can construct any valid Robin Hood state with one `cudaMemcpy` and hand-predict the outcome of small insert sequences. The randomized tests apply a bit-mixing function (the same `fmix32` finalizer used by `default_hash`) as a permutation on sequential integers — this gives distinct random-looking keys without the host-side dedup overhead that an `unordered_set` would require, which is what makes the gigabyte-scale tests fit in memory.

The test suite runs about 34 tests across ~1100 lines of code, against ~350 lines of library code. The high ratio is deliberate: this is a concurrent data structure, and the failure modes for concurrent CAS retries, displacement chains, and probe-cap interactions are subtle enough that we put significant effort into checking each path.

## Known limitations

- **Deletion** is not supported. Users that need it allocate a fresh table and re-insert.
- **Resize** is not supported. Users that need growth allocate a bigger table and re-insert.
- **Multi-GPU** is not supported.
- **Concurrent get during insert is not linearizable.** A get running simultaneously with insert can return `nullopt` for a key in flight. Workloads should synchronize between insert and get phases.
- **128-bit slot performance on pre-Hopper hardware** (sm_70–sm_89) is bottlenecked by libcu++'s lock-table emulation of 128-bit CAS. Functionally correct, but considerably slower than the native single-instruction path that sm_90+ has.

## Out of scope

- **Non-NVIDIA backends.** The abstractions (`CacheLineBytes` and `WarpSize` as table-level template parameters) don't preclude AMD support, but we haven't written or tested it. The same algorithm with AMD's 64-byte cache line and 64-thread wavefront would give `BucketSize = 8` and `TilesPerWarp = 8`.
- **Stream support.** All operations currently run on the default stream. Adding a `cudaStream_t` parameter to host-side methods is a small change but hasn't been made.

## Future work

- Validate the 128-bit slot path on a Hopper (sm_90+) system; benchmark it against the 64-bit path.
- Benchmark against `cuCollections` and `WarpCore` on the same workloads to validate the bandwidth-bound claim.
- Add a CPU reference Robin Hood for randomized differential testing if test signal demands it (currently considered low priority — the invariant scanner plus the sum-reduction equivalence check cover most of what differential testing would add).
- Implement deletion (backward-shift, lock-free) if a use case appears.
