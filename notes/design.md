# GPU Robin Hood Hash Table — Design Brainstorm

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

## Lookup and deletion

- **Lookup.**
  Same probing rule, no CAS.
  Stop as soon as we find the key, an empty slot, or a resident whose probe distance is smaller than ours.
  Lookup is fully read-only and trivially parallel — no synchronization required.
- **Deletion.**
  Backward-shift: starting from the deleted slot, while the next slot is non-empty and its probe distance is positive, shift it back by one.
  On a concurrent GPU this is the hardest operation to make correct without locks.
  **Initial scope: defer deletion**, support insert + lookup only, and add it once we have benchmarks in place.

## API sketch

```cpp
namespace gpurhh {

template <class Key,
          class Value,
          class Hash             = default_hash<Key>,
          Key   EmptyKey         = default_empty_key<Key>::key,
          int   CacheLineBytes   = 128,
          int   WarpSize         = 32,
          int   MaxProbeBuckets  = 8>
class HashTable {
public:
    // Host-side: allocate and initialize the device-resident slot array.
    explicit HashTable(std::size_t min_capacity);
    ~HashTable();

    // Non-copyable, movable.
    HashTable(HashTable&&) noexcept;
    HashTable& operator=(HashTable&&) noexcept;

    // Device-side, copyable handle. Trivially passed by value into kernels.
    class View {
    public:
        template <class Tile>
        __device__ bool insert(const Tile& tile, Key key, Value value);

        template <class Tile>
        __device__ bool get(const Tile& tile, Key key, Value& out) const;
    };

    View view() const noexcept;
    std::size_t capacity() const noexcept;  // power of two, in slots
};

} // namespace gpurhh
```

The `View` is the centerpiece for header-only use: a user kernel that already has its keys in registers can call `view.insert(tile, ...)` directly.
Bulk host-side operations are deliberately *not* part of the table — callers manage their device buffers and write their own driver kernels, which calls `view.insert` / `view.get` cooperatively from a tile of `tile_size` threads.
The tests in `tests/kernels.cuh` show the canonical pattern.

## Open questions / things to settle

1. **Slot width.**
   Both 64-bit (4+4 byte) and 128-bit (8+8 byte) packings are supported via `cuda::atomic_ref<Slot>`.
   The 64-bit case is what current tests exercise; the 128-bit case compiles and links but uses libcu++'s lock-emulated 128-bit CAS on pre-Hopper GPUs, which is significantly slower than 64-bit CAS.
   Open: validate the 128-bit path on a Hopper system, and benchmark.
2. **Failure semantics on full table.**
   `View::insert` currently returns `false` when the probe-length cap is hit.
   Open: whether a bulk operation should return a survivor count or just a failure flag — depends on whether we add bulk operations to the API at all.
3. **Resize.**
   Out of scope for v1.
   The table is allocated at a fixed capacity.
   Users that need growth allocate a bigger one and re-insert.
4. **Multi-GPU.**
   Out of scope for v1.
5. **Portability across vendors.**
   Out of scope for v1 but informs the abstractions — see the AMD aside above.
   `CacheLineBytes` and `WarpSize` are table-level parameters, not magic numbers in the implementation.
6. **Testing.**
   Basic correctness tests exist (construction, single-tile insert, single-tile get).
   Still missing: a CPU reference for randomized differential testing; concurrency stress tests at varying load factors; probe-cap stress tests; benchmarks against `cuCollections` / `WarpCore`.

## Suggested next steps

1. Smoke-test the end-to-end pipeline on a real GPU (build, run `test_construction`, `test_insert`, `test_get`).
2. Wire up `examples/basic.cu` as a self-contained demonstration of construct + bulk insert + bulk lookup.
3. Add randomized differential tests against a CPU reference Robin Hood implementation.
4. Add stress tests for concurrency and load-factor edges.
5. Benchmark against `cuCollections` / `WarpCore` to validate the bandwidth-bound claim.
