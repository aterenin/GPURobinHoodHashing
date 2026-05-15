# GPU Robin Hood Hash Table — Design Brainstorm

## Goals

- **Header-only CUDA library.**
  A single `.cuh` (or a small set) that users `#include` and template-instantiate for their key/value types.
  No separate compilation of the table itself.
- **Massively parallel insertion.**
  A batch of `N` keys is inserted by `N` (or more) GPU threads cooperatively.
  Worst-case behaviour must remain bounded even under heavy contention.
- **Memory-bandwidth bound.**
  On large tables, the table is much bigger than the L2 cache, so performance is determined by how many useful bytes each DRAM transaction delivers.
  The aim is to come as close to peak HBM bandwidth as the access pattern allows.

## Language and toolchain

- **Target language: C++20.**
- **Target CUDA toolkit: CUDA 13.**

Within the implementation, we stick to C++17-shape idioms by default and only reach for C++20 features where they demonstrably improve the code.
Today that is one use: `std::bit_cast` for the hash function's bit reinterpret, which replaces the `memcpy` idiom common in pre-C++20 GPU code.
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
struct Slot {                       // 8 bytes total — fits in atomicCAS<uint64_t>
    uint32_t key;
    uint32_t value;                 // or a 32-bit payload / pointer index
};
```

Empty slots use a reserved sentinel key (e.g. `0xFFFFFFFF`); users are forbidden from inserting that key.
For 64-bit keys or larger payloads we have two options:

1. **128-bit slots** with `__int128` CAS via `atomicCAS` on `unsigned long long` pairs — supported on sm_70+ but with reduced throughput.
2. **Separate parallel arrays** (`keys[]` and `values[]`) with the key array being the source of truth for CAS, and values being written only by the thread that successfully claimed the slot.
   This halves the bytes-per-CAS but requires care: a finder must re-read the key after reading the value to make sure the slot wasn't reused mid-read.

We will start with the 64-bit packed slot and parameterise later.

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
Two concurrent inserts targeting the same slot serialise through CAS, and whichever loses retries with the freshest state.
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
  Cap probes at, say, 32 buckets.
  If a thread overruns the cap during insertion, mark the batch as failed and rehash.
  With Robin Hood at 0.7 load this happens with vanishing probability for reasonable `N`.
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

| Parameter         | Value (default)              | Source                                   |
| ----------------- | ---------------------------- | ---------------------------------------- |
| `SlotBytes`       | 8                            | Key/value packing choice                 |
| `CacheLineBytes`  | 128 (NVIDIA)                 | Hardware                                 |
| `BucketSize`      | `CacheLineBytes / SlotBytes` | **Derived**                              |
| `WarpSize`        | 32 (NVIDIA)                  | Hardware                                 |
| `TileSize`        | `BucketSize`                 | **Derived** (the central design choice)  |
| `TilesPerWarp`    | `WarpSize / BucketSize`      | **Derived**                              |
| `BlockSize`       | tuned (e.g. 128 or 256)      | Occupancy — empirical                    |
| `TargetLoadFactor`| tuned (e.g. 0.7)             | Robin Hood quality — empirical           |

So the only genuinely free knobs are `BlockSize` and `TargetLoadFactor`.
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
  Lookup is fully read-only and trivially parallel — no synchronisation required.
- **Deletion.**
  Backward-shift: starting from the deleted slot, while the next slot is non-empty and its probe distance is positive, shift it back by one.
  On a concurrent GPU this is the hardest operation to make correct without locks.
  **Initial scope: defer deletion**, support insert + lookup only, and add it once we have benchmarks in place.

## API sketch

```cpp
namespace gpurhh {

template <class Key,
          class Value,
          class Hash = default_hash<Key>,
          Key   EmptyKey = Key{} - Key{1}>            // sentinel
class HashTable {
public:
    // Host-side construction allocates device memory.
    explicit HashTable(std::size_t capacity);

    // Bulk operations. Stream-aware, non-owning input buffers.
    void insert(const Key* keys, const Value* values,
                std::size_t n, cudaStream_t stream = 0);

    void get(const Key* keys, Value* values_out,
             std::size_t n, cudaStream_t stream = 0) const;

    // A device-side, copyable handle for use inside user kernels.
    struct View {
        __device__ bool  insert(Key k, Value v);
        __device__ bool  get   (Key k, Value& out) const;
    };
    View view();

private:
    Slot*       slots_;
    std::size_t capacity_;
};

} // namespace gpurhh
```

The `View` is the centrepiece for header-only use: a user kernel that already has its keys in registers can call `view.insert(...)` directly without a separate kernel launch.

## Open questions / things to settle before coding

1. **Slot width.**
   Start with 64-bit packed `(uint32 key, uint32 value)`, or go straight to a 128-bit slot for 64-bit keys?
   The 64-bit case is the canonical micro-benchmark and is what most published GPU hash tables use.
   Note that changing this also changes the derived `BucketSize`.
2. **Failure semantics on full table.**
   Return a count of failed inserts, or `assert`?
   A bulk-insert API can naturally return the number of survivors and let the caller resize.
3. **Resize.**
   Out of scope for v1.
   The table is allocated at a fixed capacity.
   Users that need growth allocate a bigger one and re-insert.
4. **Multi-GPU.**
   Out of scope for v1.
5. **Portability across vendors.**
   Out of scope for v1 but informs the abstractions — see the AMD aside above.
   `CacheLineBytes` and `WarpSize` should be table-level parameters, not magic numbers in the implementation.
6. **Testing.**
   A small CPU reference implementation (sequential Robin Hood) plus a randomised differential test against the GPU implementation, plus stress tests at 0.5/0.7/0.9 load.

## Suggested next steps

1. Sketch the `Slot`, hash, and probe primitives in a single `.cuh`.
2. Implement the insert kernel without bucketing, just to validate correctness and the CAS loop.
3. Add the bucketed (cache-line-aligned) variant and benchmark against the naive one.
4. Add `get`, then benchmark against `cuCollections` / `WarpCore` as external baselines.
