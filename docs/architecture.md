# Architecture

## Why Robin Hood?

Robin-Hood refers to open addressing with the invariant *"the entry in slot `i` whose probe distance is smaller gets evicted by an entry whose probe distance is larger"*, which gives:

- A tight, low-variance distribution of probe distances → small worst-case lookup length even at load factors of 0.85–0.95.
- A natural early-exit rule for lookups (stop as soon as the slot's probe distance is smaller than the probe distance of the key we're searching for).
- Backward-shift deletion that avoids tombstones.

These properties matter even more on a GPU than on a CPU because the warp's tail latency is set by the longest probe sequence among its 32 lanes.

## Slot layout

A slot is the unit of an atomic update.
The simplest design that admits lock-free CAS on a single machine word is:

```cpp
struct alignas(sizeof(Key) + sizeof(Value)) Slot {
    Key   key;       // typically uint32_t or uint64_t
    Value value;     // typically the same width as Key
};
```

Empty slots use a reserved sentinel key (default: all-bits-set for unsigned integer Key); users are forbidden from inserting that key.
The packed slot supports two widths:

1. **64-bit slots** (4-byte key + 4-byte value): atomic CAS via `cuda::atomic_ref<Slot>`, which on every NVIDIA arch compiles down to a 64-bit `atom.cas`.
2. **128-bit slots** (8-byte key + 8-byte value): atomic CAS still goes through `cuda::atomic_ref<Slot>`; on sm_90+ this is the single-instruction `atom.cas.b128`. On sm_70–sm_89 this path is untested here and may be very slow or fail outright.

We use the same control flow for both widths — the only thing that changes is which atomic instruction `cuda::atomic_ref::compare_exchange_strong` resolves to.
We do not use a parallel-arrays layout (separate `keys[]` and `values[]`): it halves the per-CAS bytes but at the cost of an awkward second-read-the-key step for finders that complicates correctness.

### Probe distance

We do *not* store probe distance in the slot.
It is derivable as `(slot_index - hash(key)) mod capacity`.
Storing it would cost bits we'd rather give to the payload, and the derivation is a single subtract once we've read the slot.

If we ever wanted to explore a double-hashing variant, one could modify the scheme to store the probe distance as well — but this would require smaller-than-32-bit keys to keep the slot packable into a single CAS word.

## Achieving bandwidth saturation

The 64-bit slot is small relative to a 128-byte L2 cache line: one transaction brings 16 slots.
The bandwidth strategy follows from this:

- **Warp-cooperative probing.**
  Within a warp, threads whose home slots land in the same cache line should service their probes from one load, not 32 loads.
  We have the warp ballot/shuffle primitives to do this cheaply.
- **Bucketed probing.**
  Treat the table as an array of *buckets* of 16 slots (one cache line).
  A probe inspects an entire bucket at a time — one coalesced 128-byte load — and uses warp intrinsics (`__ballot_sync`, `__match_any_sync`) to find the candidate slot.
  Reads thus happen at cache-line granularity (one coalesced 128-byte load per probe), while writes are CAS on a single 8-byte slot inside the bucket — since no atomic CAS spans an entire cache line.
  This is the same idea as `bucketed_cuckoo` and Bucketized Cuckoo Hashing.
- **Probe length capping.**
  Exposed as a `MaxProbeBuckets` template parameter (default 8).
  Using this parameter, an insert that would place a key past the cap fails (the caller is expected to rehash); a get that probes the cap without finding the key returns "not found" (correct because of the insert-side cap).
  With Robin Hood at the design's target load factors of 0.85–0.95 and `bucket_size` 16, the expected longest probe is well under the default cap; the cap protects against adversarial inputs and over-subscription.
  Empirically, on `Uniform(0, 2^32)` uint32 keys (effectively unique inputs), the default cap-8 budget handles `F = n_ops / capacity` up to ~1 with essentially zero failures, and starts losing inserts as `F` grows past 1 into the over-subscription regime where insertion only works at all thanks to sum-reduction (see `benchmarks/memory_bandwidth/insert.csv`'s `total_failures` column).
  A failing insert hands the leftover `(key, value)` back to the caller via the return value in order to avoid dropping it silently — see [Usage: Lossless insert failure via returned slots](usage.md#lossless-insert-failure-via-returned-slots).
- **Avoid divergence.**
  All threads in a warp execute the same probe loop.
  The "I'm done, others aren't" case is handled with masked operations rather than early-return-then-divergence.
  Worth noting here that on NVIDIA hardware we cannot diverge by more than a factor of 2 thanks to sub-warp tile granularity.

In expectation, an insert of `N` keys at load factor α costs approximately `(1/(1-α)) * sizeof(slot)` bytes per insert — the information-theoretic floor for an open-addressed table with this layout. At the design's target range of 0.85–0.95 the bytes/insert is 7–20× the slot size, and the bandwidth advantage over simpler schemes should be most pronounced precisely here.

## Work decomposition: how insertions map onto the GPU hierarchy

We now discuss the unit of work — what GPU resource cooperates on a single key insertion.
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
  This is the design we adopted.

### The structure that follows

1. A **bucket** is a contiguous, cache-line-aligned run of `B` slots.
2. A **cooperative tile** of `B` threads (`cooperative_groups::thread_block_tile<B>`) owns one in-flight key at a time.
   Each tile lane holds one slot of the current bucket in a register; one bucket probe is one coalesced load.
3. **Tiles per warp** = `WarpSize / B`.
   For NVIDIA with 8-byte slots that is 2.
4. **Block size** is purely a hyperparameter that controls occupancy — some multiple of `WarpSize`, tuned empirically.
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
| `TargetLoadFactor` | tuned (0.85–0.95)            | Robin Hood quality — empirical           |
| `MaxProbeBuckets`  | 8                            | Worst-case time vs. insert-failure rate  |

So the only genuinely tunable hyperparameters are `BlockSize`, `TargetLoadFactor`, and `MaxProbeBuckets`.
The rest are expressed in the code as compile-time constants derived from `SlotBytes`, `CacheLineBytes`, and `WarpSize`, with `CacheLineBytes` and `WarpSize` exposed as template parameters of the table (with sensible defaults) rather than hard-coded.

### Could this scheme also work for AMD GPUs?

In principle, yes.
AMD GPUs (CDNA, RDNA) have a 64-byte cache line and a 64-thread wavefront.
With 8-byte slots:

- NVIDIA: `BucketSize` = 128 / 8 = 16, `TilesPerWarp` = 32 / 16 = **2**.
- AMD:    `BucketSize` =  64 / 8 =  8, `TilesPerWarp` = 64 /  8 = **8**.

The same algorithm runs unchanged; only the derived constants differ.
We are not targeting AMD, but this is the reason `CacheLineBytes` and `WarpSize` are parameters of the table type rather than hard-coded `128` and `32`.
Hard-coding would result in a loss of type information, and make reasoning conceptually about the algorithm more difficult.
