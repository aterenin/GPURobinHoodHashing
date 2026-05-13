# GPU Robin Hood Hash Table — Design Brainstorm

## Goals

- **Header-only CUDA library.** A single `.cuh` (or a small set) that users `#include` and
  template-instantiate for their key/value types. No separate compilation of the table itself.
- **Massively parallel insertion.** A batch of `N` keys is inserted by `N` (or more) GPU threads
  cooperatively. Worst-case behaviour must remain bounded even under heavy contention.
- **Memory-bandwidth bound.** On large tables, the table is much bigger than the L2 cache, so
  performance is determined by how many useful bytes each DRAM transaction delivers. The aim is
  to come as close to peak HBM bandwidth as the access pattern allows.

## Why Robin Hood?

Open-addressing with the invariant *"the entry in slot `i` whose probe distance is smaller gets
evicted by an entry whose probe distance is larger"* gives:

- A tight, low-variance distribution of probe distances → small worst-case lookup length even at
  load factors of 0.85–0.9.
- A natural early-exit rule for lookups (stop as soon as the slot's probe distance is smaller
  than the probe distance of the key we're searching for).
- Backward-shift deletion that avoids tombstones.

These properties matter even more on a GPU than on a CPU because the warp's tail latency is set
by the longest probe sequence among its 32 lanes.

## Slot layout

A slot is the unit of an atomic update. The simplest design that admits lock-free CAS on a single
machine word is:

```
struct Slot {                       // 8 bytes total — fits in atomicCAS<uint64_t>
    uint32_t key;
    uint32_t value;                 // or a 32-bit payload / pointer index
};
```

Empty slots use a reserved sentinel key (e.g. `0xFFFFFFFF`); users are forbidden from inserting
that key. For 64-bit keys or larger payloads we have two options:

1. **128-bit slots** with `__int128` CAS via `atomicCAS` on `unsigned long long` pairs — supported
   on sm_70+ but with reduced throughput.
2. **Separate parallel arrays** (`keys[]` and `values[]`) with the key array being the source of
   truth for CAS, and values being written only by the thread that successfully claimed the slot.
   This halves the bytes-per-CAS but requires care: a finder must re-read the key after reading
   the value to make sure the slot wasn't reused mid-read.

We will start with the 64-bit packed slot and parameterise later.

**Probe distance.** We do *not* store probe distance in the slot. It is derivable as
`(slot_index - hash(key)) mod capacity`. Storing it would cost bits we'd rather give to the
payload, and the derivation is a single subtract once we've read the slot.

## Capacity, hash function, load factor

- Capacity is a power of two so `mod` becomes a mask. Users specify a target load factor
  (default 0.7); we round capacity up to `next_pow2(N / load_factor)`.
- Hash function is a template parameter; default is a finalizer-style mixer (e.g. the
  splitmix64/Murmur3 finalizer) that is cheap on GPU and decorrelates structured keys well.
- Robin Hood tolerates higher load than linear probing, but on a GPU the warp tail-latency
  argument pushes us back toward 0.7–0.8 rather than 0.9.

## Parallel insertion algorithm

Each thread owns one `(key, value)` pair. The thread walks the probe sequence and, at every slot,
performs one of three actions:

1. **Empty slot, CAS succeeds.** Done.
2. **Same key, CAS succeeds.** This is an update; done.
3. **Occupied by another key.** Compare probe distances. If the resident is "richer" (its probe
   distance from *its* home is smaller than ours from *our* home), we attempt to CAS our pair
   into the slot. On success, we adopt the displaced pair and continue probing from this slot.
   If the resident is "poorer" or equal, we move on.

A CAS failure is treated as "the world changed under us": we re-read the slot and re-evaluate
without advancing the probe index. This is the standard lock-free retry.

### Why this is correct under contention

The Robin Hood invariant is maintained by every individual CAS because each CAS only swaps in a
pair with probe distance strictly greater than the one already in the slot (or the slot was
empty). Two concurrent inserts targeting the same slot serialise through CAS, and whichever
loses retries with the freshest state. There is no global ordering requirement: as long as every
write respects the local invariant, the final table is a valid Robin Hood layout for *some*
permutation of the input.

Caveat: the *value* portion of an update (case 2) needs care. If two threads insert the same key
concurrently and the user wants "last write wins" semantics, we need an additional CAS loop on
the value, or we tolerate non-deterministic resolution.

## Achieving bandwidth saturation

The 64-bit slot is small relative to a 128-byte L2 cache line: one transaction brings 16 slots.
The bandwidth strategy follows from this:

- **Warp-cooperative probing.** Within a warp, threads whose home slots land in the same cache
  line should service their probes from one load, not 32 loads. We have the warp ballot/shuffle
  primitives to do this cheaply.
- **Bucketed probing.** Treat the table as an array of *buckets* of 16 slots (one cache line).
  A probe inspects an entire bucket at a time — one coalesced 128-byte load — and uses warp
  intrinsics (`__ballot_sync`, `__match_any_sync`) to find the candidate slot. Insertion still
  uses CAS on a single 8-byte slot inside the bucket, but the read traffic per probe is exactly
  one cache line. This is the same idea as `bucketed_cuckoo` and Bucketized Cuckoo Hashing.
- **Probe length capping.** Cap probes at, say, 32 buckets. If a thread overruns the cap during
  insertion, mark the batch as failed and rehash. With Robin Hood at 0.7 load this happens with
  vanishing probability for reasonable `N`.
- **Avoid divergence.** All threads in a warp execute the same probe loop. The "I'm done, others
  aren't" case is handled with masked operations rather than early-return-then-divergence.

If we get this right, an insert of `N` keys at load factor 0.7 should cost approximately
`(1/(1-α)) * sizeof(slot)` bytes per insert in expectation, which is the information-theoretic
floor for an open-addressed table with this layout.

## Lookup and deletion

- **Lookup.** Same probing rule, no CAS. Stop as soon as we find the key, an empty slot, or a
  resident whose probe distance is smaller than ours. Lookup is fully read-only and trivially
  parallel — no synchronisation required.
- **Deletion.** Backward-shift: starting from the deleted slot, while the next slot is non-empty
  and its probe distance is positive, shift it back by one. On a concurrent GPU this is the
  hardest operation to make correct without locks. **Initial scope: defer deletion**, support
  insert + lookup only, and add it once we have benchmarks in place.

## API sketch

```cpp
namespace gprh {

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

    void find(const Key* keys, Value* values_out,
              std::size_t n, cudaStream_t stream = 0) const;

    // A device-side, copyable handle for use inside user kernels.
    struct View {
        __device__ bool  insert(Key k, Value v);
        __device__ bool  find  (Key k, Value& out) const;
    };
    View view();

private:
    Slot*       slots_;
    std::size_t capacity_;
};

} // namespace gprh
```

The `View` is the centrepiece for header-only use: a user kernel that already has its keys in
registers can call `view.insert(...)` directly without a separate kernel launch.

## Open questions / things to settle before coding

1. **Slot width.** Start with 64-bit packed `(uint32 key, uint32 value)`, or go straight to a
   128-bit slot for 64-bit keys? The 64-bit case is the canonical micro-benchmark and is what
   most published GPU hash tables use.
2. **Bucket size.** 16 (one 128-byte cache line of 8-byte slots) is the obvious choice. Worth
   measuring 8 and 32 as well.
3. **Failure semantics on full table.** Return a count of failed inserts, or `assert`? A
   bulk-insert API can naturally return the number of survivors and let the caller resize.
4. **Resize.** Out of scope for v1. The table is allocated at a fixed capacity. Users that need
   growth allocate a bigger one and re-insert.
5. **Multi-GPU.** Out of scope for v1.
6. **Testing.** A small CPU reference implementation (sequential Robin Hood) plus a randomised
   differential test against the GPU implementation, plus stress tests at 0.5/0.7/0.9 load.

## Suggested next steps

1. Sketch the `Slot`, hash, and probe primitives in a single `.cuh`.
2. Implement the insert kernel without bucketing, just to validate correctness and the CAS loop.
3. Add the bucketed (cache-line-aligned) variant and benchmark against the naive one.
4. Add `find`, then benchmark against `cuCollections` / `WarpCore` as external baselines.
