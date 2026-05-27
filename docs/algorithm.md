# Algorithm

## Insertion

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

Caveat: like many other concurrent operations, the exact table state may end up non-deterministic in the sense of depending on the exact order in which operations execute.

### Reduction operators

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

`replace_op` gives last-writer-wins semantics.
`sum_op` accumulates values for repeated-key inserts — useful as a parallel histogram or counter primitive.
Users can define their own functor with the same signature.

For commutative and associative reductions like `sum_op`, the final stored value is fully deterministic regardless of how the CAS retries interleave across tiles: the result is just the reduction of all inserted values for that key.
For non-commutative reductions, the outcome depends on which writers win their CAS races.
Note that floating point operations can be non-associative, which can lead to non-determinism if float-valued data is stored in the table.

## Lookup

`View::get` follows the same probing rule as insert, read-only, returning `cuda::std::optional<Value>` — populated on hit, empty on miss.
The probe terminates as soon as it sees one of:

- A key match → return `{value}`.
- An empty slot → return `nullopt`. Robin Hood would have placed the key before this point.
- A resident "richer" than ours (smaller probe distance) → return `nullopt`. Our key would have evicted that resident if it were in the table.
- The probe cap → return `nullopt`. The insert-side cap means the key cannot live further than this.

Lookups are read-only and trivially parallel: concurrent lookups never race each other.

**Concurrent get during insert is not supported but might potentially work for some use cases.** A get running simultaneously with an insert can return `nullopt` for a key that is about to be inserted, or for a key in the middle of being displaced.
Workloads should ordinarily synchronize between insert and get phases.

## Deletion

Not supported.
The natural design is backward-shift deletion, but a correct lock-free implementation is significantly harder than insert/lookup, and the use case (this is an append-only table for most consumers) hasn't pushed us to do it.
