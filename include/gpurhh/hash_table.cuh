#pragma once

// gpurhh — GPU Robin Hood Hash table.
//
// Header-only CUDA library implementing a parallel hash table for NVIDIA GPUs,
// using Robin Hood open-addressing with bucketed, sub-warp-cooperative probing.
//
// See the notes/ directory for the design rationale behind this layout.
//
// The hash table itself lives in device memory. Host code constructs and
// destructs a HashTable, which allocates and zero-initializes the slot array
// on the device. All operations on the contents — insert, get — are device
// code, exposed through the View handle which is meant to be passed by value
// into user kernels. Data movement between host and device (e.g. building the
// input arrays, reading back results) is the caller's responsibility.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace gpurhh {

// -----------------------------------------------------------------------------
// Defaults: empty-key sentinel and hash function.
// -----------------------------------------------------------------------------

// Reserved value signaling an empty slot. Users may not insert this key.
//
// Default is the all-bits-set representation of `Key` — every byte is 0xFF,
// so `value` is the maximum representable value (e.g. 0xFFFFFFFFu for
// uint32_t). The all-ones bit pattern lets `HashTable` initialize its
// slot array with a single cudaMemset(0xFF, ...) at construction.
//
// Compiles for: unsigned integral types only. For signed integers,
// floating-point types, scoped enums, or user-defined structs, supply an
// explicit `EmptyKey` template argument to `HashTable` (and likely your
// own `Hash` functor too). We deliberately don't provide a default for
// signed types because no single repeated byte pattern gives a "good"
// signed sentinel: 0xFF gives -1 (a common legitimate key); 0x80 gives
// type-dependent values that are only INT_MIN for int8_t; INT_MIN itself
// isn't a repeated-byte pattern. Leaving the choice to the user avoids
// committing to any of those compromises.
//
// The trait exposes both `key` (the empty-key value itself) and
// `memset_byte` (the byte that, repeated, produces that value).
// `HashTable`'s constructor uses `memset_byte` to init the slot array
// with one cudaMemset call.
template <class Key>
struct default_empty_key {
    static_assert(std::is_unsigned_v<Key>,
                  "default_empty_key only provides a default for unsigned "
                  "integral types; supply an explicit EmptyKey for signed "
                  "or other key types");

    static constexpr unsigned char memset_byte = 0xFFu;
    static constexpr Key           key         = static_cast<Key>(~Key{});
};

// Compile-time verification for the supported integer types.
static_assert(default_empty_key<std::uint8_t >::key == 0xFFu);
static_assert(default_empty_key<std::uint16_t>::key == 0xFFFFu);
static_assert(default_empty_key<std::uint32_t>::key == 0xFFFFFFFFu);
static_assert(default_empty_key<std::uint64_t>::key == 0xFFFFFFFFFFFFFFFFull);

// Default hash: MurmurHash3 32-bit finalizer (fmix32) for 4-byte keys; an
// 8-byte specialization is reserved for when 128-bit slot support is added.
// Both are cheap on the GPU, branch-free, and have strong avalanche. Users
// can supply their own via the Hash template parameter.

namespace detail {

__host__ __device__ inline std::uint32_t fmix32(std::uint32_t h) noexcept {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__host__ __device__ inline std::uint64_t fmix64(std::uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// Round `n` up to the next power of two. Returns 1 for n == 0 or n == 1.
constexpr inline std::size_t next_pow2(std::size_t n) noexcept {
    if (n <= 1) return 1;
    return std::size_t{1} << std::bit_width(n - 1);
}

} // namespace detail

// Primary template: dispatch on sizeof(Key). Defined only for 4- and 8-byte
// keys, matching the supported slot widths.
template <class Key, std::size_t = sizeof(Key)>
struct default_hash;

template <class Key>
struct default_hash<Key, 4> {
    __host__ __device__ inline std::uint32_t operator()(Key key) const noexcept {
        return detail::fmix32(std::bit_cast<std::uint32_t>(key));
    }
};

template <class Key>
struct default_hash<Key, 8> {
    __host__ __device__ inline std::uint64_t operator()(Key key) const noexcept {
        return detail::fmix64(std::bit_cast<std::uint64_t>(key));
    }
};

// -----------------------------------------------------------------------------
// HashTable: host-side owner of the device-resident slot array.
// -----------------------------------------------------------------------------

template <
    class Key,
    class Value,
    class Hash             = default_hash<Key>,
    Key   EmptyKey         = default_empty_key<Key>::key,
    int   CacheLineBytes   = 128,
    int   WarpSize         = 32,
    int   MaxProbeBuckets  = 8
>
class HashTable {
public:
    using key_type   = Key;
    using value_type = Value;
    using hasher     = Hash;

    // Packed (key, value) slot. One Slot is the unit of an atomic CAS.
    // The alignas forces alignof(Slot) == sizeof(Slot), which is required
    // by cuda::atomic_ref<Slot> (it needs the storage aligned to the
    // atomic op width). For the supported Key/Value pairs (same-size
    // integral types) sizeof(Key) + sizeof(Value) == sizeof(Slot) with no
    // internal padding.
    struct alignas(sizeof(Key) + sizeof(Value)) Slot {
        Key   key;
        Value value;
    };

    // ---- Architecture-derived constants ----

    static constexpr Key empty_key        = EmptyKey;
    static constexpr int slot_bytes       = sizeof(Slot);
    static constexpr int cache_line_bytes = CacheLineBytes;
    static constexpr int warp_size        = WarpSize;
    static constexpr int bucket_size      = CacheLineBytes / slot_bytes;
    static constexpr int tile_size        = bucket_size;
    static constexpr int tiles_per_warp   = WarpSize / bucket_size;

    // Slot / cache-line invariants.
    static_assert(slot_bytes == 8 || slot_bytes == 16,
                  "slot must be 8 or 16 bytes (a 64-bit or 128-bit packed "
                  "key+value pair). The 16-byte case uses 128-bit CAS via "
                  "cuda::atomic_ref — single-instruction on sm_90+, "
                  "emulated by libcu++ on earlier architectures.");
    static_assert(slot_bytes <= CacheLineBytes,
                  "slot does not fit in a cache line");
    static_assert(CacheLineBytes % slot_bytes == 0,
                  "cache line must be a whole multiple of slot size");

    // Tile / bucket invariants. tile_size == bucket_size by construction; the
    // cooperative_groups::thread_block_tile requirements drive these.
    static_assert(tile_size > 0,
                  "tile size must be positive");
    static_assert((tile_size & (tile_size - 1)) == 0,
                  "tile size must be a power of two "
                  "(required by cooperative_groups::tiled_partition)");
    static_assert(tile_size <= WarpSize,
                  "tile size must not exceed warp size");
    static_assert(WarpSize % bucket_size == 0,
                  "warp size must be a whole multiple of bucket size");

    // A bucket is a contiguous run of `bucket_size` slots aligned to one
    // cache line. The alignas guarantees that loading a bucket via a
    // cooperative tile produces a single coalesced cache-line transaction
    // (the bucket can never straddle two cache lines).
    struct alignas(CacheLineBytes) Bucket {
        Slot slots[bucket_size];
    };

    static_assert(sizeof(Bucket) == CacheLineBytes,
                  "bucket must occupy exactly one cache line");
    static_assert(alignof(Bucket) == CacheLineBytes,
                  "bucket alignment must equal cache line size");

    // Probe-length cap. An insert that would place a key further than this
    // many buckets from its home bucket fails (the caller is expected to
    // rehash into a larger table). Get takes advantage of the same cap: if
    // it probes this many buckets without finding the key, the key cannot
    // be in the table, because insert would have failed before placing it
    // beyond the cap.
    //
    // The default of 8 comfortably handles load factors up to ~0.9 with
    // bucket_size 16, where the expected longest probe is well under the
    // cap. The cap exists to bound the worst case for adversarial inputs
    // and very high load factors. Users targeting load factors above ~0.9
    // can override via the `MaxProbeBuckets` template parameter; lowering
    // it tightens the time bound at the cost of higher insert-failure
    // probability.
    static constexpr int max_probe_buckets = MaxProbeBuckets;
    static_assert(max_probe_buckets > 0,
                  "max_probe_buckets must be positive");

    // Byte value used to initialize the slot array via cudaMemset. Derived
    // from `empty_key`: every byte of `empty_key` must equal this byte for
    // the cudaMemset-based init path to produce a correctly-empty slot array.
    static constexpr unsigned char empty_key_byte =
        std::bit_cast<std::array<unsigned char, sizeof(Key)>>(empty_key)[0];

    // EmptyKey memsettability invariant.
    static_assert([]() {
        const auto bytes = std::bit_cast<std::array<unsigned char, sizeof(Key)>>(empty_key);
        for (std::size_t i = 1; i < sizeof(Key); ++i) {
            if (bytes[i] != bytes[0]) return false;
        }
        return true;
    }(),
    "EmptyKey must be a single byte pattern repeated across sizeof(Key) "
    "bytes so the slot array can be initialized via cudaMemset. Use the "
    "default for unsigned keys, or provide an EmptyKey of the form 0xBB..BB.");

    // -------------------------------------------------------------------------
    // View: lightweight device-side handle. Trivially copyable; pass by value
    // into user kernels. Holds only a device pointer and a couple of integers.
    // -------------------------------------------------------------------------
    class View {
    public:
        // Cooperative single-key insert.
        // Called by a tile of exactly tile_size threads in lock-step. Returns
        // true on success, false if the table is full (probe length cap hit).
        template <class Tile>
        __device__ bool insert(const Tile& tile, Key key, Value value);

        // Cooperative single-key lookup.
        // Returns true and writes the stored value to `out_value` if the key
        // is present; returns false otherwise.
        template <class Tile>
        __device__ bool get(const Tile& tile, Key key, Value& out_value) const;

        __host__ __device__ std::size_t capacity() const noexcept { return capacity_; }

    private:
        friend class HashTable;

        Bucket*     buckets_       = nullptr;
        std::size_t capacity_      = 0;   // power of two, in slots
        std::size_t capacity_mask_ = 0;   // capacity_ - 1, for modulo via AND
        Hash        hash_{};
    };

    // -------------------------------------------------------------------------
    // Host-side resource management.
    // -------------------------------------------------------------------------

    // Allocates a slot array on the current device sized for at least
    // `min_capacity` slots, rounded up to the next power of two, and
    // initializes every slot to (empty_key, _).
    explicit HashTable(std::size_t min_capacity);

    ~HashTable();

    HashTable(const HashTable&)            = delete;
    HashTable& operator=(const HashTable&) = delete;
    HashTable(HashTable&&) noexcept;
    HashTable& operator=(HashTable&&) noexcept;

    // Returns a device-side handle for use inside user kernels.
    View view() const noexcept;

    // Actual capacity (input capacity rounded up to a power of two).
    std::size_t capacity() const noexcept { return capacity_; }

private:
    Bucket*     buckets_  = nullptr;
    std::size_t capacity_ = 0;  // in slots
};

// -----------------------------------------------------------------------------
// Definitions.
// -----------------------------------------------------------------------------
//
// Function bodies live in the header because the class is a template and
// instantiations need full definitions visible at the call site. Host-side
// methods (constructor, destructor, move ops, view) are implemented; the
// device-side View::insert and View::get bodies are still TODO.

template <class K, class V, class H, K E, int CL, int WS, int MPB>
HashTable<K, V, H, E, CL, WS, MPB>::HashTable(std::size_t min_capacity) {
    // Round up to the next power of two, but never below a single bucket —
    // the bucketed probing logic requires at least bucket_size slots.
    const std::size_t rounded = detail::next_pow2(min_capacity);
    capacity_ = rounded < bucket_size ? std::size_t{bucket_size} : rounded;

    const std::size_t num_buckets = capacity_ / bucket_size;

    // Allocate the bucket array on the current device. cudaMalloc leaves
    // buckets_ as nullptr on failure; callers can check via
    // cudaPeekAtLastError().
    cudaMalloc(&buckets_, num_buckets * sizeof(Bucket));

    // Initialize every byte to empty_key_byte. For the supported default
    // (unsigned EmptyKey = 0xFF..FF) this writes the empty sentinel into
    // each slot's key field in a single bandwidth-bound DMA.
    cudaMemset(buckets_, empty_key_byte, num_buckets * sizeof(Bucket));
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
HashTable<K, V, H, E, CL, WS, MPB>::~HashTable() {
    // cudaFree(nullptr) is a documented no-op, so no need to guard.
    cudaFree(buckets_);
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
HashTable<K, V, H, E, CL, WS, MPB>::HashTable(HashTable&& other) noexcept
    : buckets_(other.buckets_), capacity_(other.capacity_)
{
    other.buckets_  = nullptr;
    other.capacity_ = 0;
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
HashTable<K, V, H, E, CL, WS, MPB>&
HashTable<K, V, H, E, CL, WS, MPB>::operator=(HashTable&& other) noexcept {
    if (this != &other) {
        // Free the current allocation before taking ownership of the new one.
        // cudaFree(nullptr) is a no-op, so this is safe even if *this is in
        // a moved-from / empty state.
        cudaFree(buckets_);
        buckets_        = other.buckets_;
        capacity_       = other.capacity_;
        other.buckets_  = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
typename HashTable<K, V, H, E, CL, WS, MPB>::View
HashTable<K, V, H, E, CL, WS, MPB>::view() const noexcept {
    View v;
    v.buckets_       = buckets_;
    v.capacity_      = capacity_;
    // For a usable table capacity_ is a power of two and the mask is
    // capacity_ - 1. For a moved-from / empty table capacity_ is 0; we set
    // the mask to 0 explicitly to avoid a wrap-around to ~0 which would be
    // misleading even though the View is unusable in that state anyway.
    v.capacity_mask_ = capacity_ > 0 ? capacity_ - 1 : 0;
    // v.hash_ is default-constructed by View's in-class default initializer.
    return v;
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
template <class Tile>
__device__ bool HashTable<K, V, H, E, CL, WS, MPB>::View::insert(
    const Tile& tile, K key, V value)
{
    // Bucketed Robin Hood insertion with lock-free CAS. The tile carries an
    // in-flight pair (cur_key, cur_value) that may change due to Robin Hood
    // displacement — when we evict a "richer" resident, that resident
    // becomes our new in-flight pair and we keep probing from the next
    // bucket.

    K cur_key   = key;
    V cur_value = value;

    const std::size_t num_buckets = capacity_ / bucket_size;
    const std::size_t bucket_mask = num_buckets > 0 ? num_buckets - 1 : 0;
    const std::size_t probe_bound = num_buckets < max_probe_buckets
        ? num_buckets
        : static_cast<std::size_t>(max_probe_buckets);

    std::size_t cur_home = (hash_(cur_key) & capacity_mask_) / bucket_size;
    std::size_t probe    = 0;

    while (probe < probe_bound) {
        const std::size_t bucket_idx = (cur_home + probe) & bucket_mask;

        // Cooperative load: one coalesced cache-line transaction. Each
        // lane reads one slot of the bucket (lane i reads slots[i]).
        const Slot slot = buckets_[bucket_idx].slots[tile.thread_rank()];

        // Helper: `target_lane` performs an atomic CAS to replace its slot
        // (whose current contents we have in `slot`) with the in-flight
        // pair. Returns the broadcast result of the CAS to every lane.
        auto try_cas = [&](int target_lane) -> bool {
            bool cas_ok = false;
            if (tile.thread_rank() == target_lane) {
                Slot expected = slot;
                const Slot desired{cur_key, cur_value};
                cuda::atomic_ref<Slot, cuda::thread_scope_device> atomic_slot(
                    buckets_[bucket_idx].slots[target_lane]);
                cas_ok = atomic_slot.compare_exchange_strong(expected, desired);
            }
            return tile.shfl(cas_ok, target_lane);
        };

        // (1) Key match — update the existing entry's value.
        const auto match_mask = tile.ballot(slot.key == cur_key);
        if (match_mask != 0) {
            if (try_cas(__ffs(match_mask) - 1)) return true;
            continue;  // CAS lost the race — retry this bucket.
        }

        // (2) Empty slot — claim it for our in-flight pair.
        const auto empty_mask = tile.ballot(slot.key == empty_key);
        if (empty_mask != 0) {
            if (try_cas(__ffs(empty_mask) - 1)) return true;
            continue;  // Someone else took the slot — retry.
        }

        // (3) Displaceable slot — Robin Hood swap with a richer resident.
        // Compute each resident's probe distance. (For empty slots this
        // gives nonsense, but we already eliminated the empty case above,
        // so the next ballot is meaningful.)
        const std::size_t resident_home =
            (hash_(slot.key) & capacity_mask_) / bucket_size;
        const std::size_t resident_probe =
            (bucket_idx - resident_home) & bucket_mask;
        const auto displaceable_mask = tile.ballot(resident_probe < probe);
        if (displaceable_mask != 0) {
            const int target_lane = __ffs(displaceable_mask) - 1;
            if (try_cas(target_lane)) {
                // Adopt the evicted pair as our new in-flight pair. We
                // continue from the next bucket; the evicted pair's probe
                // distance there is `resident_probe + 1`.
                cur_key   = tile.shfl(slot.key,        target_lane);
                cur_value = tile.shfl(slot.value,      target_lane);
                cur_home  = tile.shfl(resident_home,   target_lane);
                probe     = tile.shfl(resident_probe,  target_lane) + 1;
                continue;
            }
            continue;  // CAS lost the race — retry.
        }

        // (4) Nothing applies — advance one bucket.
        ++probe;
    }

    // Probe budget exhausted. Caller is expected to rehash into a larger
    // table or accept the failure.
    return false;
}

template <class K, class V, class H, K E, int CL, int WS, int MPB>
template <class Tile>
__device__ bool HashTable<K, V, H, E, CL, WS, MPB>::View::get(
    const Tile& tile, K key, V& out_value) const
{
    // Bucket-level Robin Hood lookup. Probe home_bucket, then home_bucket+1,
    // etc., wrapping modulo num_buckets. Each iteration is one coalesced
    // cache-line load (one slot per tile lane) and a few tile-wide ballots
    // to identify the terminating condition.

    const auto h = hash_(key);
    const std::size_t home_slot   = h & capacity_mask_;
    const std::size_t home_bucket = home_slot / bucket_size;
    const std::size_t num_buckets = capacity_ / bucket_size;
    const std::size_t bucket_mask = num_buckets > 0 ? num_buckets - 1 : 0;

    // Probe at most max_probe_buckets buckets, or every bucket in the table
    // if it's smaller than the cap.
    const std::size_t probe_bound = num_buckets < max_probe_buckets
        ? num_buckets
        : static_cast<std::size_t>(max_probe_buckets);

    for (std::size_t probe = 0; probe < probe_bound; ++probe) {
        const std::size_t bucket_idx = (home_bucket + probe) & bucket_mask;

        // Each lane reads one slot — one coalesced cache-line transaction
        // for the whole tile.
        const Slot slot = buckets_[bucket_idx].slots[tile.thread_rank()];

        // (1) Key match. The matching lane broadcasts its value to the rest
        //     of the tile via shfl.
        const auto match_mask = tile.ballot(slot.key == key);
        if (match_mask != 0) {
            const int matching_lane = __ffs(match_mask) - 1;
            out_value = tile.shfl(slot.value, matching_lane);
            return true;
        }

        // (2) Empty slot. The key isn't in the table — Robin Hood would have
        //     placed it earlier in the probe sequence.
        if (tile.ballot(slot.key == empty_key) != 0) return false;

        // (3) Robin Hood early termination. If any resident is "richer"
        //     (closer to its home bucket than we are to ours), our key would
        //     have evicted it on insertion, so it can't be later in the
        //     probe sequence.
        const std::size_t resident_home =
            (hash_(slot.key) & capacity_mask_) / bucket_size;
        const std::size_t resident_probe =
            (bucket_idx - resident_home) & bucket_mask;
        if (tile.ballot(resident_probe < probe) != 0) return false;

        // No match, no empty slot, no richer resident — every slot here is
        // occupied by a resident at least as displaced as we would be.
        // Advance one bucket.
    }

    // Either hit the probe cap, or wrapped fully around in a small table,
    // without finding the key, an empty slot, or a richer resident. The
    // insert invariant guarantees the key cannot be at probe distance
    // greater than max_probe_buckets, so this is a definite "not present".
    return false;
}

} // namespace gpurhh
