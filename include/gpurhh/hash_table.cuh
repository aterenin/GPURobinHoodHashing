#pragma once

// gpurhh — GPU Robin Hood Hash table.
//
// Header-only CUDA library implementing a parallel hash table for NVIDIA GPUs,
// using Robin Hood open-addressing with bucketed, sub-warp-cooperative probing.
//
// See the notes/ directory for the design rationale behind this layout.
//
// The hash table itself lives in device memory. Host code constructs and
// destructs a HashTable, which allocates and zero-initialises the slot array
// on the device. All operations on the contents — insert, get — are device
// code, exposed through the View handle which is meant to be passed by value
// into user kernels. Data movement between host and device (e.g. building the
// input arrays, reading back results) is the caller's responsibility.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

namespace gpurhh {

// -----------------------------------------------------------------------------
// Defaults: empty-key sentinel and hash function.
// -----------------------------------------------------------------------------

// Reserved value signalling an empty slot. Users may not insert this key.
//
// Default is the all-bits-set representation of `Key` — every byte is 0xFF,
// so `value` is the maximum representable value (e.g. 0xFFFFFFFFu for
// uint32_t). The all-ones bit pattern lets `HashTable` initialise its
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
// The trait exposes both `value` (the key value itself) and `memset_byte`
// (the byte that, repeated, produces that value). `HashTable`'s
// constructor uses `memset_byte` to init the slot array with one
// cudaMemset call.
template <class Key>
struct default_empty_key {
    static_assert(std::is_unsigned_v<Key>,
                  "default_empty_key only provides a default for unsigned "
                  "integral types; supply an explicit EmptyKey for signed "
                  "or other key types");

    static constexpr unsigned char memset_byte = 0xFFu;
    static constexpr Key           value       = static_cast<Key>(~Key{});
};

// Compile-time verification for the supported integer types.
static_assert(default_empty_key<std::uint8_t >::value == 0xFFu);
static_assert(default_empty_key<std::uint16_t>::value == 0xFFFFu);
static_assert(default_empty_key<std::uint32_t>::value == 0xFFFFFFFFu);
static_assert(default_empty_key<std::uint64_t>::value == 0xFFFFFFFFFFFFFFFFull);

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
    class Hash         = default_hash<Key>,
    Key   EmptyKey     = default_empty_key<Key>::value,
    int   CacheLineBytes = 128,
    int   WarpSize     = 32
>
class HashTable {
public:
    using key_type   = Key;
    using value_type = Value;
    using hasher     = Hash;

    // Packed (key, value) slot. One Slot is the unit of an atomic CAS.
    struct Slot {
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
                  "key+value pair)");
    // Temporary: v1 implements only 64-bit slots, which use a single
    // atomicCAS<uint64_t>. Remove this assertion once 128-bit slots are
    // supported (needed for 64-bit keys with 64-bit values, requiring
    // 128-bit CAS on sm_70+).
    static_assert(slot_bytes == 8,
                  "only 64-bit slots are currently supported");
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

        Slot*       slots_         = nullptr;
        std::size_t capacity_      = 0;   // power of two
        std::size_t capacity_mask_ = 0;   // capacity_ - 1, for modulo via AND
        Hash        hash_{};
    };

    // -------------------------------------------------------------------------
    // Host-side resource management.
    // -------------------------------------------------------------------------

    // Allocates a slot array on the current device sized for at least
    // `min_capacity` slots, rounded up to the next power of two, and
    // initialises every slot to (empty_key, _).
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
    Slot*       slots_    = nullptr;
    std::size_t capacity_ = 0;
};

// -----------------------------------------------------------------------------
// Definitions.
// -----------------------------------------------------------------------------
//
// All function bodies are TODO. They live in the header because the class is a
// template and instantiations need full definitions visible at the call site.

template <class K, class V, class H, K E, int CL, int WS>
HashTable<K, V, H, E, CL, WS>::HashTable(std::size_t /*min_capacity*/) {
    // TODO: round min_capacity up to next power of two; cudaMalloc slots_;
    //       launch an init kernel (or cudaMemset to a chosen byte pattern) to
    //       set every slot's key field to empty_key.
}

template <class K, class V, class H, K E, int CL, int WS>
HashTable<K, V, H, E, CL, WS>::~HashTable() {
    // TODO: cudaFree(slots_) if non-null.
}

template <class K, class V, class H, K E, int CL, int WS>
HashTable<K, V, H, E, CL, WS>::HashTable(HashTable&&) noexcept {
    // TODO: steal slots_ and capacity_ from the moved-from instance.
}

template <class K, class V, class H, K E, int CL, int WS>
HashTable<K, V, H, E, CL, WS>&
HashTable<K, V, H, E, CL, WS>::operator=(HashTable&&) noexcept {
    // TODO: free current state if any, then steal from RHS.
    return *this;
}

template <class K, class V, class H, K E, int CL, int WS>
typename HashTable<K, V, H, E, CL, WS>::View
HashTable<K, V, H, E, CL, WS>::view() const noexcept {
    // TODO: populate a View with slots_, capacity_, capacity_ - 1, and a
    //       default-constructed Hash.
    return {};
}

template <class K, class V, class H, K E, int CL, int WS>
template <class Tile>
__device__ bool HashTable<K, V, H, E, CL, WS>::View::insert(
    const Tile& /*tile*/, K /*key*/, V /*value*/)
{
    // TODO: bucketed Robin Hood insertion. Each lane of `tile` reads one slot
    //       of the current bucket; tile.ballot() identifies empty/match/
    //       displaceable slots; the chosen lane performs the CAS; on
    //       displacement, the evicted (key, value) becomes the new in-flight
    //       pair and the tile advances by one bucket.
    return false;
}

template <class K, class V, class H, K E, int CL, int WS>
template <class Tile>
__device__ bool HashTable<K, V, H, E, CL, WS>::View::get(
    const Tile& /*tile*/, K /*key*/, V& /*out_value*/) const
{
    // TODO: bucketed lookup. Same probing rule as insert but read-only; stop
    //       on key-match (return true), on empty slot (return false), or on a
    //       resident whose probe distance is smaller than ours (return false).
    return false;
}

} // namespace gpurhh
