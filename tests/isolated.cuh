#pragma once

// Shared infrastructure for the "isolated" tests — those that exercise
// View::insert or View::get on controlled table states, by-passing the
// end-to-end bulk path used in kernels.cuh.
//
// The defining choice here is the IdentityHash: with hash(K) = K, every
// key's home bucket is fully predictable from its value, which lets tests
// hand-build or hand-predict any valid Robin Hood state. The header also
// provides:
//
//   - set_state / read_state — direct cudaMemcpy into / out of the table's
//     internal bucket array (requires GPURHH_ENABLE_INTERNAL_ACCESS, set
//     by tests.cuh).
//   - insert_many_kernel / do_insert — tile-strided insert and host-side
//     launcher.
//   - assert_robin_hood_invariant — structural check usable as a safety
//     net after any sequence of inserts.

#include <cassert>
#include <cstdint>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// tests.cuh must precede the gpurhh header — it sets the
// GPURHH_ENABLE_INTERNAL_ACCESS macro that exposes HashTable::data().
#include "tests.cuh"
#include <gpurhh/hash_table.cuh>

#include <optional>
#include <vector>

namespace cg = cooperative_groups;

// Identity hash for predictable bucket placement: hash(K) = K, so home
// slot = K & capacity_mask and home bucket = home_slot / bucket_size.
struct IdentityHash {
    __device__ std::uint32_t operator()(std::uint32_t k) const noexcept { return k; }
};

// Default table type for the isolated tests. The Reduction is left at the
// `replace_op` default; tests that want a different reduction declare their
// own alias (see test_randomized.cu).
using TestTable = gpurhh::HashTable<std::uint32_t, std::uint32_t, IdentityHash>;
using TestView  = TestTable::View;

// Construct an all-empty state matching `table`'s capacity. The Table type
// is deduced from the argument.
template <class Table>
inline std::vector<typename Table::Slot> empty_state(const Table& table) {
    return std::vector<typename Table::Slot>(table.capacity(),
                                             {Table::empty_key, 0});
}

// cudaMemcpy a host-built state directly into the table's bucket array.
// The flat Slot[] layout matches Bucket[N] layout because bucket_size *
// sizeof(Slot) == sizeof(Bucket) with no internal padding.
template <class Table>
inline void set_state(Table& table,
                      const std::vector<typename Table::Slot>& host_slots) {
    assert(host_slots.size() == table.capacity());
    cudaMemcpy(table.data(), host_slots.data(),
               host_slots.size() * sizeof(typename Table::Slot),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;
}

// Read the entire table state back to host.
template <class Table>
inline std::vector<typename Table::Slot> read_state(const Table& table) {
    std::vector<typename Table::Slot> result(table.capacity());
    cudaMemcpy(result.data(), table.data(),
               result.size() * sizeof(typename Table::Slot),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    return result;
}

// Insert kernel: one tile per (key, value) pair, tile-strided over n.
template <class Table>
__global__ void insert_many_kernel(typename Table::View view,
                                   const typename Table::key_type* keys,
                                   const typename Table::value_type* values,
                                   std::size_t n) {
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id = blockIdx.x * tiles_per_block +
                                 threadIdx.x / Table::tile_size;
    const std::size_t total_tiles = gridDim.x * tiles_per_block;
    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        view.insert(tile, keys[i], values[i]);
    }
}

// Host-side bulk insert helper.
template <class Table>
inline void do_insert(Table& table,
                      const std::vector<typename Table::key_type>& keys,
                      const std::vector<typename Table::value_type>& values) {
    using Key   = typename Table::key_type;
    using Value = typename Table::value_type;

    assert(keys.size() == values.size());
    if (keys.empty()) return;

    Key*   d_keys   = nullptr;
    Value* d_values = nullptr;
    cudaMalloc(&d_keys,   keys.size()   * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values, values.size() * sizeof(Value)) >> CUDA_CHECK;
    cudaMemcpy(d_keys,   keys.data(),   keys.size()   * sizeof(Key),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;
    cudaMemcpy(d_values, values.data(), values.size() * sizeof(Value),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;

    constexpr int block_size = 128;
    const int tiles_per_block = block_size / Table::tile_size;
    const int grid_size =
        static_cast<int>((keys.size() + tiles_per_block - 1) / tiles_per_block);
    insert_many_kernel<Table><<<grid_size, block_size>>>(
        table.view(), d_keys, d_values, keys.size());
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    cudaFree(d_keys);
    cudaFree(d_values);
}

// Insert kernel that captures per-key success/failure. `outcomes[i]` is set
// to 0 if the i-th insert succeeded, 1 if it returned false (probe cap hit).
template <class Table>
__global__ void insert_many_with_outcomes_kernel(
    typename Table::View view,
    const typename Table::key_type* keys,
    const typename Table::value_type* values,
    int* outcomes,
    std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id = blockIdx.x * tiles_per_block +
                                 threadIdx.x / Table::tile_size;
    const std::size_t total_tiles = gridDim.x * tiles_per_block;
    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        const auto leftover = view.insert(tile, keys[i], values[i]);
        if (tile.thread_rank() == 0) {
            outcomes[i] = leftover.has_value() ? 1 : 0;
        }
    }
}

// Host-side bulk-insert helper that captures each insert's return value as
// a per-key outcome (0 = succeeded, 1 = probe cap hit / failed).
template <class Table>
inline std::vector<int> do_insert_with_outcomes(
    Table& table,
    const std::vector<typename Table::key_type>& keys,
    const std::vector<typename Table::value_type>& values)
{
    using Key   = typename Table::key_type;
    using Value = typename Table::value_type;

    assert(keys.size() == values.size());
    std::vector<int> outcomes(keys.size(), 0);
    if (keys.empty()) return outcomes;

    Key*   d_keys     = nullptr;
    Value* d_values   = nullptr;
    int*   d_outcomes = nullptr;
    cudaMalloc(&d_keys,     keys.size()   * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values,   values.size() * sizeof(Value)) >> CUDA_CHECK;
    cudaMalloc(&d_outcomes, keys.size()   * sizeof(int))   >> CUDA_CHECK;
    cudaMemcpy(d_keys,   keys.data(),   keys.size()   * sizeof(Key),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;
    cudaMemcpy(d_values, values.data(), values.size() * sizeof(Value),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;

    constexpr int block_size = 128;
    const int tiles_per_block = block_size / Table::tile_size;
    const int grid_size =
        static_cast<int>((keys.size() + tiles_per_block - 1) / tiles_per_block);
    insert_many_with_outcomes_kernel<Table><<<grid_size, block_size>>>(
        table.view(), d_keys, d_values, d_outcomes, keys.size());
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    cudaMemcpy(outcomes.data(), d_outcomes, keys.size() * sizeof(int),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_outcomes);
    return outcomes;
}

// Single-tile get kernel: one block, one tile, looks up one key.
template <class Table>
__global__ void get_one_kernel(typename Table::View view,
                               typename Table::key_type key,
                               typename Table::value_type* value_out,
                               int* found_out)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const auto result = view.get(tile, key);
    if (tile.thread_rank() == 0) {
        *value_out = result.value_or(typename Table::value_type{});
        *found_out = result.has_value() ? 1 : 0;
    }
}

// Host-side single-key get returning an optional value.
template <class Table>
inline std::optional<typename Table::value_type>
get_one(Table& table, typename Table::key_type key)
{
    using Value = typename Table::value_type;
    Value* d_value = nullptr;
    int*   d_found = nullptr;
    cudaMalloc(&d_value, sizeof(Value)) >> CUDA_CHECK;
    cudaMalloc(&d_found, sizeof(int))   >> CUDA_CHECK;

    get_one_kernel<Table><<<1, Table::tile_size>>>(
        table.view(), key, d_value, d_found);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    Value value{};
    int found = 0;
    cudaMemcpy(&value, d_value, sizeof(Value), cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    cudaMemcpy(&found, d_found, sizeof(int),   cudaMemcpyDeviceToHost) >> CUDA_CHECK;

    cudaFree(d_value);
    cudaFree(d_found);
    return found ? std::optional<Value>{value} : std::nullopt;
}

// Bulk get kernel: one tile per key, tile-strided.
template <class Table>
__global__ void get_many_kernel(typename Table::View view,
                                const typename Table::key_type* keys,
                                typename Table::value_type* values_out,
                                int* found_out,
                                std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id = blockIdx.x * tiles_per_block +
                                 threadIdx.x / Table::tile_size;
    const std::size_t total_tiles = gridDim.x * tiles_per_block;
    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        const auto result = view.get(tile, keys[i]);
        if (tile.thread_rank() == 0) {
            values_out[i] = result.value_or(typename Table::value_type{});
            found_out[i]  = result.has_value() ? 1 : 0;
        }
    }
}

// Host-side bulk get. Resizes `values_out` and `found_out` to match keys.
template <class Table>
inline void do_get(Table& table,
                   const std::vector<typename Table::key_type>& keys,
                   std::vector<typename Table::value_type>& values_out,
                   std::vector<int>& found_out)
{
    using Key   = typename Table::key_type;
    using Value = typename Table::value_type;

    values_out.assign(keys.size(), Value{});
    found_out.assign(keys.size(), 0);
    if (keys.empty()) return;

    Key*   d_keys   = nullptr;
    Value* d_values = nullptr;
    int*   d_found  = nullptr;
    cudaMalloc(&d_keys,   keys.size() * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values, keys.size() * sizeof(Value)) >> CUDA_CHECK;
    cudaMalloc(&d_found,  keys.size() * sizeof(int))   >> CUDA_CHECK;
    cudaMemcpy(d_keys, keys.data(), keys.size() * sizeof(Key),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;

    constexpr int block_size = 128;
    const int tiles_per_block = block_size / Table::tile_size;
    const int grid_size =
        static_cast<int>((keys.size() + tiles_per_block - 1) / tiles_per_block);
    get_many_kernel<Table><<<grid_size, block_size>>>(
        table.view(), d_keys, d_values, d_found, keys.size());
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    cudaMemcpy(values_out.data(), d_values, keys.size() * sizeof(Value),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    cudaMemcpy(found_out.data(), d_found, keys.size() * sizeof(int),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_found);
}

// Robin Hood invariant: for every occupied slot at index i, the resident's
// home bucket H_R satisfies (i / bucket_size - H_R) mod num_buckets ≤
// max_probe_buckets, AND every bucket strictly between H_R and i /
// bucket_size is fully occupied (no empty slot). The second condition is
// the key invariant: an empty slot at distance < the resident's actual
// probe would mean the resident should have been placed there instead.
//
// The unused `table` parameter is just for Table type deduction at call sites.
template <class Table>
inline void assert_robin_hood_invariant(const Table&,
                                        const std::vector<typename Table::Slot>& state) {
    constexpr auto B = static_cast<std::size_t>(Table::bucket_size);
    const auto num_buckets = state.size() / B;
    const auto cap_mask = state.size() - 1;

    std::vector<bool> bucket_has_empty(num_buckets, false);
    for (std::size_t b = 0; b < num_buckets; ++b) {
        for (std::size_t s = 0; s < B; ++s) {
            if (state[b * B + s].key == Table::empty_key) {
                bucket_has_empty[b] = true;
                break;
            }
        }
    }

    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i].key == Table::empty_key) continue;

        const std::size_t home_bucket = (state[i].key & cap_mask) / B;
        const std::size_t this_bucket = i / B;
        const std::size_t probe =
            (this_bucket + num_buckets - home_bucket) % num_buckets;

        assert(probe < static_cast<std::size_t>(Table::max_probe_buckets));

        for (std::size_t step = 0; step < probe; ++step) {
            const std::size_t b = (home_bucket + step) % num_buckets;
            assert(!bucket_has_empty[b] &&
                   "Robin Hood invariant violated: an earlier bucket has an "
                   "empty slot that this resident could have occupied");
        }
    }
}
