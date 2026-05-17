#pragma once

// Shared infrastructure for gpurhh tests: a concrete table instantiation, the
// small driver kernels that read keys/values from device buffers, and
// host-side helpers that wrap allocation, copy, launch, and result-readback
// around those kernels. CUDA error checking and the test-only macro switch
// live in tests.cuh.

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// tests.cuh must precede the gpurhh header — it sets the
// GPURHH_ENABLE_INTERNAL_ACCESS macro that exposes HashTable::data().
#include "tests.cuh"
#include <gpurhh/hash_table.cuh>

namespace cg = cooperative_groups;

// Concrete table instantiation used throughout the test suite.
using Table = gpurhh::HashTable<std::uint32_t, std::uint32_t>;
using View  = Table::View;

inline constexpr int kBlockSize = 128;
static_assert(kBlockSize % Table::tile_size == 0,
              "block size must be a whole multiple of tile size");

// --- Driver kernels ------------------------------------------------------
//
// One cooperative tile of Table::tile_size threads handles one (key, value)
// pair. Tiles are tile-strided across the grid so batches larger than the
// grid still work.

__global__ void insert(View view,
                                     const std::uint32_t* keys,
                                     const std::uint32_t* values,
                                     std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles = gridDim.x * tiles_per_block;

    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        view.insert(tile, keys[i], values[i]);
    }
}

// For each input key, writes (value, 1) on hit and (_, 0) on miss.
// view.get returns the same cuda::std::optional<Value> on every lane in the
// tile; lane 0 performs the store to keep the store count to one per key.
__global__ void get(View view,
                                  const std::uint32_t* keys,
                                  std::uint32_t* values_out,
                                  int* found_out,
                                  std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles = gridDim.x * tiles_per_block;

    for (std::size_t i = tile_id; i < n; i += total_tiles) {
        const auto result = view.get(tile, keys[i]);
        if (tile.thread_rank() == 0) {
            values_out[i] = result.value_or(0);
            found_out[i]  = result.has_value() ? 1 : 0;
        }
    }
}

// --- Host-side driver helpers -------------------------------------------

inline void bulk_insert(Table& table,
                        const std::uint32_t* h_keys,
                        const std::uint32_t* h_values,
                        std::size_t n)
{
    std::uint32_t *d_keys = nullptr, *d_values = nullptr;
    cudaMalloc(&d_keys,   n * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_values, n * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMemcpy(d_keys,   h_keys,   n * sizeof(std::uint32_t),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;
    cudaMemcpy(d_values, h_values, n * sizeof(std::uint32_t),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;

    const int tiles_per_block = kBlockSize / Table::tile_size;
    const int grid_size =
        static_cast<int>((n + tiles_per_block - 1) / tiles_per_block);
    insert<<<grid_size, kBlockSize>>>(table.view(), d_keys, d_values, n);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    cudaFree(d_keys);
    cudaFree(d_values);
}

inline void bulk_get(Table& table,
                     const std::uint32_t* h_keys,
                     std::uint32_t* h_values_out,
                     int* h_found_out,
                     std::size_t n)
{
    std::uint32_t *d_keys = nullptr, *d_values_out = nullptr;
    int* d_found_out = nullptr;
    cudaMalloc(&d_keys,       n * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_values_out, n * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_found_out,  n * sizeof(int))           >> CUDA_CHECK;
    cudaMemcpy(d_keys, h_keys, n * sizeof(std::uint32_t),
               cudaMemcpyHostToDevice) >> CUDA_CHECK;

    const int tiles_per_block = kBlockSize / Table::tile_size;
    const int grid_size =
        static_cast<int>((n + tiles_per_block - 1) / tiles_per_block);
    get<<<grid_size, kBlockSize>>>(
        table.view(), d_keys, d_values_out, d_found_out, n);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    cudaMemcpy(h_values_out, d_values_out, n * sizeof(std::uint32_t),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    cudaMemcpy(h_found_out,  d_found_out,  n * sizeof(int),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;

    cudaFree(d_keys);
    cudaFree(d_values_out);
    cudaFree(d_found_out);
}
