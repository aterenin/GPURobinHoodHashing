// Minimal example of using gpurhh.
//
// Constructs a HashTable on the host, launches a small kernel to insert a
// handful of (key, value) pairs, launches a second small kernel to look
// them back up, prints the round-trip, and finally pretty-prints the
// table's memory layout.

// Pull in the diagnostic pretty-printer at the end of the round-trip. It
// touches the table's internal bucket array, so we opt into the gated
// data() accessor before including any gpurhh header.
#define GPURHH_ENABLE_INTERNAL_ACCESS 1

#include <gpurhh/hash_table.cuh>
#include <gpurhh/print.cuh>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

using Table = gpurhh::HashTable<std::uint32_t, std::uint32_t>;
using View  = Table::View;

// One tile per (key, value) pair; the block size is sized so each tile
// has exactly one input to process.
__global__ void insert_kernel(View view,
                              const std::uint32_t* keys,
                              const std::uint32_t* values,
                              std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const std::size_t tile_id = threadIdx.x / Table::tile_size;
    if (tile_id < n) {
        view.insert(tile, keys[tile_id], values[tile_id]);
    }
}

// Same shape as the insert kernel: one tile per query.
__global__ void get_kernel(View view,
                           const std::uint32_t* keys,
                           std::uint32_t* values_out,
                           int* found_out,
                           std::size_t n)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);
    const std::size_t tile_id = threadIdx.x / Table::tile_size;
    if (tile_id < n) {
        const auto result = view.get(tile, keys[tile_id]);
        if (tile.thread_rank() == 0) {
            values_out[tile_id] = result.value_or(0);
            found_out[tile_id]  = result.has_value() ? 1 : 0;
        }
    }
}

int main() {
    constexpr std::size_t capacity = 64;
    Table table(capacity);

    constexpr std::size_t N = 4;
    const std::uint32_t h_keys[N]   = { 42, 100, 7, 256 };
    const std::uint32_t h_values[N] = { 1000, 2000, 3000, 4000 };

    // Stage the inputs on device.
    std::uint32_t *d_keys = nullptr, *d_values = nullptr;
    cudaMalloc(&d_keys,   N * sizeof(std::uint32_t));
    cudaMalloc(&d_values, N * sizeof(std::uint32_t));
    cudaMemcpy(d_keys,   h_keys,   N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    // Launch one block, N tiles. tile_size is 16 for our default
    // (uint32, uint32) instantiation, so block size = N * 16 = 64 threads.
    const int block_size = static_cast<int>(N * Table::tile_size);
    insert_kernel<<<1, block_size>>>(table.view(), d_keys, d_values, N);
    cudaDeviceSynchronize();

    // Read the inserted pairs back.
    std::uint32_t* d_values_out = nullptr;
    int*           d_found_out  = nullptr;
    cudaMalloc(&d_values_out, N * sizeof(std::uint32_t));
    cudaMalloc(&d_found_out,  N * sizeof(int));
    get_kernel<<<1, block_size>>>(table.view(), d_keys, d_values_out, d_found_out, N);
    cudaDeviceSynchronize();

    std::uint32_t h_values_out[N];
    int           h_found_out[N];
    cudaMemcpy(h_values_out, d_values_out, N * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_found_out,  d_found_out,  N * sizeof(int),           cudaMemcpyDeviceToHost);

    std::printf("Round-trip insert + get:\n");
    for (std::size_t i = 0; i < N; ++i) {
        std::printf("  key=%u  inserted=%u  retrieved=%u  found=%d\n",
                    h_keys[i], h_values[i], h_values_out[i], h_found_out[i]);
    }
    std::printf("\n");

    // Dump the whole table.
    gpurhh::print_slots(table, 0, table.capacity());

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_values_out);
    cudaFree(d_found_out);
    return 0;
}
