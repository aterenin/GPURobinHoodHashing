// Get benchmark for gpurhh — memory-utilization study.
//
// Mirrors benchmarks/timing/benchmark_get.cu's structure but with:
//   - sum_op (exercises the reduction path during pre-fill).
//   - Default MaxProbeBuckets (= 8) — studies the probe-cap regime.
//   - Per-tile probe / hit counters via GPURHH_BENCHMARK_COUNTERS.
//
// Writes get.csv with the same configuration columns as the timing
// build plus total_probes, total_hits, total_misses.

#define GPURHH_BENCHMARK_COUNTERS 1

#include "../benchmarks.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

using Table = gpurhh::HashTable<
    std::uint32_t,
    std::uint32_t,
    gpurhh::default_hash<std::uint32_t>,
    gpurhh::default_empty_key<std::uint32_t>::key,
    gpurhh::sum_op>;
using View = Table::View;

namespace {

struct Args {
    std::size_t   capacity   = std::size_t{1} << 27;
    std::size_t   key_range  = std::size_t{1} << 27;
    std::size_t   n_ops      = std::size_t{1} << 27;
    int           block_size = 256;
    int           warmups    = 2;
    int           reps       = 16;
    std::uint32_t seed       = 42;
    std::string   tag        = "";
    std::filesystem::path output_dir;
};

Args parse_args(int argc, char** argv) {
    Args a;
    ArgParser p(argv[0]);
    p.add("--capacity",   a.capacity,   "Table size in slots")
     .add("--key-range",  a.key_range,  "N in Uniform(0, N) for key generation")
     .add("--n-ops",      a.n_ops,      "Number of get attempts (also pre-fill count)")
     .add("--block-size", a.block_size, "Threads per CUDA block; multiple of tile_size")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed; one stream produces insert then get keys")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. get.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty())                     p.print_usage();
    if (a.n_ops == 0)                             { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.key_range == 0)                         { std::fprintf(stderr, "--key-range must be > 0\n"); std::exit(1); }
    if (a.block_size % Table::tile_size != 0)     { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024) { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
    return a;
}

__global__ void prefill(
    View view,
    const std::uint32_t* __restrict__ keys,
    std::size_t n_ops)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles =
        std::size_t{gridDim.x} * tiles_per_block;

    std::uint32_t scratch_probes = 0;  // discarded; only counts during prefill

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const std::uint32_t v = gpurhh::detail::fmix32(k);
        (void) view.insert(tile, k, v, scratch_probes);
    }
}

__global__ void get(
    View view,
    const std::uint32_t* __restrict__ keys,
    std::uint32_t* __restrict__ out,
    std::size_t n_ops,
    std::uint32_t* __restrict__ per_tile_probes,
    std::uint32_t* __restrict__ per_tile_hits)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles =
        std::size_t{gridDim.x} * tiles_per_block;

    std::uint32_t my_probes = 0;
    std::uint32_t my_hits   = 0;

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const auto result = view.get(tile, k, my_probes);
        if (tile.thread_rank() == 0) {
            out[op_id] = result.value_or(~std::uint32_t{0});
            if (result.has_value()) ++my_hits;
        }
    }

    if (tile.thread_rank() == 0) {
        per_tile_probes[tile_id] = my_probes;
        per_tile_hits  [tile_id] = my_hits;
    }
}

std::string format_row(
    std::size_t capacity, std::size_t key_range, std::size_t n_ops,
    int block_size, std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms,
    std::uint64_t total_probes, std::uint64_t total_hits, std::uint64_t total_misses)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,get,"
      << tag           << ","
      << rep           << ","
      << seed          << ","
      << capacity      << ","
      << key_range     << ","
      << n_ops         << ","
      << block_size    << ","
      << slot_bytes    << ","
      << bytes_per_op  << ","
      << time_ms       << ","
      << total_probes  << ","
      << total_hits    << ","
      << total_misses;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    constexpr std::size_t slot_bytes   = sizeof(Table::Slot);
    constexpr std::size_t bytes_per_op = sizeof(Table::Bucket);

    Table table(args.capacity);
    const std::size_t capacity = table.capacity();
    const std::uint32_t key_range = static_cast<std::uint32_t>(args.key_range);

    std::fprintf(stderr,
        "[memory get] capacity=%zu key_range=%zu n_ops=%zu "
        "block_size=%d warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        capacity, args.key_range, args.n_ops,
        args.block_size, args.warmups, args.reps,
        args.seed, args.tag.c_str());

    // Same single-generator scheme as benchmark_get.cu: n_ops draws for
    // the insert stream, then n_ops more draws for the get stream.
    UniformKeyGenerator gen(args.seed);

    // --- pre-fill (untimed) ---
    std::uint32_t* d_insert_keys = nullptr;
    cudaMalloc(&d_insert_keys, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    fill_uniform_keys(d_insert_keys, args.n_ops, key_range, gen);

    const auto shape = compute_launch_shape(args.block_size, Table::tile_size);
    const std::size_t n_tiles = shape.n_tiles;

    prefill<<<shape.grid_size, args.block_size>>>(
        table.view(), d_insert_keys, args.n_ops);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;
    cudaFree(d_insert_keys);

    // --- get keys (refilled in setup per rep) ---
    std::uint32_t* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;

    std::uint32_t* d_out = nullptr;
    cudaMalloc(&d_out, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;

    std::uint32_t* d_probes = nullptr;
    std::uint32_t* d_hits   = nullptr;
    cudaMalloc(&d_probes, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_hits,   n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    std::vector<std::uint32_t> h_probes(n_tiles);
    std::vector<std::uint32_t> h_hits  (n_tiles);

    Recorder rec(args.output_dir / "get.csv",
        "library,workload,tag,rep,seed,capacity,key_range,"
        "n_ops,block_size,slot_bytes,bytes_per_op,time_ms,"
        "total_probes,total_hits,total_misses");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            fill_uniform_keys(d_get_keys, args.n_ops, key_range, gen);
            cudaMemset(d_probes, 0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
            cudaMemset(d_hits,   0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
        },
        /*launch=*/ [&]() {
            get<<<shape.grid_size, args.block_size>>>(
                table.view(), d_get_keys, d_out, args.n_ops,
                d_probes, d_hits);
            cudaGetLastError() >> CUDA_CHECK;
        },
        /*after=*/  [&](int rep, float ms) {
            cudaMemcpy(h_probes.data(), d_probes,
                       n_tiles * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost) >> CUDA_CHECK;
            cudaMemcpy(h_hits.data(),   d_hits,
                       n_tiles * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost) >> CUDA_CHECK;
            const std::uint64_t total_probes =
                std::accumulate(h_probes.begin(), h_probes.end(), std::uint64_t{0});
            const std::uint64_t total_hits =
                std::accumulate(h_hits.begin(),   h_hits.end(),   std::uint64_t{0});
            const std::uint64_t total_misses =
                std::uint64_t{args.n_ops} - total_hits;
            rec.write_row(format_row(
                capacity, args.key_range, args.n_ops,
                args.block_size, slot_bytes, bytes_per_op,
                args.seed, rep, args.tag,
                ms, total_probes, total_hits, total_misses));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    cudaFree(d_probes);
    cudaFree(d_hits);
    return 0;
}
