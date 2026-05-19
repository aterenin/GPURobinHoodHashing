// Insert benchmark for gpurhh — counter-instrumented version.
//
// Studies gpurhh's design in its full configuration:
//   - Reduction operator: sum_op (exercise the reduction path).
//   - MaxProbeBuckets: 8 (the default — study the probe-cap regime).
//   - Per-tile probe / failure counters via GPURHH_BENCHMARK_COUNTERS.
//
// Writes to insert_counters.csv (separate from the apples-to-apples
// insert.csv that benchmark_insert.cu produces). The extra counter
// columns (total_probes, total_failures) live only here.

#define GPURHH_BENCHMARK_COUNTERS 1

#include "benchmarks.cuh"

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

[[noreturn]] void die_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --output-dir DIR [options]\n"
        "  Same shape as benchmark_insert. Writes insert_counters.csv with\n"
        "  per-rep total_probes and total_failures alongside time_ms.\n",
        prog);
    std::exit(1);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto get_val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", flag.c_str()); die_usage(argv[0]); }
            return argv[++i];
        };
        if      (flag == "--capacity")   a.capacity   = std::stoull(get_val());
        else if (flag == "--key-range")  a.key_range  = std::stoull(get_val());
        else if (flag == "--n-ops")      a.n_ops      = std::stoull(get_val());
        else if (flag == "--block-size") a.block_size = std::stoi(get_val());
        else if (flag == "--warmups")    a.warmups    = std::stoi(get_val());
        else if (flag == "--reps")       a.reps       = std::stoi(get_val());
        else if (flag == "--seed")       a.seed       = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--tag")        a.tag        = get_val();
        else if (flag == "--output-dir") a.output_dir = get_val();
        else { std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str()); die_usage(argv[0]); }
    }
    if (a.output_dir.empty())                     die_usage(argv[0]);
    if (a.n_ops == 0)                             { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.key_range == 0)                         { std::fprintf(stderr, "--key-range must be > 0\n"); std::exit(1); }
    if (a.block_size % Table::tile_size != 0)     { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024) { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
    return a;
}

__global__ void insert(
    View view,
    const std::uint32_t* __restrict__ keys,
    std::size_t n_ops,
    std::uint32_t* __restrict__ per_tile_probes,
    std::uint32_t* __restrict__ per_tile_failures)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles =
        std::size_t{gridDim.x} * tiles_per_block;

    std::uint32_t my_probes   = 0;
    std::uint32_t my_failures = 0;

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const std::uint32_t v = gpurhh::detail::fmix32(k);
        const auto leftover = view.insert(tile, k, v, my_probes);
        if (leftover.has_value() && tile.thread_rank() == 0) {
            ++my_failures;
        }
    }

    if (tile.thread_rank() == 0) {
        per_tile_probes  [tile_id] = my_probes;
        per_tile_failures[tile_id] = my_failures;
    }
}

std::string format_row(
    std::size_t capacity, std::size_t key_range, std::size_t n_ops,
    int block_size, std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, std::uint64_t total_probes, std::uint64_t total_failures)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,insert,"
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
      << total_failures;
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
        "[insert-counters] capacity=%zu key_range=%zu n_ops=%zu "
        "block_size=%d warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        capacity, args.key_range, args.n_ops,
        args.block_size, args.warmups, args.reps, args.seed, args.tag.c_str());

    std::uint32_t* d_keys = nullptr;
    cudaMalloc(&d_keys, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    UniformKeyGenerator gen(args.seed);

    const auto shape = compute_launch_shape(args.block_size, Table::tile_size);
    const std::size_t n_tiles = shape.n_tiles;

    std::uint32_t* d_probes   = nullptr;
    std::uint32_t* d_failures = nullptr;
    cudaMalloc(&d_probes,   n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_failures, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    std::vector<std::uint32_t> h_probes  (n_tiles);
    std::vector<std::uint32_t> h_failures(n_tiles);

    Recorder rec(args.output_dir / "insert_counters.csv",
        "library,workload,tag,rep,seed,capacity,key_range,n_ops,block_size,"
        "slot_bytes,bytes_per_op,time_ms,total_probes,total_failures");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            table.clear();
            fill_uniform_keys(d_keys, args.n_ops, key_range, gen);
            cudaMemset(d_probes,   0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
            cudaMemset(d_failures, 0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
        },
        /*launch=*/ [&]() {
            insert<<<shape.grid_size, args.block_size>>>(
                table.view(), d_keys, args.n_ops, d_probes, d_failures);
            cudaGetLastError() >> CUDA_CHECK;
        },
        /*after=*/  [&](int rep, float ms) {
            cudaMemcpy(h_probes.data(),   d_probes,
                       n_tiles * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost) >> CUDA_CHECK;
            cudaMemcpy(h_failures.data(), d_failures,
                       n_tiles * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost) >> CUDA_CHECK;
            const std::uint64_t total_probes =
                std::accumulate(h_probes.begin(), h_probes.end(), std::uint64_t{0});
            const std::uint64_t total_failures =
                std::accumulate(h_failures.begin(), h_failures.end(), std::uint64_t{0});
            rec.write_row(format_row(
                capacity, args.key_range, args.n_ops,
                args.block_size, slot_bytes, bytes_per_op,
                args.seed, rep, args.tag,
                ms, total_probes, total_failures));
        });

    cudaFree(d_keys);
    cudaFree(d_probes);
    cudaFree(d_failures);
    return 0;
}
