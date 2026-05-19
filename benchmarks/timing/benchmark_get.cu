// Get benchmark for gpurhh — comparison version.
//
// Mirrors benchmark_insert.cu's apples-to-apples comparison setup:
//   - replace_op reduction (no extra work on duplicate keys at pre-fill).
//   - MaxProbeBuckets effectively uncapped.
//   - No counter instrumentation.
//
// Workload structure:
//   1. Generate insert keys with a single cuRAND generator seeded by
//      --seed (on GPU, untimed) and pre-fill the table once.
//   2. For each timed rep, continue the same generator to draw a fresh
//      n_ops get-key stream (in setup, untimed), then run and time the
//      get kernel. Reps thus see iid input draws rather than 16 trials
//      on one frozen input. The pre-fill is left one-shot since we are
//      measuring get throughput; pre-fill input variability is a
//      second-order effect for uniform keys at fixed F.
//
// The two key streams share no draws — the generator advances
// sequentially — so one seed cleanly parameterizes the whole experiment.
// All derived metrics live in the analysis script.

#include "../benchmarks.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

namespace cg = cooperative_groups;

using Table = gpurhh::HashTable<
    std::uint32_t,
    std::uint32_t,
    gpurhh::default_hash<std::uint32_t>,
    gpurhh::default_empty_key<std::uint32_t>::key,
    gpurhh::replace_op,
    /*CacheLineBytes=*/128,
    /*WarpSize=*/32,
    /*MaxProbeBuckets=*/(1 << 20)>;
using View = Table::View;

namespace {

struct Args {
    std::size_t   capacity   = std::size_t{1} << 27;
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
     .add("--n-ops",      a.n_ops,      "Number of get attempts (also pre-fill count)")
     .add("--block-size", a.block_size, "Threads per CUDA block")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed; one stream produces insert then get keys")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. get.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty())                     p.print_usage();
    if (a.n_ops == 0)                             { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.block_size % Table::tile_size != 0)     { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024) { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
    return a;
}

// Pre-fill kernel (untimed). Reuses the same insert path the comparison
// version uses elsewhere. No counter tracking.
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

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const std::uint32_t v = gpurhh::detail::fmix32(k);
        (void) view.insert(tile, k, v);
    }
}

__global__ void get(
    View view,
    const std::uint32_t* __restrict__ keys,
    std::uint32_t* __restrict__ out,
    std::size_t n_ops)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles =
        std::size_t{gridDim.x} * tiles_per_block;

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const auto result = view.get(tile, k);
        if (tile.thread_rank() == 0) {
            out[op_id] = result.value_or(~std::uint32_t{0});
        }
    }
}

std::string format_row(
    std::size_t capacity, std::size_t n_ops,
    int block_size, std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,get,"
      << tag           << ","
      << rep           << ","
      << seed          << ","
      << capacity      << ","
      << n_ops         << ","
      << block_size    << ","
      << slot_bytes    << ","
      << bytes_per_op  << ","
      << time_ms;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    constexpr std::size_t slot_bytes   = sizeof(Table::Slot);
    constexpr std::size_t bytes_per_op = sizeof(Table::Bucket);

    Table table(args.capacity);
    const std::size_t capacity = table.capacity();

    std::fprintf(stderr,
        "[get] capacity=%zu n_ops=%zu "
        "block_size=%d warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        capacity, args.n_ops,
        args.block_size, args.warmups, args.reps,
        args.seed, args.tag.c_str());

    // One cuRAND generator, two sequential disjoint streams: first n_ops
    // draws become the insert (pre-fill) keys, next n_ops draws become
    // the get keys. Both fills happen on the GPU before any timed work
    // begins, so they are not part of the measurement.
    UniformKeyGenerator gen(args.seed);

    // --- pre-fill (untimed) ---
    std::uint32_t* d_insert_keys = nullptr;
    cudaMalloc(&d_insert_keys, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    fill_uniform_keys(d_insert_keys, args.n_ops, gen);

    const auto shape = compute_launch_shape(args.block_size, Table::tile_size);

    prefill<<<shape.grid_size, args.block_size>>>(
        table.view(), d_insert_keys, args.n_ops);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;
    cudaFree(d_insert_keys);

    // --- get key buffer (refilled in setup per rep from the continuing
    //     generator stream, so reps see iid get-key draws) ---
    std::uint32_t* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;

    std::uint32_t* d_out = nullptr;
    cudaMalloc(&d_out, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;

    Recorder rec(args.output_dir / "get.csv",
        "library,workload,tag,rep,seed,capacity,"
        "n_ops,block_size,slot_bytes,bytes_per_op,time_ms");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            fill_uniform_keys(d_get_keys, args.n_ops, gen);
        },
        /*launch=*/ [&]() {
            get<<<shape.grid_size, args.block_size>>>(
                table.view(), d_get_keys, d_out, args.n_ops);
            cudaGetLastError() >> CUDA_CHECK;
        },
        /*after=*/  [&](int rep, float ms) {
            rec.write_row(format_row(
                capacity, args.n_ops,
                args.block_size, slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    return 0;
}
