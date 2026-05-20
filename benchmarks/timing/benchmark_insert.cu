// Insert benchmark for gpurhh — comparison version.
//
// Apples-to-apples with the cuco / warpcore baselines:
//   - Reduction operator: replace_op (no extra work on duplicate keys).
//   - MaxProbeBuckets: 1 << 20 (effectively uncapped, matching the
//     baselines' unbounded probe semantics).
//   - No counter instrumentation in the hot path.
//   - Values pre-materialized in a separate d_values buffer that the
//     timed insert kernel reads alongside d_keys. Mirrors warpcore's
//     API shape and makes cuco's pair-iterator path do the same work.
//     Under replace_op the *content* of the values is irrelevant to
//     timing (the insert kernel writes whatever it reads), so we just
//     zero d_values once before the loop and skip the per-rep derive.
//     The memory benchmark under benchmarks/memory/ keeps the inline
//     `v = fmix32(k)` pattern since it uses sum_op, where the actual
//     reduction work is what's being instrumented.
//
// Workload:
//   - Key type uint32_t, value type uint32_t.
//   - Keys drawn from Uniform(0, 2^32) — i.e. raw cuRAND output, no
//     clamping. With capacity 2^27 this gives α ≈ 32 (essentially
//     unique-keys), which is the regime where collision-resolution
//     strategy actually matters. See notes/design.md for the α=1
//     duplicate-stress alternative.
//   - Single ~1 GiB table by default, DRAM-resident.
//
// Per timed sample:
//   1. Empty the table, refill d_keys (untimed setup).
//   2. Sort+unique a scratch copy of d_keys to count distinct inputs
//      (untimed). This is the ground-truth denominator for drop
//      accounting: at α≈32 most uint32 draws are unique, but a small
//      fraction are duplicates, and we don't want input duplicates to
//      show up as "drops" — only true insert failures should.
//   3. Time one kernel that processes n_ops insert ops in a grid-stride loop.
//   4. After timing, walk the table and count occupied slots (untimed).
//      Report drops = unique_inputs − occupied_after.
//   5. Emit a CSV row with the raw configuration, measured time, and
//      `drops` column.
//
// All derived metrics (ops_per_sec, gbps_floor, load_factor, occupancy,
// drop_rate, ...) are computed in the analysis script from the raw
// CSV columns, not in this binary.

// We use HashTable::data() to reinterpret the bucket storage as a flat
// Slot array for the post-insert occupancy scan.

#include "../benchmarks.cuh"

#include <thrust/sort.h>
#include <thrust/unique.h>

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
    std::size_t   capacity   = std::size_t{1} << 27;  // 1 GiB at 8 B/slot
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
    p.add("--capacity",   a.capacity,   "Table size in slots (rounded up to next pow2)")
     .add("--n-ops",      a.n_ops,      "Number of insert attempts")
     .add("--block-size", a.block_size, "Threads per CUDA block; multiple of tile_size")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed for key generation")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. insert.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty())                     p.print_usage();
    if (a.n_ops == 0)                             { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.block_size % Table::tile_size != 0)     { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024) { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
    return a;
}

__global__ void insert(
    View view,
    const std::uint32_t* __restrict__ keys,
    const std::uint32_t* __restrict__ values,
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
        const std::uint32_t v = values[op_id];
        (void) view.insert(tile, k, v);
    }
}

std::string format_row(
    std::size_t capacity, std::size_t n_ops,
    int block_size, std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, std::size_t drops)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,insert,"
      << tag           << ","
      << rep           << ","
      << seed          << ","
      << capacity      << ","
      << n_ops         << ","
      << block_size    << ","
      << slot_bytes    << ","
      << bytes_per_op  << ","
      << time_ms       << ","
      << drops;
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
        "[insert] capacity=%zu n_ops=%zu "
        "block_size=%d warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        capacity, args.n_ops,
        args.block_size, args.warmups, args.reps, args.seed, args.tag.c_str());

    // --- input buffers ---
    // d_keys:   refreshed per rep in setup via cuRAND (cheap, iid draws).
    // d_values: zeroed once below. Under replace_op the value content
    //           doesn't affect work; we only need the buffer present so
    //           the timed kernel pays the input read.
    // d_sorted: scratch copy of d_keys, sorted in setup and uniqued to
    //           count distinct inputs for drop accounting.
    std::uint32_t* d_keys   = nullptr;
    std::uint32_t* d_values = nullptr;
    std::uint32_t* d_sorted = nullptr;
    cudaMalloc(&d_keys,   args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_values, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_sorted, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMemset(d_values, 0, args.n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    UniformKeyGenerator gen(args.seed);

    const auto shape = compute_launch_shape(args.block_size, Table::tile_size);

    Recorder rec(args.output_dir / "insert.csv",
        "library,workload,tag,rep,seed,capacity,n_ops,block_size,"
        "slot_bytes,bytes_per_op,time_ms,drops");

    EventTimer timer;

    // Ground-truth unique input count for the current rep, set in setup
    // and read in after.
    std::size_t n_unique = 0;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            table.clear();
            fill_uniform_keys(d_keys, args.n_ops, gen);
            // Sort + unique on a scratch copy to count distinct inputs.
            cudaMemcpy(d_sorted, d_keys,
                       args.n_ops * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToDevice) >> CUDA_CHECK;
            thrust::sort(thrust::device, d_sorted, d_sorted + args.n_ops);
            auto end = thrust::unique(thrust::device, d_sorted,
                                      d_sorted + args.n_ops);
            n_unique = end - d_sorted;
        },
        /*launch=*/ [&]() {
            insert<<<shape.grid_size, args.block_size>>>(
                table.view(), d_keys, d_values, args.n_ops);
            cudaGetLastError() >> CUDA_CHECK;
        },
        /*after=*/  [&](int rep, float ms) {
            // Count occupied slots in the table and derive drops.
            const auto* slots =
                reinterpret_cast<const Table::Slot*>(table.data());
            const std::size_t occupied = count_occupied_slots(
                slots, capacity, Table::empty_key);
            const std::size_t drops =
                n_unique > occupied ? n_unique - occupied : 0;
            rec.write_row(format_row(
                capacity, args.n_ops,
                args.block_size, slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms, drops));
        });

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_sorted);
    return 0;
}
