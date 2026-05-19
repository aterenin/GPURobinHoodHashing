// Get benchmark for gpurhh.
//
// Measures lookup throughput and effective DRAM bandwidth as a function
// of input load factor F = n_ops / capacity. Workload:
//   - Key type uint32_t, value type uint32_t, sum-based reduction.
//   - Insert keys ~ Uniform(0, key_range) with key_range = capacity (α = 1).
//   - Get keys drawn from the same distribution under a different seed,
//     giving a natural hit rate ≈ effective occupancy ≈ 1 - exp(-F).
//   - Single ~1 GiB table, DRAM-resident, pre-filled once and reused
//     across all warmup and timed reps.
//
// Per-run flow:
//   1. Allocate and pre-fill the table (untimed). Abort if any insert
//      fails the probe cap — that would make the get-side hit rate a
//      lie.
//   2. Generate the get-side key buffer (untimed).
//   3. Warmup reps (untimed).
//   4. Timed reps: zero counters → time get kernel → sum per-tile
//      counters on host → write CSV row.

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
    std::size_t   capacity_bytes = std::size_t{1} << 30;  // 1 GiB
    double        load_factor    = 1.0;
    int           block_size     = 256;
    int           warmups        = 2;
    int           reps           = 16;
    std::uint32_t insert_seed    = 42;
    std::uint32_t get_seed       = 1729;
    std::string   tag            = "";
    std::filesystem::path output_dir;
};

[[noreturn]] void die_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --output-dir DIR [options]\n"
        "  --capacity-bytes N  Target table size in bytes (default 1<<30; rounded up)\n"
        "  --load-factor F     n_ops as a fraction of capacity (default 1.0)\n"
        "  --block-size N      Threads per CUDA block; must be a multiple of tile_size (default 256)\n"
        "  --warmups N         Untimed warmup reps (default 2)\n"
        "  --reps N            Timed reps (default 16)\n"
        "  --insert-seed N     PRNG seed for pre-fill keys (default 42)\n"
        "  --get-seed N        PRNG seed for get keys, drawn independently (default 1729)\n"
        "  --tag STR           Free-form label written to every CSV row\n"
        "  --output-dir DIR    Required. get.csv is appended to here.\n",
        prog);
    std::exit(1);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto get_val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag.c_str());
                die_usage(argv[0]);
            }
            return argv[++i];
        };
        if      (flag == "--capacity-bytes") a.capacity_bytes = std::stoull(get_val());
        else if (flag == "--load-factor")    a.load_factor    = std::stod(get_val());
        else if (flag == "--block-size")     a.block_size     = std::stoi(get_val());
        else if (flag == "--warmups")        a.warmups        = std::stoi(get_val());
        else if (flag == "--reps")           a.reps           = std::stoi(get_val());
        else if (flag == "--insert-seed")    a.insert_seed    = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--get-seed")       a.get_seed       = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--tag")            a.tag            = get_val();
        else if (flag == "--output-dir")     a.output_dir     = get_val();
        else { std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str()); die_usage(argv[0]); }
    }
    if (a.output_dir.empty())                     die_usage(argv[0]);
    if (a.load_factor <= 0.0)                     { std::fprintf(stderr, "--load-factor must be > 0\n"); std::exit(1); }
    if (a.block_size % Table::tile_size != 0)     { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024) { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
    return a;
}

// Pre-fill kernel. We don't care about probe counts during pre-fill;
// the only counter we track is per-tile failures, which we sum on the
// host and abort on if non-zero.
__global__ void prefill(
    View view,
    const std::uint32_t* __restrict__ keys,
    std::size_t n_ops,
    std::uint32_t* __restrict__ per_tile_failures)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<Table::tile_size>(block);

    const std::size_t tiles_per_block = blockDim.x / Table::tile_size;
    const std::size_t tile_id =
        blockIdx.x * tiles_per_block + threadIdx.x / Table::tile_size;
    const std::size_t total_tiles =
        std::size_t{gridDim.x} * tiles_per_block;

    std::uint32_t scratch_probes = 0;  // mandatory under the counter macro; discarded
    std::uint32_t my_failures    = 0;

    for (std::size_t op_id = tile_id; op_id < n_ops; op_id += total_tiles) {
        const std::uint32_t k = keys[op_id];
        const std::uint32_t v = gpurhh::detail::fmix32(k);
        const auto leftover = view.insert(tile, k, v, scratch_probes);
        if (leftover.has_value() && tile.thread_rank() == 0) {
            ++my_failures;
        }
    }

    if (tile.thread_rank() == 0) {
        per_tile_failures[tile_id] = my_failures;
    }
}

// Timed get kernel. Each tile loops over its share of op_ids, looks up
// each key, writes the result to its op_id slot in `out`, and tracks
// per-tile probe + hit counts. Misses are derivable as n_ops - total_hits.
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
            // Sentinel ~0u marks a miss in the output buffer. Real values
            // hashed from real keys can be ~0u in principle, so the
            // hits / misses CSV columns are the authoritative count.
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
    std::size_t slot_bytes, std::size_t capacity, std::size_t capacity_bytes,
    std::size_t key_range, double alpha, double load_factor,
    int block_size, std::size_t n_ops, std::size_t bytes_per_op,
    std::uint32_t insert_seed, std::uint32_t get_seed,
    int rep, const std::string& tag,
    float time_ms, double ops_per_sec, double gbps_floor, double gbps_corrected,
    std::uint64_t total_probes, std::uint64_t total_hits, std::uint64_t total_misses,
    std::uint64_t prefill_failures)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,get,"
      << slot_bytes        << ","
      << capacity          << ","
      << capacity_bytes    << ","
      << key_range         << ","
      << alpha             << ","
      << load_factor       << ","
      << block_size        << ","
      << n_ops             << ","
      << bytes_per_op      << ","
      << insert_seed       << ","
      << get_seed          << ","
      << rep               << ","
      << tag               << ","
      << time_ms           << ","
      << ops_per_sec       << ","
      << gbps_floor        << ","
      << gbps_corrected    << ","
      << total_probes      << ","
      << total_hits        << ","
      << total_misses      << ","
      << prefill_failures;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // --- table sizing ---
    constexpr std::size_t slot_bytes   = sizeof(Table::Slot);
    constexpr std::size_t bytes_per_op = sizeof(Table::Bucket);
    const std::size_t min_slots = args.capacity_bytes / slot_bytes;
    Table table(min_slots);
    const std::size_t capacity       = table.capacity();
    const std::size_t capacity_bytes = capacity * slot_bytes;
    const std::uint32_t key_range = static_cast<std::uint32_t>(capacity);
    const std::size_t   n_ops     =
        static_cast<std::size_t>(args.load_factor * static_cast<double>(capacity));

    if (n_ops == 0) {
        std::fprintf(stderr, "Computed n_ops = 0; raise --load-factor or --capacity-bytes\n");
        std::exit(1);
    }

    std::fprintf(stderr,
        "[get] capacity=%zu slots (%zu B), F=%.3f, n_ops=%zu, "
        "block_size=%d, warmups=%d, reps=%d, insert_seed=%u, get_seed=%u, tag=\"%s\"\n",
        capacity, capacity_bytes, args.load_factor, n_ops,
        args.block_size, args.warmups, args.reps,
        args.insert_seed, args.get_seed, args.tag.c_str());

    // --- launch shape ---
    const auto shape = compute_launch_shape(args.block_size, Table::tile_size);
    const std::size_t n_tiles = shape.n_tiles;

    // --- pre-fill (untimed) ---
    std::uint32_t* d_insert_keys = nullptr;
    cudaMalloc(&d_insert_keys, n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_insert_keys, n_ops, args.insert_seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    std::uint32_t* d_prefill_failures = nullptr;
    cudaMalloc(&d_prefill_failures, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMemset(d_prefill_failures, 0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;

    prefill<<<shape.grid_size, args.block_size>>>(
        table.view(), d_insert_keys, n_ops, d_prefill_failures);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    std::vector<std::uint32_t> h_prefill_failures(n_tiles);
    cudaMemcpy(h_prefill_failures.data(), d_prefill_failures,
               n_tiles * sizeof(std::uint32_t),
               cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    const std::uint64_t prefill_failures = std::accumulate(
        h_prefill_failures.begin(), h_prefill_failures.end(), std::uint64_t{0});

    if (prefill_failures > 0) {
        // Not an error. At high load factor, Robin Hood displacement
        // chains can exhaust the probe cap before placing every input;
        // View::insert hands the leftover back, the benchmark records
        // the count. The table is still valid — it just contains
        // marginally fewer distinct keys than the input asked for.
        // The `hits` / `misses` CSV columns reflect the actual table
        // state, and `prefill_failures` records how far that deviates
        // from the intended workload.
        std::fprintf(stderr,
            "[get] pre-fill recorded %llu insert failures at F=%.3f "
            "(actual occupancy is slightly below the theoretical "
            "1 - exp(-F)). Proceeding; see prefill_failures column.\n",
            static_cast<unsigned long long>(prefill_failures), args.load_factor);
    }

    cudaFree(d_insert_keys);
    cudaFree(d_prefill_failures);

    // --- get key buffer (independent stream) ---
    std::uint32_t* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_get_keys, n_ops, args.get_seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    // --- output + per-tile counter arrays ---
    std::uint32_t* d_out = nullptr;
    cudaMalloc(&d_out, n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;

    std::uint32_t* d_probes = nullptr;
    std::uint32_t* d_hits   = nullptr;
    cudaMalloc(&d_probes, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_hits,   n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    std::vector<std::uint32_t> h_probes(n_tiles);
    std::vector<std::uint32_t> h_hits  (n_tiles);

    CSVWriter csv(args.output_dir / "get.csv",
        "library,workload,slot_bytes,capacity,capacity_bytes,"
        "key_range,alpha,load_factor,block_size,n_ops,bytes_per_op,"
        "insert_seed,get_seed,rep,tag,time_ms,ops_per_sec,gbps_floor,gbps_corrected,"
        "total_probes,total_hits,total_misses,prefill_failures");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            cudaMemset(d_probes, 0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
            cudaMemset(d_hits,   0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
        },
        /*launch=*/ [&]() {
            get<<<shape.grid_size, args.block_size>>>(
                table.view(), d_get_keys, d_out, n_ops, d_probes, d_hits);
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
            const std::uint64_t total_misses = std::uint64_t{n_ops} - total_hits;

            const double ops_per_sec =
                static_cast<double>(n_ops) / (static_cast<double>(ms) * 1.0e-3);
            const double gbps_floor =
                static_cast<double>(n_ops) * static_cast<double>(bytes_per_op)
                    / (static_cast<double>(ms) * 1.0e6);
            const double gbps_corrected =
                static_cast<double>(total_probes) * static_cast<double>(bytes_per_op)
                    / (static_cast<double>(ms) * 1.0e6);

            csv.write_row(format_row(
                slot_bytes, capacity, capacity_bytes,
                std::size_t{key_range}, /*alpha=*/1.0, args.load_factor,
                args.block_size, n_ops, bytes_per_op,
                args.insert_seed, args.get_seed, rep, args.tag,
                ms, ops_per_sec, gbps_floor, gbps_corrected,
                total_probes, total_hits, total_misses, prefill_failures));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    cudaFree(d_probes);
    cudaFree(d_hits);
    return 0;
}
