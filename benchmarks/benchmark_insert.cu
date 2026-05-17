// Insert benchmark for gpurhh.
//
// Measures insert throughput and effective DRAM bandwidth as a function
// of input load factor F = n_ops / capacity. Workload:
//   - Key type uint32_t, value type uint32_t, sum-based reduction.
//   - Keys ~ Uniform(0, key_range) with key_range = capacity (α = 1).
//   - Single ~1 GiB table, DRAM-resident.
//
// Per timed rep:
//   1. Empty the table via cudaMemset(0xFF) (untimed).
//   2. Zero the per-tile probe and failure counters (untimed).
//   3. Time one kernel that processes n_ops insert ops in a grid-stride loop.
//   4. Sum per-tile counters on host; write a CSV row.

// Opt in to the gpurhh benchmark counter API and the internal data()
// accessor (needed to memset the bucket array between reps without
// destroying / reconstructing the table). Both macros must be defined
// before any gpurhh header inclusion.
#define GPURHH_BENCHMARK_COUNTERS    1
#define GPURHH_ENABLE_INTERNAL_ACCESS 1

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
    double        load_factor    = 1.0;                    // F = n_ops / capacity
    int           block_size     = 256;
    int           warmups        = 2;
    int           reps           = 16;
    std::uint32_t seed           = 42;
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
        "  --seed N            PRNG seed for key generation (default 42)\n"
        "  --tag STR           Free-form label written to every CSV row\n"
        "  --output-dir DIR    Required. insert.csv is appended to here.\n",
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
        else if (flag == "--seed")           a.seed           = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--tag")            a.tag            = get_val();
        else if (flag == "--output-dir")     a.output_dir     = get_val();
        else { std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str()); die_usage(argv[0]); }
    }
    if (a.output_dir.empty())                        die_usage(argv[0]);
    if (a.load_factor <= 0.0)                        { std::fprintf(stderr, "--load-factor must be > 0\n"); std::exit(1); }
    if (a.block_size % Table::tile_size != 0)        { std::fprintf(stderr, "--block-size must be a multiple of tile_size (%d)\n", Table::tile_size); std::exit(1); }
    if (a.block_size <= 0 || a.block_size > 1024)    { std::fprintf(stderr, "--block-size out of range\n"); std::exit(1); }
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
        // Derive value from key by hashing again. With sum_op, value content
        // affects the sum's correctness but not throughput; we just need
        // non-trivial bytes per op.
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
    std::size_t slot_bytes, std::size_t capacity, std::size_t capacity_bytes,
    std::size_t key_range, double alpha, double load_factor,
    int block_size, std::size_t n_ops, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, double ops_per_sec, double gbps_floor, double gbps_corrected,
    std::uint64_t total_probes, std::uint64_t total_failures)
{
    std::ostringstream s;
    s.precision(9);
    s << "gpurhh,insert,"
      << slot_bytes      << ","
      << capacity        << ","
      << capacity_bytes  << ","
      << key_range       << ","
      << alpha           << ","
      << load_factor     << ","
      << block_size      << ","
      << n_ops           << ","
      << bytes_per_op    << ","
      << seed            << ","
      << rep             << ","
      << tag             << ","
      << time_ms         << ","
      << ops_per_sec     << ","
      << gbps_floor      << ","
      << gbps_corrected  << ","
      << total_probes    << ","
      << total_failures;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // --- table sizing ---
    constexpr std::size_t slot_bytes   = sizeof(Table::Slot);    // 8
    constexpr std::size_t bytes_per_op = sizeof(Table::Bucket);  // 128
    const std::size_t min_slots = args.capacity_bytes / slot_bytes;
    Table table(min_slots);
    const std::size_t capacity       = table.capacity();
    const std::size_t capacity_bytes = capacity * slot_bytes;
    // α = 1: key range matches capacity. Cast to uint32 — capacity must
    // fit, since key type is uint32_t.
    const std::uint32_t key_range = static_cast<std::uint32_t>(capacity);
    const std::size_t   n_ops     =
        static_cast<std::size_t>(args.load_factor * static_cast<double>(capacity));

    if (n_ops == 0) {
        std::fprintf(stderr, "Computed n_ops = 0; raise --load-factor or --capacity-bytes\n");
        std::exit(1);
    }

    std::fprintf(stderr,
        "[insert] capacity=%zu slots (%zu B), F=%.3f, n_ops=%zu, "
        "block_size=%d, warmups=%d, reps=%d, seed=%u, tag=\"%s\"\n",
        capacity, capacity_bytes, args.load_factor, n_ops,
        args.block_size, args.warmups, args.reps, args.seed, args.tag.c_str());

    // --- input buffer ---
    std::uint32_t* d_keys = nullptr;
    cudaMalloc(&d_keys, n_ops * sizeof(std::uint32_t)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_keys, n_ops, args.seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    // --- launch shape ---
    const int tiles_per_block = args.block_size / Table::tile_size;
    int device  = 0;
    int num_sms = 0;
    cudaGetDevice(&device) >> CUDA_CHECK;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device) >> CUDA_CHECK;
    // 8 blocks per SM keeps every SM well-occupied without inflating the
    // grid past the point of useful concurrency.
    const int grid_size = num_sms * 8;
    const std::size_t n_tiles =
        static_cast<std::size_t>(grid_size)
      * static_cast<std::size_t>(tiles_per_block);

    // --- per-tile counter arrays ---
    std::uint32_t* d_probes   = nullptr;
    std::uint32_t* d_failures = nullptr;
    cudaMalloc(&d_probes,   n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_failures, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    std::vector<std::uint32_t> h_probes  (n_tiles);
    std::vector<std::uint32_t> h_failures(n_tiles);

    CSVWriter csv(args.output_dir / "insert.csv",
        "library,workload,slot_bytes,capacity,capacity_bytes,"
        "key_range,alpha,load_factor,block_size,n_ops,bytes_per_op,"
        "seed,rep,tag,time_ms,ops_per_sec,gbps_floor,gbps_corrected,"
        "total_probes,total_failures");

    EventTimer timer;

    auto reset_table = [&]() {
        const std::size_t num_buckets = capacity / Table::bucket_size;
        cudaMemset(table.data(), 0xFF,
                   num_buckets * sizeof(Table::Bucket)) >> CUDA_CHECK;
    };
    auto reset_counters = [&]() {
        cudaMemset(d_probes,   0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
        cudaMemset(d_failures, 0, n_tiles * sizeof(std::uint32_t)) >> CUDA_CHECK;
    };
    auto launch = [&]() {
        insert<<<grid_size, args.block_size>>>(
            table.view(), d_keys, n_ops, d_probes, d_failures);
        cudaGetLastError() >> CUDA_CHECK;
    };

    // --- warmups ---
    for (int w = 0; w < args.warmups; ++w) {
        reset_table();
        reset_counters();
        launch();
    }
    cudaDeviceSynchronize() >> CUDA_CHECK;

    // --- timed reps ---
    for (int r = 0; r < args.reps; ++r) {
        reset_table();
        reset_counters();
        timer.begin();
        launch();
        const float ms = timer.end_ms();

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
            args.seed, r, args.tag,
            ms, ops_per_sec, gbps_floor, gbps_corrected,
            total_probes, total_failures));
    }

    cudaFree(d_keys);
    cudaFree(d_probes);
    cudaFree(d_failures);
    return 0;
}
