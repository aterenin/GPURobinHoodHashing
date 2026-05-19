// Insert benchmark against WarpCore's `SingleValueHashTable`.
//
// Mirrors benchmarks/benchmark_insert.cu and the cuco baseline so all
// three libraries share insert.csv. WarpCore's bulk insert takes
// separate key + value buffers; we derive values via fmix32(key) into
// a side buffer to match the gpurhh / cuco convention.

#include "../../benchmarks.cuh"

#include <warpcore/single_value_hash_table.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

using Key   = std::uint32_t;
using Value = std::uint32_t;
using Table = warpcore::SingleValueHashTable<Key, Value>;

namespace {

struct Args {
    std::size_t   capacity_bytes = std::size_t{1} << 30;
    double        load_factor    = 1.0;
    int           block_size     = 256;  // unused; kept for CSV uniformity
    int           warmups        = 2;
    int           reps           = 16;
    std::uint32_t seed           = 42;
    std::string   tag            = "";
    std::filesystem::path output_dir;
};

[[noreturn]] void die_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --output-dir DIR [options]\n"
        "  Same shape as benchmarks/benchmark_insert; see that for the\n"
        "  full flag list. WarpCore's bulk insert picks its own launch\n"
        "  configuration; block_size is recorded for CSV uniformity but\n"
        "  not used.\n",
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
    if (a.output_dir.empty()) die_usage(argv[0]);
    if (a.load_factor <= 0.0) { std::fprintf(stderr, "--load-factor must be > 0\n"); std::exit(1); }
    return a;
}

// Match the CSV header of benchmarks/benchmark_insert.cu. Columns
// warpcore can't fill (gbps_corrected, total_probes, total_failures)
// are written empty.
std::string format_row(
    std::size_t slot_bytes, std::size_t capacity, std::size_t capacity_bytes,
    std::size_t key_range, double alpha, double load_factor,
    int block_size, std::size_t n_ops, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, double ops_per_sec, double gbps_floor)
{
    std::ostringstream s;
    s.precision(9);
    s << "warpcore,insert,"
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
      << ","   // gbps_corrected
      << ","   // total_probes
      << "";   // total_failures
    return s.str();
}

// Derive values from keys: value = fmix32(key). Same convention used by
// the cuco baseline and the gpurhh insert benchmark, so the per-key
// bytes-of-state-flowing match across libraries.
__global__ void derive_values(
    const Key* __restrict__ keys, Value* __restrict__ values, std::size_t n)
{
    const std::size_t tid = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (tid >= n) return;
    values[tid] = gpurhh::detail::fmix32(keys[tid]);
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    constexpr std::size_t slot_bytes   = sizeof(Key) + sizeof(Value);
    constexpr std::size_t bytes_per_op = 128;
    const std::size_t capacity       = args.capacity_bytes / slot_bytes;
    const std::size_t capacity_bytes = capacity * slot_bytes;
    const Key key_range = static_cast<Key>(capacity);
    const std::size_t n_ops =
        static_cast<std::size_t>(args.load_factor * static_cast<double>(capacity));

    if (n_ops == 0) {
        std::fprintf(stderr, "Computed n_ops = 0; raise --load-factor or --capacity-bytes\n");
        std::exit(1);
    }

    std::fprintf(stderr,
        "[warpcore insert] capacity=%zu slots (%zu B), F=%.3f, n_ops=%zu, "
        "warmups=%d, reps=%d, seed=%u, tag=\"%s\"\n",
        capacity, capacity_bytes, args.load_factor, n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // --- input buffers ---
    Key*   d_keys   = nullptr;
    Value* d_values = nullptr;
    cudaMalloc(&d_keys,   n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values, n_ops * sizeof(Value)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_keys, n_ops, args.seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
        derive_values<<<fill_grid, fill_block>>>(d_keys, d_values, n_ops);
        cudaGetLastError() >> CUDA_CHECK;
    }

    // Construct the table. WarpCore auto-inits at construction.
    Table table(capacity);

    CSVWriter csv(args.output_dir / "insert.csv",
        "library,workload,slot_bytes,capacity,capacity_bytes,"
        "key_range,alpha,load_factor,block_size,n_ops,bytes_per_op,"
        "seed,rep,tag,time_ms,ops_per_sec,gbps_floor,gbps_corrected,"
        "total_probes,total_failures");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() { table.init(); },  // (re)initialize all slots to empty
        /*launch=*/ [&]() { table.insert(d_keys, d_values, n_ops); },
        /*after=*/  [&](int rep, float ms) {
            const double ops_per_sec =
                static_cast<double>(n_ops) / (static_cast<double>(ms) * 1.0e-3);
            const double gbps_floor =
                static_cast<double>(n_ops) * static_cast<double>(bytes_per_op)
                    / (static_cast<double>(ms) * 1.0e6);
            csv.write_row(format_row(
                slot_bytes, capacity, capacity_bytes,
                std::size_t{key_range}, /*alpha=*/1.0, args.load_factor,
                args.block_size, n_ops, bytes_per_op,
                args.seed, rep, args.tag,
                ms, ops_per_sec, gbps_floor));
        });

    cudaFree(d_keys);
    cudaFree(d_values);
    return 0;
}
