// Insert benchmark against cuCollections's `static_map`.
//
// Mirrors benchmarks/benchmark_insert.cu so the two libraries can share
// the same CSV file (`output/.../insert.csv`) and overlay cleanly on a
// plot. cuCollections's API differs in detail — `insert_async` takes a
// pair iterator rather than separate key/value buffers, and there's no
// per-probe instrumentation hook to feed `gbps_corrected` or per-tile
// counters — so a few CSV columns are written empty for cuco rows.

#include "../../benchmarks.cuh"

#include <cuco/static_map.cuh>

#include <thrust/iterator/transform_iterator.h>

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

// Sentinels for cuco::static_map. Match gpurhh's empty-key convention
// (all-bits-set) for a fair comparison.
constexpr Key   EMPTY_KEY   = ~Key{};
constexpr Value EMPTY_VALUE = ~Value{};

namespace {

struct Args {
    std::size_t   capacity_bytes = std::size_t{1} << 30;
    double        load_factor    = 1.0;
    int           block_size     = 256;  // unused by cuco's bulk API; kept for CSV uniformity
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
        "  full flag list. cuco's bulk insert_async takes the key\n"
        "  buffer, derives values via fmix32, and uses the entire pair\n"
        "  iterator at once — block_size is recorded for CSV uniformity\n"
        "  but does not affect cuco's internal launch.\n",
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
    if (a.output_dir.empty())            die_usage(argv[0]);
    if (a.load_factor <= 0.0)            { std::fprintf(stderr, "--load-factor must be > 0\n"); std::exit(1); }
    if (a.load_factor > 0.95)            { std::fprintf(stderr, "[cuco] --load-factor > 0.95 may overflow static_map capacity; capping is the caller's job\n"); }
    return a;
}

// Match the CSV header of benchmarks/benchmark_insert.cu exactly.
// Columns cuco can't fill (gbps_corrected, total_probes, total_failures)
// are written as empty cells so a single insert.csv holds rows from
// both libraries side by side.
std::string format_row(
    std::size_t slot_bytes, std::size_t capacity, std::size_t capacity_bytes,
    std::size_t key_range, double alpha, double load_factor,
    int block_size, std::size_t n_ops, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, double ops_per_sec, double gbps_floor)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuco,insert,"
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
      << ","   // gbps_corrected (cuco has no probe instrumentation)
      << ","   // total_probes
      << "";   // total_failures
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // Slot bytes match cuco::pair<uint32, uint32>'s footprint. bytes_per_op
    // stays at 128 (one cache line) for direct comparison with gpurhh's
    // floor estimate.
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
        "[cuco insert] capacity=%zu slots (%zu B), F=%.3f, n_ops=%zu, "
        "warmups=%d, reps=%d, seed=%u, tag=\"%s\"\n",
        capacity, capacity_bytes, args.load_factor, n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // --- input buffer ---
    Key* d_keys = nullptr;
    cudaMalloc(&d_keys, n_ops * sizeof(Key)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_keys, n_ops, args.seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    // Transform iterator: produces cuco::pair<Key, Value> from each key
    // on the fly, with value = fmix32(key). Matches the gpurhh
    // benchmark's "derive value from key" choice; no separate values
    // buffer needed.
    auto pair_begin = thrust::make_transform_iterator(
        d_keys,
        [] __device__ (Key k) {
            return cuco::pair<Key, Value>{k, gpurhh::detail::fmix32(k)};
        });

    // Construct the map. capacity is in slots (cuco rounds up internally
    // for whatever per-bucket layout it picks).
    cuco::static_map<Key, Value> map{
        capacity,
        cuco::empty_key<Key>{EMPTY_KEY},
        cuco::empty_value<Value>{EMPTY_VALUE}};

    CSVWriter csv(args.output_dir / "insert.csv",
        "library,workload,slot_bytes,capacity,capacity_bytes,"
        "key_range,alpha,load_factor,block_size,n_ops,bytes_per_op,"
        "seed,rep,tag,time_ms,ops_per_sec,gbps_floor,gbps_corrected,"
        "total_probes,total_failures");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            map.clear_async();
        },
        /*launch=*/ [&]() {
            map.insert_async(pair_begin, pair_begin + n_ops);
        },
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
    return 0;
}
