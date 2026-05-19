// Get benchmark against cuCollections's `static_map`.
//
// Mirrors benchmarks/benchmark_get.cu so the two libraries share one
// get.csv. cuCollections's `find_async` returns the value (or the
// empty-value sentinel on miss) for each key in a flat output buffer,
// which closely matches the shape of our own get kernel's output.
// As with the insert baseline, cuco-only-empty columns (gbps_corrected,
// total_probes) are left blank for cuco rows.

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
#include <vector>

using Key   = std::uint32_t;
using Value = std::uint32_t;

constexpr Key   EMPTY_KEY   = ~Key{};
constexpr Value EMPTY_VALUE = ~Value{};

namespace {

struct Args {
    std::size_t   capacity_bytes = std::size_t{1} << 30;
    double        load_factor    = 1.0;
    int           block_size     = 256;  // unused by cuco's bulk API; kept for CSV uniformity
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
        "  Same shape as benchmarks/benchmark_get; see that for the full flag list.\n",
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
        else if (flag == "--insert-seed")    a.insert_seed    = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--get-seed")       a.get_seed       = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--tag")            a.tag            = get_val();
        else if (flag == "--output-dir")     a.output_dir     = get_val();
        else { std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str()); die_usage(argv[0]); }
    }
    if (a.output_dir.empty()) die_usage(argv[0]);
    if (a.load_factor <= 0.0) { std::fprintf(stderr, "--load-factor must be > 0\n"); std::exit(1); }
    return a;
}

// Match the CSV header of benchmarks/benchmark_get.cu. Columns cuco
// can't fill (gbps_corrected, total_probes, prefill_failures) are
// written as empty cells so one get.csv holds both libraries' rows.
std::string format_row(
    std::size_t slot_bytes, std::size_t capacity, std::size_t capacity_bytes,
    std::size_t key_range, double alpha, double load_factor,
    int block_size, std::size_t n_ops, std::size_t bytes_per_op,
    std::uint32_t insert_seed, std::uint32_t get_seed,
    int rep, const std::string& tag,
    float time_ms, double ops_per_sec, double gbps_floor,
    std::uint64_t total_hits, std::uint64_t total_misses)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuco,get,"
      << slot_bytes      << ","
      << capacity        << ","
      << capacity_bytes  << ","
      << key_range       << ","
      << alpha           << ","
      << load_factor     << ","
      << block_size      << ","
      << n_ops           << ","
      << bytes_per_op    << ","
      << insert_seed     << ","
      << get_seed        << ","
      << rep             << ","
      << tag             << ","
      << time_ms         << ","
      << ops_per_sec     << ","
      << gbps_floor      << ","
      << ","             // gbps_corrected
      << ","             // total_probes
      << total_hits      << ","
      << total_misses    << ","
      << "";             // prefill_failures (cuco can't report this directly)
    return s.str();
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
        "[cuco get] capacity=%zu slots (%zu B), F=%.3f, n_ops=%zu, "
        "warmups=%d, reps=%d, insert_seed=%u, get_seed=%u, tag=\"%s\"\n",
        capacity, capacity_bytes, args.load_factor, n_ops,
        args.warmups, args.reps, args.insert_seed, args.get_seed, args.tag.c_str());

    // --- pre-fill (untimed) ---
    Key* d_insert_keys = nullptr;
    cudaMalloc(&d_insert_keys, n_ops * sizeof(Key)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_insert_keys, n_ops, args.insert_seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    auto insert_pair_begin = thrust::make_transform_iterator(
        d_insert_keys,
        [] __device__ (Key k) {
            return cuco::pair<Key, Value>{k, gpurhh::detail::fmix32(k)};
        });

    cuco::static_map<Key, Value> map{
        capacity,
        cuco::empty_key<Key>{EMPTY_KEY},
        cuco::empty_value<Value>{EMPTY_VALUE}};

    map.insert_async(insert_pair_begin, insert_pair_begin + n_ops);
    cudaDeviceSynchronize() >> CUDA_CHECK;
    cudaFree(d_insert_keys);

    // --- get key buffer (independent stream) ---
    Key* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, n_ops * sizeof(Key)) >> CUDA_CHECK;
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_ops + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_get_keys, n_ops, args.get_seed, key_range);
        cudaGetLastError() >> CUDA_CHECK;
    }

    // Output buffer: one Value slot per get op. cuco writes the looked-up
    // value, or EMPTY_VALUE on miss.
    Value* d_out = nullptr;
    cudaMalloc(&d_out, n_ops * sizeof(Value)) >> CUDA_CHECK;

    CSVWriter csv(args.output_dir / "get.csv",
        "library,workload,slot_bytes,capacity,capacity_bytes,"
        "key_range,alpha,load_factor,block_size,n_ops,bytes_per_op,"
        "insert_seed,get_seed,rep,tag,time_ms,ops_per_sec,gbps_floor,gbps_corrected,"
        "total_probes,total_hits,total_misses,prefill_failures");

    EventTimer timer;

    // Hits/misses are counted on the host by reading back the output
    // buffer once after the first timed rep (the result is the same
    // every rep — table state doesn't change between reps).
    std::vector<Value> h_out(n_ops);
    std::uint64_t cached_hits   = 0;
    std::uint64_t cached_misses = 0;
    bool counted = false;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  []() {},   // table state unchanged between reps
        /*launch=*/ [&]() {
            map.find_async(d_get_keys, d_get_keys + n_ops, d_out);
        },
        /*after=*/  [&](int rep, float ms) {
            if (!counted) {
                cudaMemcpy(h_out.data(), d_out, n_ops * sizeof(Value),
                           cudaMemcpyDeviceToHost) >> CUDA_CHECK;
                for (std::size_t i = 0; i < n_ops; ++i) {
                    if (h_out[i] == EMPTY_VALUE) ++cached_misses;
                    else                          ++cached_hits;
                }
                counted = true;
            }
            const double ops_per_sec =
                static_cast<double>(n_ops) / (static_cast<double>(ms) * 1.0e-3);
            const double gbps_floor =
                static_cast<double>(n_ops) * static_cast<double>(bytes_per_op)
                    / (static_cast<double>(ms) * 1.0e6);
            csv.write_row(format_row(
                slot_bytes, capacity, capacity_bytes,
                std::size_t{key_range}, /*alpha=*/1.0, args.load_factor,
                args.block_size, n_ops, bytes_per_op,
                args.insert_seed, args.get_seed, rep, args.tag,
                ms, ops_per_sec, gbps_floor,
                cached_hits, cached_misses));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    return 0;
}
