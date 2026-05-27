// Insert benchmark against cuCollections's `static_map`, with the
// probing scheme overridden to double hashing (CG=8). The sibling
// `cuco/` binary tests static_map with its default (linear probing,
// CG=4); this one isolates "linear probing vs double hashing" within
// the cuCollections codebase. CSV `library` column reads "cuco_dh".

#include "../../../benchmarks.cuh"

#include <cuco/static_map.cuh>

#include <thrust/iterator/counting_iterator.h>
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

constexpr Key   EMPTY_KEY   = ~Key{};
constexpr Value EMPTY_VALUE = ~Value{};

// static_map specialized with double hashing instead of the default
// `cuco::linear_probing<4, default_hash_function<Key>>`. CG size 8 to
// match warpcore's default; same hash function.
using Map = cuco::static_map<
    Key, Value,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<Key>,
    cuco::double_hashing<8, cuco::default_hash_function<Key>>>;

namespace {

struct Args {
    std::size_t   capacity   = std::size_t{1} << 27;
    std::size_t   n_ops      = std::size_t{1} << 27;
    int           warmups    = 2;
    int           reps       = 16;
    std::uint32_t seed       = 42;
    std::string   tag        = "";
    std::filesystem::path output_dir;
};

// Same shape as benchmarks/timing/benchmark_insert. cuco picks its own
// internal launch shape; --block-size is not a flag here.
Args parse_args(int argc, char** argv) {
    Args a;
    ArgParser p(argv[0]);
    p.add("--capacity",   a.capacity,   "Table size in slots")
     .add("--n-ops",      a.n_ops,      "Number of insert attempts")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed for key generation")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. insert.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty()) p.print_usage();
    if (a.n_ops == 0)         { std::fprintf(stderr, "--n-ops must be > 0\n"); std::exit(1); }
    return a;
}

// Matches benchmark_insert.cu's schema. block_size cell is empty for
// cuco (no exposed launch knob).
std::string format_row(
    std::size_t capacity, std::size_t n_ops,
    std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, std::size_t n_unique, std::size_t drops)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuco_dh,insert,"
      << tag          << ","
      << rep          << ","
      << seed         << ","
      << capacity     << ","
      << n_ops        << ","
      << ","          // block_size empty for cuco_dh
      << slot_bytes   << ","
      << bytes_per_op << ","
      << time_ms      << ","
      << n_unique     << ","
      << drops;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    constexpr std::size_t slot_bytes   = sizeof(Key) + sizeof(Value);
    constexpr std::size_t bytes_per_op = 128;

    std::fprintf(stderr,
        "[cuco_dh insert] capacity=%zu n_ops=%zu "
        "warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        args.capacity, args.n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // d_keys:   refreshed per rep in setup.
    // d_values: zeroed once below. Apples-to-apples with gpurhh's timing
    //           build — the cuco_dh insert kernel pays the d_values input
    //           read via the transform iterator below, even though under
    //           replace_op the value content is irrelevant.
    // d_sorted: scratch copy of d_keys, sorted in setup and uniqued to
    //           count distinct inputs for drop accounting (n_ops_unique).
    Key*   d_keys   = nullptr;
    Value* d_values = nullptr;
    Key*   d_sorted = nullptr;
    cudaMalloc(&d_keys,   args.n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values, args.n_ops * sizeof(Value)) >> CUDA_CHECK;
    cudaMalloc(&d_sorted, args.n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMemset(d_values, 0, args.n_ops * sizeof(Value)) >> CUDA_CHECK;
    UniformKeyGenerator gen(args.seed);

    // Transform iterator over a counting iterator: at index i, reads
    // d_keys[i] and d_values[i] and packs them into a cuco::pair. The
    // captured pointers are device-side; the lambda runs on GPU
    // (--extended-lambda).
    auto pair_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<std::size_t>(0),
        [d_keys, d_values] __device__ (std::size_t i) -> cuco::pair<Key, Value> {
            return cuco::pair<Key, Value>{d_keys[i], d_values[i]};
        });

    Map map{args.capacity,
            cuco::empty_key<Key>{EMPTY_KEY},
            cuco::empty_value<Value>{EMPTY_VALUE}};

    Recorder rec(args.output_dir / "insert.csv",
        "library,workload,tag,rep,seed,capacity,n_ops,block_size,"
        "slot_bytes,bytes_per_op,time_ms,n_unique,drops");

    EventTimer timer;

    std::size_t n_unique = 0;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            map.clear_async();
            n_unique = fill_uniform_keys_below_capacity(
                d_keys, d_sorted, args.n_ops, args.capacity, gen,
                "cuco_dh insert");
        },
        /*launch=*/ [&]() { map.insert_async(pair_begin, pair_begin + args.n_ops); },
        /*after=*/  [&](int rep, float ms) {
            // cuco's size() syncs the stream internally.
            const std::size_t occupied = map.size();
            const std::size_t drops =
                n_unique > occupied ? n_unique - occupied : 0;
            rec.write_row(format_row(
                args.capacity, args.n_ops,
                slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms, n_unique, drops));
        });

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_sorted);
    return 0;
}
