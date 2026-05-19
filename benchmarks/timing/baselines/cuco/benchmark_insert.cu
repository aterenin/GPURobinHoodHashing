// Insert benchmark against cuCollections's `static_map`.
//
// Apples-to-apples with benchmarks/benchmark_insert.cu: same CSV schema,
// same raw configuration columns (library distinguishes rows). Derived
// metrics live in the analysis script, not here.

#include "../../../benchmarks.cuh"

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

constexpr Key   EMPTY_KEY   = ~Key{};
constexpr Value EMPTY_VALUE = ~Value{};

namespace {

struct Args {
    std::size_t   capacity   = std::size_t{1} << 27;
    std::size_t   key_range  = std::size_t{1} << 27;
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
     .add("--key-range",  a.key_range,  "N in Uniform(0, N) for key generation")
     .add("--n-ops",      a.n_ops,      "Number of insert attempts")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed for key generation")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. insert.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty()) p.print_usage();
    if (a.n_ops == 0)         { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.key_range == 0)     { std::fprintf(stderr, "--key-range must be > 0\n"); std::exit(1); }
    return a;
}

// Matches benchmark_insert.cu's schema. block_size cell is empty for
// cuco (no exposed launch knob).
std::string format_row(
    std::size_t capacity, std::size_t key_range, std::size_t n_ops,
    std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuco,insert,"
      << tag          << ","
      << rep          << ","
      << seed         << ","
      << capacity     << ","
      << key_range    << ","
      << n_ops        << ","
      << ","          // block_size empty for cuco
      << slot_bytes   << ","
      << bytes_per_op << ","
      << time_ms;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    constexpr std::size_t slot_bytes   = sizeof(Key) + sizeof(Value);
    constexpr std::size_t bytes_per_op = 128;
    const Key key_range = static_cast<Key>(args.key_range);

    std::fprintf(stderr,
        "[cuco insert] capacity=%zu key_range=%zu n_ops=%zu "
        "warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        args.capacity, args.key_range, args.n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    Key* d_keys = nullptr;
    cudaMalloc(&d_keys, args.n_ops * sizeof(Key)) >> CUDA_CHECK;
    UniformKeyGenerator gen(args.seed);

    // Transform iterator: builds cuco::pair<Key, Value> on the fly from
    // each key, with value = fmix32(key). Matches the gpurhh benchmark's
    // value-derivation choice.
    auto pair_begin = thrust::make_transform_iterator(
        d_keys,
        [] __device__ (Key k) -> cuco::pair<Key, Value> {
            return cuco::pair<Key, Value>{k, gpurhh::detail::fmix32(k)};
        });

    cuco::static_map<Key, Value> map{
        args.capacity,
        cuco::empty_key<Key>{EMPTY_KEY},
        cuco::empty_value<Value>{EMPTY_VALUE}};

    Recorder rec(args.output_dir / "insert.csv",
        "library,workload,tag,rep,seed,capacity,key_range,n_ops,block_size,"
        "slot_bytes,bytes_per_op,time_ms");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            map.clear_async();
            fill_uniform_keys(d_keys, args.n_ops, key_range, gen);
        },
        /*launch=*/ [&]() { map.insert_async(pair_begin, pair_begin + args.n_ops); },
        /*after=*/  [&](int rep, float ms) {
            rec.write_row(format_row(
                args.capacity, args.key_range, args.n_ops,
                slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms));
        });

    cudaFree(d_keys);
    return 0;
}
