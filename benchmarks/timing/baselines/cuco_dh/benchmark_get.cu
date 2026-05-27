// Get benchmark against cuCollections's `static_map`, with the probing
// scheme overridden to double hashing (CG=8). See
// benchmarks/timing/baselines/cuco_dh/benchmark_insert.cu for the
// rationale; this is the matching get benchmark. CSV `library` column
// reads "cuco_dh".

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

// static_map with double hashing (CG=8) instead of the default linear
// probing (CG=4). Same hash function and same {empty_key, empty_value}
// sentinels as the default-cuco baseline; only the probing scheme
// differs.
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

Args parse_args(int argc, char** argv) {
    Args a;
    ArgParser p(argv[0]);
    p.add("--capacity",   a.capacity,   "Table size in slots")
     .add("--n-ops",      a.n_ops,      "Number of get attempts (also pre-fill count)")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed; one stream produces insert then get keys")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. get.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty()) p.print_usage();
    if (a.n_ops == 0)         { std::fprintf(stderr, "--n-ops must be > 0\n"); std::exit(1); }
    return a;
}

std::string format_row(
    std::size_t capacity, std::size_t n_ops,
    std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuco_dh,get,"
      << tag          << ","
      << rep          << ","
      << seed         << ","
      << capacity     << ","
      << n_ops        << ","
      << ","          // block_size empty for cuco_dh
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

    std::fprintf(stderr,
        "[cuco_dh get] capacity=%zu n_ops=%zu "
        "warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        args.capacity, args.n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // Single cuRAND generator: first n_ops draws → insert keys, next
    // n_ops draws → get keys. No shared draws between the streams.
    UniformKeyGenerator gen(args.seed);

    // Pre-fill: build a pair iterator on the insert key stream and
    // bulk-insert. cuco's static_map has no reduction operator; duplicate
    // keys are no-ops.
    Key* d_insert_keys = nullptr;
    cudaMalloc(&d_insert_keys, args.n_ops * sizeof(Key)) >> CUDA_CHECK;
    fill_uniform_keys(d_insert_keys, args.n_ops, gen);

    auto insert_pair_begin = thrust::make_transform_iterator(
        d_insert_keys,
        [] __device__ (Key k) -> cuco::pair<Key, Value> {
            return cuco::pair<Key, Value>{k, gpurhh::detail::fmix32(k)};
        });

    Map map{args.capacity,
            cuco::empty_key<Key>{EMPTY_KEY},
            cuco::empty_value<Value>{EMPTY_VALUE}};

    map.insert_async(insert_pair_begin, insert_pair_begin + args.n_ops);
    cudaDeviceSynchronize() >> CUDA_CHECK;
    cudaFree(d_insert_keys);

    // Get keys (continuation of the same generator — disjoint draws,
    // refilled in setup per rep).
    Key* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, args.n_ops * sizeof(Key)) >> CUDA_CHECK;

    Value* d_out = nullptr;
    cudaMalloc(&d_out, args.n_ops * sizeof(Value)) >> CUDA_CHECK;

    Recorder rec(args.output_dir / "get.csv",
        "library,workload,tag,rep,seed,capacity,"
        "n_ops,block_size,slot_bytes,bytes_per_op,time_ms");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            fill_uniform_keys(d_get_keys, args.n_ops, gen);
        },
        /*launch=*/ [&]() {
            map.find_async(d_get_keys, d_get_keys + args.n_ops, d_out);
        },
        /*after=*/  [&](int rep, float ms) {
            rec.write_row(format_row(
                args.capacity, args.n_ops,
                slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    return 0;
}
