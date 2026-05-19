// Get benchmark against WarpCore's `SingleValueHashTable`.

#include "../../../benchmarks.cuh"

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
    std::size_t   capacity   = std::size_t{1} << 27;
    std::size_t   key_range  = std::size_t{1} << 27;
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
     .add("--key-range",  a.key_range,  "N in Uniform(0, N) for key generation")
     .add("--n-ops",      a.n_ops,      "Number of get attempts (also pre-fill count)")
     .add("--warmups",    a.warmups,    "Untimed warmup reps")
     .add("--reps",       a.reps,       "Timed reps")
     .add("--seed",       a.seed,       "cuRAND seed; one stream produces insert then get keys")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. get.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty()) p.print_usage();
    if (a.n_ops == 0)         { std::fprintf(stderr, "--n-ops must be > 0\n");     std::exit(1); }
    if (a.key_range == 0)     { std::fprintf(stderr, "--key-range must be > 0\n"); std::exit(1); }
    return a;
}

std::string format_row(
    std::size_t capacity, std::size_t key_range, std::size_t n_ops,
    std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms)
{
    std::ostringstream s;
    s.precision(9);
    s << "warpcore,get,"
      << tag          << ","
      << rep          << ","
      << seed         << ","
      << capacity     << ","
      << key_range    << ","
      << n_ops        << ","
      << ","          // block_size empty for warpcore
      << slot_bytes   << ","
      << bytes_per_op << ","
      << time_ms;
    return s.str();
}

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
    const Key key_range = static_cast<Key>(args.key_range);

    std::fprintf(stderr,
        "[warpcore get] capacity=%zu key_range=%zu n_ops=%zu "
        "warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        args.capacity, args.key_range, args.n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // Single cuRAND generator, sequential disjoint streams: first n_ops
    // draws → insert keys, next n_ops draws → get keys.
    UniformKeyGenerator gen(args.seed);

    // Pre-fill.
    Key*   d_insert_keys   = nullptr;
    Value* d_insert_values = nullptr;
    cudaMalloc(&d_insert_keys,   args.n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_insert_values, args.n_ops * sizeof(Value)) >> CUDA_CHECK;
    fill_uniform_keys(d_insert_keys, args.n_ops, key_range, gen);
    {
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((args.n_ops + fill_block - 1) / fill_block);
        derive_values<<<fill_grid, fill_block>>>(d_insert_keys, d_insert_values, args.n_ops);
        cudaGetLastError() >> CUDA_CHECK;
    }

    Table table(args.capacity);
    table.insert(d_insert_keys, d_insert_values, args.n_ops);
    cudaDeviceSynchronize() >> CUDA_CHECK;
    cudaFree(d_insert_keys);
    cudaFree(d_insert_values);

    // Get keys (continuation of the same generator; refilled per rep).
    Key* d_get_keys = nullptr;
    cudaMalloc(&d_get_keys, args.n_ops * sizeof(Key)) >> CUDA_CHECK;

    Value* d_out = nullptr;
    cudaMalloc(&d_out, args.n_ops * sizeof(Value)) >> CUDA_CHECK;

    Recorder rec(args.output_dir / "get.csv",
        "library,workload,tag,rep,seed,capacity,key_range,"
        "n_ops,block_size,slot_bytes,bytes_per_op,time_ms");

    EventTimer timer;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            fill_uniform_keys(d_get_keys, args.n_ops, key_range, gen);
        },
        /*launch=*/ [&]() {
            table.retrieve(d_get_keys, args.n_ops, d_out);
        },
        /*after=*/  [&](int rep, float ms) {
            rec.write_row(format_row(
                args.capacity, args.key_range, args.n_ops,
                slot_bytes, bytes_per_op,
                args.seed, rep, args.tag, ms));
        });

    cudaFree(d_get_keys);
    cudaFree(d_out);
    return 0;
}
