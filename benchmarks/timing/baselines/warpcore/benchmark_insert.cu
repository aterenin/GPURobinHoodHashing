// Insert benchmark against WarpCore's `SingleValueHashTable`.
//
// Apples-to-apples with benchmarks/benchmark_insert.cu's CSV schema.

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

std::string format_row(
    std::size_t capacity, std::size_t n_ops,
    std::size_t slot_bytes, std::size_t bytes_per_op,
    std::uint32_t seed, int rep, const std::string& tag,
    float time_ms, std::size_t n_unique, std::size_t drops)
{
    std::ostringstream s;
    s.precision(9);
    s << "warpcore,insert,"
      << tag          << ","
      << rep          << ","
      << seed         << ","
      << capacity     << ","
      << n_ops        << ","
      << ","          // block_size empty for warpcore
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
        "[warpcore insert] capacity=%zu n_ops=%zu "
        "warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        args.capacity, args.n_ops,
        args.warmups, args.reps, args.seed, args.tag.c_str());

    // d_keys:   refreshed per rep in setup.
    // d_values: zeroed once below. Apples-to-apples with gpurhh / cuco
    //           — the warpcore insert kernel pays the d_values input
    //           read, and under replace_op semantics the value content
    //           is irrelevant.
    // d_sorted: scratch copy of d_keys for sort + unique drop counting.
    Key*   d_keys   = nullptr;
    Value* d_values = nullptr;
    Key*   d_sorted = nullptr;
    cudaMalloc(&d_keys,   args.n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMalloc(&d_values, args.n_ops * sizeof(Value)) >> CUDA_CHECK;
    cudaMalloc(&d_sorted, args.n_ops * sizeof(Key))   >> CUDA_CHECK;
    cudaMemset(d_values, 0, args.n_ops * sizeof(Value)) >> CUDA_CHECK;
    UniformKeyGenerator gen(args.seed);

    Table table(args.capacity);

    Recorder rec(args.output_dir / "insert.csv",
        "library,workload,tag,rep,seed,capacity,n_ops,block_size,"
        "slot_bytes,bytes_per_op,time_ms,n_unique,drops");

    EventTimer timer;

    std::size_t n_unique = 0;

    run_benchmark_loop(args.warmups, args.reps, timer,
        /*setup=*/  [&]() {
            table.init();
            n_unique = fill_uniform_keys_below_capacity(
                d_keys, d_sorted, args.n_ops, args.capacity, gen,
                "warpcore insert");
        },
        /*launch=*/ [&]() { table.insert(d_keys, d_values, args.n_ops); },
        /*after=*/  [&](int rep, float ms) {
            // warpcore's size() is synchronous w.r.t. the stream argument.
            const std::size_t occupied = table.size();
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
