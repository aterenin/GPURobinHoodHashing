// Memcpy baseline benchmark.
//
// Measures device-to-device memory bandwidth via two flavors:
//   - workload=memcpy_d2d    — cudaMemcpyAsync(..., DeviceToDevice)
//   - workload=memcpy_kernel — a trivial uint4-aligned copy kernel
// Both report effective GB/s assuming two bytes of DRAM traffic per byte
// of payload (one read plus one write). Used as a peak-bandwidth
// reference line for the insert / get benchmarks.

#include "../benchmarks.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

struct Args {
    std::size_t bytes      = std::size_t{1} << 30;  // 1 GiB
    int         warmups    = 2;
    int         reps       = 16;
    std::uint32_t seed     = 42;
    std::string tag        = "";
    std::filesystem::path output_dir;
};

Args parse_args(int argc, char** argv) {
    Args a;
    ArgParser p(argv[0]);
    p.add("--bytes",      a.bytes,      "Buffer size in bytes (rounded down to 16 B)")
     .add("--warmups",    a.warmups,    "Untimed warmup runs per workload")
     .add("--reps",       a.reps,       "Timed runs per workload")
     .add("--seed",       a.seed,       "cuRAND seed for the source buffer fill")
     .add("--tag",        a.tag,        "Free-form label written to every CSV row")
     .add("--output-dir", a.output_dir, "Required. memcpy.csv is appended to here.");
    p.parse(argc, argv);

    if (a.output_dir.empty()) p.print_usage();
    a.bytes -= a.bytes % sizeof(uint4);  // round down for vectorized copy
    if (a.bytes == 0) {
        std::fprintf(stderr, "--bytes must be at least %zu\n", sizeof(uint4));
        std::exit(1);
    }
    return a;
}

__global__ void copy(uint4* dst, const uint4* src, std::size_t n) {
    const std::size_t tid = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

std::string format_row(const std::string& workload, const std::string& tag,
                       int rep, std::uint32_t seed, std::size_t bytes,
                       float time_ms)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuda,"
      << workload   << ","
      << tag        << ","
      << rep        << ","
      << seed       << ","
      << bytes      << ","
      << time_ms;
    return s.str();
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    const std::size_t bytes = args.bytes;
    const std::size_t n_u32 = bytes / sizeof(std::uint32_t);
    const std::size_t n_u4  = bytes / sizeof(uint4);

    std::fprintf(stderr,
        "[memcpy] bytes=%zu warmups=%d reps=%d seed=%u tag=\"%s\"\n",
        bytes, args.warmups, args.reps, args.seed, args.tag.c_str());

    std::uint32_t* d_src = nullptr;
    std::uint32_t* d_dst = nullptr;
    cudaMalloc(&d_src, bytes) >> CUDA_CHECK;
    cudaMalloc(&d_dst, bytes) >> CUDA_CHECK;

    // Fill the source buffer with non-trivial bytes so the kernel
    // doesn't accidentally measure a zero-page fast path.
    {
        UniformKeyGenerator gen(args.seed);
        fill_uniform_keys(d_src, n_u32, std::uint32_t{1} << 31, gen);
    }
    cudaMemset(d_dst, 0, bytes) >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    Recorder rec(args.output_dir / "memcpy.csv",
        "library,workload,tag,rep,seed,bytes,time_ms");

    EventTimer timer;

    auto run = [&](const char* workload, auto&& launch) {
        run_benchmark_loop(args.warmups, args.reps, timer,
            /*setup=*/  []() {},
            /*launch=*/ launch,
            /*after=*/  [&](int rep, float ms) {
                rec.write_row(format_row(
                    workload, args.tag, rep, args.seed, bytes, ms));
            });
    };

    run("memcpy_d2d", [&]() {
        cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice)
            >> CUDA_CHECK;
    });

    constexpr int block = 256;
    const int grid = static_cast<int>((n_u4 + block - 1) / block);
    run("memcpy_kernel", [&]() {
        copy<<<grid, block>>>(
            reinterpret_cast<uint4*>(d_dst),
            reinterpret_cast<const uint4*>(d_src),
            n_u4);
        cudaGetLastError() >> CUDA_CHECK;
    });

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
