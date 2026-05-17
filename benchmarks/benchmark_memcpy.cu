// Memcpy baseline benchmark.
//
// Measures device-to-device memory bandwidth via two flavors:
//   - workload=memcpy_d2d    — cudaMemcpyAsync(..., DeviceToDevice)
//   - workload=memcpy_kernel — a trivial uint4-aligned copy kernel
// Both report effective GB/s assuming two bytes of DRAM traffic per byte
// of payload (one read plus one write). Used as a peak-bandwidth
// reference line for the insert / get benchmarks.

#include "benchmarks.cuh"

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

[[noreturn]] void die_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --output-dir DIR [options]\n"
        "  --bytes N         Buffer size in bytes (default 1<<30; rounded down to 16 B)\n"
        "  --warmups N       Untimed warmup runs per workload (default 2)\n"
        "  --reps N          Timed runs per workload (default 16)\n"
        "  --seed N          PRNG seed for the source buffer fill (default 42)\n"
        "  --tag STR         Free-form label written to every CSV row\n"
        "  --output-dir DIR  Required. memcpy.csv is appended to here.\n",
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
        if      (flag == "--bytes")      a.bytes      = std::stoull(get_val());
        else if (flag == "--warmups")    a.warmups    = std::stoi(get_val());
        else if (flag == "--reps")       a.reps       = std::stoi(get_val());
        else if (flag == "--seed")       a.seed       = static_cast<std::uint32_t>(std::stoul(get_val()));
        else if (flag == "--tag")        a.tag        = get_val();
        else if (flag == "--output-dir") a.output_dir = get_val();
        else { std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str()); die_usage(argv[0]); }
    }
    if (a.output_dir.empty()) die_usage(argv[0]);
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

std::string format_row(const std::string& workload, std::size_t bytes,
                       int rep, const std::string& tag,
                       float time_ms, double gbps)
{
    std::ostringstream s;
    s.precision(9);
    s << "cuda,"
      << workload << ","
      << bytes << ","
      << rep << ","
      << tag << ","
      << time_ms << ","
      << gbps;
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
        constexpr int fill_block = 256;
        const int fill_grid =
            static_cast<int>((n_u32 + fill_block - 1) / fill_block);
        fill_uniform_keys<<<fill_grid, fill_block>>>(
            d_src, n_u32, args.seed, std::uint32_t{1} << 31);
        cudaGetLastError() >> CUDA_CHECK;
    }
    cudaMemset(d_dst, 0, bytes) >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    CSVWriter csv(args.output_dir / "memcpy.csv",
        "library,workload,bytes,rep,tag,time_ms,gbps_effective");

    EventTimer timer;

    auto run = [&](const char* workload, auto&& launch) {
        for (int w = 0; w < args.warmups; ++w) launch();
        cudaDeviceSynchronize() >> CUDA_CHECK;
        for (int r = 0; r < args.reps; ++r) {
            timer.begin();
            launch();
            const float ms = timer.end_ms();
            // Two bytes of DRAM traffic per byte of payload.
            const double gbps = 2.0 * static_cast<double>(bytes)
                              / (static_cast<double>(ms) * 1.0e6);
            csv.write_row(format_row(workload, bytes, r, args.tag, ms, gbps));
        }
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
