#pragma once

// Shared benchmark infrastructure: CUDA error checking, event-based
// timing, append-mode CSV recorder, deterministic uniform key generation.
// Independent of the test suite — benchmarks should not pull in
// <tests/...>.
//
// This header deliberately does *not* define GPURHH_BENCHMARK_COUNTERS.
// Counter-instrumented binaries (e.g. benchmark_insert_counters.cu)
// define it themselves before including this header; comparison
// binaries omit the define and get the counter-free View::insert /
// View::get signatures.

#include <gpurhh/hash_table.cuh>

#include <cuda_runtime.h>
#include <curand.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// ----------------------------------------------------------------------------
// CUDA_CHECK: postfix error-check operator. Identical in shape to the macro
// in <tests/tests.cuh>; copy-pasted rather than shared because the benchmark
// suite intentionally does not depend on test infrastructure.
//
//     cudaMalloc(&p, n) >> CUDA_CHECK;
// ----------------------------------------------------------------------------

struct CudaCheckLoc {
    const char* file;
    int         line;
};

inline void operator>>(cudaError_t e, CudaCheckLoc loc) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     loc.file, loc.line, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Same trick for cuRAND. cuRAND returns its own status code rather than
// the CUDA error code, so it gets its own overload; the macro is shared.
inline void operator>>(curandStatus_t e, CudaCheckLoc loc) {
    if (e != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuRAND error at %s:%d: code=%d\n",
                     loc.file, loc.line, static_cast<int>(e));
        std::exit(1);
    }
}

#define CUDA_CHECK CudaCheckLoc{__FILE__, __LINE__}

// CUDA event timer. begin() records the start event; end_ms() records
// the stop event, synchronizes on it, and returns elapsed milliseconds.
class EventTimer {
public:
    EventTimer() {
        cudaEventCreate(&start_) >> CUDA_CHECK;
        cudaEventCreate(&stop_)  >> CUDA_CHECK;
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&)            = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void begin() { cudaEventRecord(start_) >> CUDA_CHECK; }
    float end_ms() {
        cudaEventRecord(stop_) >> CUDA_CHECK;
        cudaEventSynchronize(stop_) >> CUDA_CHECK;
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_) >> CUDA_CHECK;
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Recorder: appends rows to a CSV file in the output directory, and
// also echoes every row (plus the header) to stdout for live visibility.
//
// If the target file is new or empty when constructed, the header line
// is written to the file first; if not, the file is appended to without
// a fresh header (so a sweep over many invocations builds one CSV).
class Recorder {
public:
    Recorder(const std::filesystem::path& path, std::string header)
        : header_(std::move(header))
    {
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        const bool needs_header = !std::filesystem::exists(path)
                                || std::filesystem::file_size(path) == 0;
        file_.open(path, std::ios::app);
        if (!file_) {
            std::fprintf(stderr, "Failed to open %s for writing\n",
                         path.string().c_str());
            std::exit(1);
        }
        if (needs_header) {
            file_ << header_ << "\n";
            file_.flush();
        }
        // Echo header once per invocation, regardless of file state, so
        // stdout is self-documenting even when appending to an existing
        // CSV from the sweep driver.
        std::cout << header_ << "\n";
    }

    void write_row(const std::string& row) {
        file_ << row << "\n";
        file_.flush();
        std::cout << row << "\n";
        std::cout.flush();
    }

private:
    std::string   header_;
    std::ofstream file_;
};

// RAII wrapper around a cuRAND Philox4_32_10 generator. Construct once
// from a seed, then pass to fill_uniform_keys one or more times — cuRAND
// holds the internal counter, so consecutive fills produce disjoint
// streams from a single seed. This preserves the original "one seed,
// two non-overlapping streams" design used by the get benchmarks.
//
// Philox is counter-based, deterministic, parallel, and fast (tens of
// GB/s on a modern GPU). Bias from x % range is at most range / 2^32
// per element, which is negligible for our key_range values.
class UniformKeyGenerator {
public:
    explicit UniformKeyGenerator(std::uint64_t seed) {
        curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_PHILOX4_32_10) >> CUDA_CHECK;
        curandSetPseudoRandomGeneratorSeed(gen_, seed)                >> CUDA_CHECK;
    }
    ~UniformKeyGenerator() { curandDestroyGenerator(gen_); }
    UniformKeyGenerator(const UniformKeyGenerator&)            = delete;
    UniformKeyGenerator& operator=(const UniformKeyGenerator&) = delete;

    curandGenerator_t handle() const { return gen_; }

private:
    curandGenerator_t gen_{};
};

// Map raw uint32 bits in-place to Uniform(0, range). For power-of-two
// range (the common case — capacity is forced to a power of two), the
// AND mask is many cycles cheaper than the divmod; the branch is uniform
// within a warp since `range` is a kernel parameter.
__global__ void map_to_range(
    std::uint32_t* x, std::size_t n, std::uint32_t range)
{
    const std::size_t tid = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (tid >= n) return;
    const std::uint32_t mask = range - 1;
    if ((range & mask) == 0) x[tid] = x[tid] & mask;
    else                     x[tid] = x[tid] % range;
}

// Fill `d_out[0..n)` with iid Uniform(0, range) draws using `gen`. All
// work happens on the GPU: cuRAND writes raw bits into d_out, then the
// map_to_range kernel folds them into [0, range). Both launches use the
// default stream, so they serialize naturally with the timed kernel that
// follows in run_benchmark_loop — and run_benchmark_loop's
// cudaDeviceSynchronize between warmups and timed reps guarantees this
// work has completed before the timing window opens.
inline void fill_uniform_keys(
    std::uint32_t* d_out, std::size_t n,
    std::uint32_t range, UniformKeyGenerator& gen)
{
    curandGenerate(gen.handle(), d_out, n) >> CUDA_CHECK;
    constexpr int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    map_to_range<<<grid, block>>>(d_out, n, range);
    cudaGetLastError() >> CUDA_CHECK;
}

// Launch-shape sizing: queries the current device for SM count, derives a
// grid size proportional to it (8 blocks per SM keeps SMs occupied without
// inflating the grid past the point of useful concurrency), and returns
// the per-block and total tile counts that the benchmark kernels need.
struct LaunchShape {
    int         grid_size;
    int         tiles_per_block;
    std::size_t n_tiles;
};

inline LaunchShape compute_launch_shape(int block_size, int tile_size) {
    int device  = 0;
    int num_sms = 0;
    cudaGetDevice(&device) >> CUDA_CHECK;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device)
        >> CUDA_CHECK;

    const int tiles_per_block = block_size / tile_size;
    const int grid_size       = num_sms * 8;
    const std::size_t n_tiles =
        static_cast<std::size_t>(grid_size)
      * static_cast<std::size_t>(tiles_per_block);
    return {grid_size, tiles_per_block, n_tiles};
}

// Boilerplate runner for the warmup-then-timed-reps pattern shared by
// every benchmark. The caller passes three callables:
//
//   - setup    — invoked before every iteration (both warmups and timed
//                reps). Use it to zero per-rep counters or clear the
//                table between insert reps. Defaults to a no-op if the
//                benchmark has nothing to reset.
//   - launch   — issues the kernel (or memcpy) being timed.
//   - after    — invoked after each timed rep with the rep index and
//                elapsed milliseconds. Use it to read back counters and
//                emit a CSV row.
template <class SetupFn, class LaunchFn, class AfterFn>
inline void run_benchmark_loop(
    int warmups, int reps, EventTimer& timer,
    SetupFn&& setup, LaunchFn&& launch, AfterFn&& after)
{
    for (int w = 0; w < warmups; ++w) {
        setup();
        launch();
    }
    cudaDeviceSynchronize() >> CUDA_CHECK;

    for (int r = 0; r < reps; ++r) {
        setup();
        timer.begin();
        launch();
        const float ms = timer.end_ms();
        after(r, ms);
    }
}
