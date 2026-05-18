#pragma once

// Shared benchmark infrastructure: CUDA error checking, event-based
// timing, append-mode CSV writer, deterministic uniform key generation.
// Independent of the test suite — benchmarks should not pull in
// <tests/...>.

// Opt every benchmark TU into the gpurhh probe-counter API. Defined
// before any gpurhh include so that View::insert / View::get expose
// their counter parameters. See notes/design.md for the contract.
#define GPURHH_BENCHMARK_COUNTERS 1

#include <gpurhh/hash_table.cuh>

#include <cuda_runtime.h>

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

// CSV writer: appends rows to a file in the output directory, and also
// echoes every row (plus the header) to stdout for live visibility.
//
// If the target file is new or empty when constructed, the header line
// is written to the file first; if not, the file is appended to without
// a fresh header (so a sweep over many invocations builds one CSV).
class CSVWriter {
public:
    CSVWriter(const std::filesystem::path& path, std::string header)
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

// Fill out[0..n) with fmix32(seed + i) mod range. For `range` a power
// of two (the case here, since key range = α × capacity with capacity
// a power of two), the modulus yields a uniform distribution.
//
// Not marked inline: __global__ functions cannot be inline, but each
// benchmark binary is a single translation unit that includes this
// header exactly once, so there is no multiple-definition concern.
__global__ void fill_uniform_keys(
    std::uint32_t* out, std::size_t n,
    std::uint32_t seed, std::uint32_t range)
{
    const std::size_t tid = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (tid >= n) return;
    out[tid] = gpurhh::detail::fmix32(static_cast<std::uint32_t>(tid) + seed) % range;
}
