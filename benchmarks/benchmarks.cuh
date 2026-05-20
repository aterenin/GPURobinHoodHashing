#pragma once

// Shared benchmark infrastructure: CUDA error checking, event-based
// timing, append-mode CSV recorder, deterministic uniform key generation,
// a tiny declarative command-line parser. Independent of the test suite
// — benchmarks should not pull in <tests/...>.
//
// This header deliberately does *not* define GPURHH_BENCHMARK_COUNTERS.
// The memory-benchmark binaries (benchmarks/memory/*) define it
// themselves before including this header; the timing binaries
// (benchmarks/timing/*) omit it and get the counter-free View::insert /
// View::get signatures.

#include <gpurhh/hash_table.cuh>

#include <cuda_runtime.h>
#include <curand.h>

#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

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
// GB/s on a modern GPU).
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

// Fill `d_out[0..n)` with iid uniform uint32 draws — i.e. Uniform(0, 2^32),
// the full type range. cuRAND writes directly into the uint32 buffer, so
// no clamping kernel is needed.
//
// The benchmark grid intentionally fixes the key distribution at the full
// uint32 range: combined with our 2^27-slot table, this gives α ≈ 32,
// keys are effectively unique, and we are squarely in the collision-
// resolution regime where table designs differ. If you instead want to
// stress the duplicate-collapse / reduction path (α ≈ 1, ~63% peak
// occupancy), see the note in notes/design.md — restoring that mode
// takes ~30 lines (a small clamp kernel plus a --key-range flag).
inline void fill_uniform_keys(
    std::uint32_t* d_out, std::size_t n, UniformKeyGenerator& gen)
{
    curandGenerate(gen.handle(), d_out, n) >> CUDA_CHECK;
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

// Count slots in `[slots, slots + n)` whose key is not `empty_key`. Used
// by the drop-counting plumbing in the insert benchmarks: after the
// timed kernel, scan the table and compare to the pre-counted unique-
// input count to derive how many keys the library failed to land.
//
// Synchronous (blocks on `stream` via thrust's reduction). Pulls in
// `<thrust/count.h>`; lives in the benchmark header so the gpurhh public
// API stays dependency-free. Templated on Slot so it works against any
// table whose slot type has a `key` field — i.e. gpurhh's own slots and,
// if needed later, slot-like reinterpretations of baseline storage.
template <class Slot, class Key>
inline std::size_t count_occupied_slots(
    const Slot* slots, std::size_t n, Key empty_key,
    cudaStream_t stream = 0)
{
    const Key ek = empty_key;
    return thrust::count_if(
        thrust::cuda::par.on(stream),
        slots, slots + n,
        [ek] __device__ (const Slot& s) { return s.key != ek; });
}

// ----------------------------------------------------------------------------
// ArgParser: tiny declarative command-line parser shared by every binary.
//
// Each binary builds an Args struct with default values, then writes:
//
//     ArgParser p(argv[0]);
//     p.add("--capacity", a.capacity, "Table size in slots");
//     p.add("--n-ops",    a.n_ops,    "Number of operations");
//     ...
//     p.parse(argc, argv);
//     // post-parse validation here, calling p.print_usage() on failure
//
// Supported target types: std::size_t, int, std::uint32_t, std::string,
// std::filesystem::path. Unknown / missing-value / --help all print a
// generated usage block and exit. There is no support for positional
// args, short flags, or boolean toggles — the benchmark CLIs don't need
// any of that.
// ----------------------------------------------------------------------------

namespace detail {

template <class T>
inline void assign_arg(T& dst, std::string_view s) {
    const std::string str(s);
    if constexpr (std::is_same_v<T, std::size_t>) {
        dst = std::stoull(str);
    } else if constexpr (std::is_same_v<T, int>) {
        dst = std::stoi(str);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        dst = static_cast<std::uint32_t>(std::stoul(str));
    } else if constexpr (std::is_same_v<T, std::string>) {
        dst = str;
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        dst = std::filesystem::path(str);
    } else {
        // Force a compile error mentioning T when an unsupported type sneaks in.
        static_assert(sizeof(T) == 0, "ArgParser: unsupported target type");
    }
}

} // namespace detail

class ArgParser {
public:
    explicit ArgParser(const char* prog) : prog_(prog) {}

    template <class T>
    ArgParser& add(std::string flag, T& target, std::string help = "") {
        handlers_.push_back(Handler{
            std::move(flag),
            std::move(help),
            [&target](std::string_view s) { detail::assign_arg(target, s); }
        });
        return *this;
    }

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help" || flag == "-h") print_usage(0);
            const Handler* h = find(flag);
            if (!h) {
                std::fprintf(stderr, "Unknown flag: %s\n", argv[i]);
                print_usage(1);
            }
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", argv[i]);
                print_usage(1);
            }
            h->setter(argv[++i]);
        }
    }

    [[noreturn]] void print_usage(int code = 1) const {
        std::fprintf(stderr, "usage: %s [options]\n", prog_);
        for (const auto& h : handlers_) {
            std::fprintf(stderr, "  %-16s %s\n",
                         h.flag.c_str(), h.help.c_str());
        }
        std::exit(code);
    }

private:
    struct Handler {
        std::string flag;
        std::string help;
        std::function<void(std::string_view)> setter;
    };

    const Handler* find(std::string_view flag) const {
        for (const auto& h : handlers_) if (h.flag == flag) return &h;
        return nullptr;
    }

    const char*          prog_;
    std::vector<Handler> handlers_;
};

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
