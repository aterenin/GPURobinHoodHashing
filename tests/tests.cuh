#pragma once

// Small CUDA utilities shared across tests (and, eventually, examples).
//
// CUDA_CHECK provides a postfix error-check operator:
//
//     cudaMalloc(&p, n) >> CUDA_CHECK;
//
// The macro captures the call site so errors are reported with file:line and
// individual call sites do not need to carry a context string.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

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
