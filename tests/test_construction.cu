// Construction / allocation tests for gpurhh::HashTable.

#include "kernels.cuh"

namespace {

void test_construct_with_capacity() {
    Table table(1024);
    assert(table.capacity() >= 1024);
    // Capacity should be rounded up to a power of two.
    assert((table.capacity() & (table.capacity() - 1)) == 0);
}

void test_destruct_releases_memory() {
    // Smoke test: repeatedly construct and destruct to surface any CUDA
    // errors from the (de)allocation paths.
    for (int i = 0; i < 4; ++i) {
        Table table(1024);
        (void)table;
    }
    cudaDeviceSynchronize() >> CUDA_CHECK;
}

} // namespace

int main() {
    test_construct_with_capacity();
    test_destruct_releases_memory();
    std::printf("test_construction: all tests passed.\n");
    return 0;
}
