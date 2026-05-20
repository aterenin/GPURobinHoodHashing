#pragma once

// gpurhh — printing helpers for HashTable.
//
// Pretty-prints HashTable internal state for debugging and diagnostics.
// Lives in a separate header so the core library is independent of
// <cstdio> and <vector>, and so that pulling print support in is opt-in.
//
// Relies on HashTable::data() to access the underlying bucket storage.
// All operations here copy bucket contents to host via cudaMemcpy and
// print to stdout; none of this is synchronized against concurrent
// device-side activity.

#include <gpurhh/hash_table.cuh>

#include <cstddef>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

namespace gpurhh {

// Pretty-print slots in [start, stop) of `table` to stdout, with bucket
// boundaries marked. Copies (stop - start) slots from device to host and
// is not synchronized against concurrent insert / get from other kernels.
template <class Table>
inline void print_slots(const Table& table, std::size_t start, std::size_t stop) {
    using Slot = typename Table::Slot;
    using Key  = typename Table::key_type;
    using V    = typename Table::value_type;

    const std::size_t capacity = table.capacity();
    if (start >= stop || stop > capacity) {
        std::printf("print_slots: invalid range [%zu, %zu) (capacity = %zu)\n",
                    start, stop, capacity);
        return;
    }

    // The Bucket array is laid out such that reinterpreting it as a flat
    // Slot[] is byte-equivalent — bucket_size * sizeof(Slot) == sizeof(Bucket)
    // with no internal padding.
    const std::size_t n = stop - start;
    std::vector<Slot> host_slots(n);
    cudaMemcpy(host_slots.data(),
               reinterpret_cast<const Slot*>(table.data()) + start,
               n * sizeof(Slot),
               cudaMemcpyDeviceToHost);

    std::printf("gpurhh::HashTable slots [%zu, %zu) "
                "(capacity = %zu, bucket_size = %d):\n",
                start, stop, capacity, Table::bucket_size);

    constexpr std::size_t B = static_cast<std::size_t>(Table::bucket_size);
    std::size_t current_bucket = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t slot_idx   = start + i;
        const std::size_t bucket_idx = slot_idx / B;
        if (bucket_idx != current_bucket) {
            std::printf("  bucket %zu:\n", bucket_idx);
            current_bucket = bucket_idx;
        }

        const Slot& slot = host_slots[i];
        std::printf("    slot %zu: ", slot_idx);
        if (slot.key == Table::empty_key) {
            std::printf("empty\n");
        } else if constexpr (sizeof(Key) == 4 && sizeof(V) == 4) {
            std::printf("key=0x%08x value=0x%08x\n",
                        static_cast<unsigned int>(slot.key),
                        static_cast<unsigned int>(slot.value));
        } else {
            std::printf("key=0x%016llx value=0x%016llx\n",
                        static_cast<unsigned long long>(slot.key),
                        static_cast<unsigned long long>(slot.value));
        }
    }
}

} // namespace gpurhh
