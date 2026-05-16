// Isolated tests for gpurhh::HashTable::View::get.
//
// Each test cudaMemcpys a hand-built state into the table via set_state
// and then verifies that get returns the expected result. This exercises
// View::get independently of View::insert.

#include "isolated.cuh"

namespace {

// Single-tile kernel: one block, one tile, looks up one key.
__global__ void get_one_kernel(TestView view, std::uint32_t key,
                               std::uint32_t* value_out, int* found_out) {
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<TestTable::tile_size>(block);
    const auto result = view.get(tile, key);
    if (tile.thread_rank() == 0) {
        *value_out = result.value_or(0);
        *found_out = result.has_value() ? 1 : 0;
    }
}

struct GetResult { bool found; std::uint32_t value; };

GetResult get_one(TestTable& table, std::uint32_t key) {
    std::uint32_t* d_value = nullptr;
    int* d_found = nullptr;
    cudaMalloc(&d_value, sizeof(std::uint32_t)) >> CUDA_CHECK;
    cudaMalloc(&d_found, sizeof(int))           >> CUDA_CHECK;

    get_one_kernel<<<1, TestTable::tile_size>>>(table.view(), key, d_value, d_found);
    cudaGetLastError()      >> CUDA_CHECK;
    cudaDeviceSynchronize() >> CUDA_CHECK;

    std::uint32_t value = 0;
    int found = 0;
    cudaMemcpy(&value, d_value, sizeof(std::uint32_t), cudaMemcpyDeviceToHost) >> CUDA_CHECK;
    cudaMemcpy(&found, d_found, sizeof(int),           cudaMemcpyDeviceToHost) >> CUDA_CHECK;

    cudaFree(d_value);
    cudaFree(d_found);
    return {found != 0, value};
}

// Small table for all tests: capacity = 64, bucket_size = 16, num_buckets = 4.
constexpr std::size_t kCapacity = 64;

void test_hit_in_home_bucket() {
    // Place (5, 50) at slot 5 — its home slot. Lookup should find it
    // immediately in the first probe.
    TestTable table(kCapacity);
    auto state = empty_state(table);
    state[5] = {5, 50};
    set_state(table, state);

    auto r = get_one(table, 5);
    assert(r.found);
    assert(r.value == 50);
}

void test_hit_in_second_bucket() {
    // Fill bucket 0 with keys 0..15 (all home bucket 0). Place key 64
    // (also home bucket 0, since 64 & 63 == 0) in slot 16 — i.e., bucket 1.
    // Get(64) should probe bucket 0 (no hit, no empty, no richer because
    // every resident has probe 0 == ours) and advance to bucket 1.
    TestTable table(kCapacity);
    auto state = empty_state(table);
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        state[i] = {static_cast<std::uint32_t>(i),
                    static_cast<std::uint32_t>(i * 10)};
    }
    state[TestTable::bucket_size] = {64, 6400};  // slot 16 = bucket 1, lane 0
    set_state(table, state);

    auto r = get_one(table, 64);
    assert(r.found);
    assert(r.value == 6400);
}

void test_miss_in_empty_table() {
    TestTable table(kCapacity);
    set_state(table, empty_state(table));

    auto r = get_one(table, 42);
    assert(!r.found);
}

void test_miss_via_empty_slot_in_home_bucket() {
    // Bucket 0 partly occupied (slots 0, 1 only); slots 2..15 empty.
    // Lookup of key 5 sees its home bucket and finds an empty slot in the
    // ballot — Robin Hood would have placed it at one of those empties,
    // so it isn't in the table.
    TestTable table(kCapacity);
    auto state = empty_state(table);
    state[0] = {0, 100};
    state[1] = {1, 101};
    set_state(table, state);

    auto r = get_one(table, 5);
    assert(!r.found);
}

void test_miss_via_richer_resident() {
    // Bucket 0 full of keys 0..15 (home bucket 0, probe 0).
    // Bucket 1 full of keys 16..31 (home bucket 1, probe 0).
    // Lookup of key 64 (home bucket 0): no match in bucket 0, no empty,
    // residents all have probe 0 == ours so no richer-termination yet.
    // Advance to bucket 1 (our probe now 1): residents have probe 0 < ours.
    // Richer-resident termination fires → return nullopt.
    TestTable table(kCapacity);
    auto state = empty_state(table);
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        state[i] = {static_cast<std::uint32_t>(i), 0};
    }
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        const std::uint32_t k = static_cast<std::uint32_t>(TestTable::bucket_size + i);
        state[TestTable::bucket_size + i] = {k, 0};
    }
    set_state(table, state);

    auto r = get_one(table, 64);
    assert(!r.found);
}

} // namespace

int main() {
    test_hit_in_home_bucket();
    test_hit_in_second_bucket();
    test_miss_in_empty_table();
    test_miss_via_empty_slot_in_home_bucket();
    test_miss_via_richer_resident();
    std::printf("test_get: all tests passed.\n");
    return 0;
}
