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

// (E2) Move construction transfers the device allocation to the new
// instance; the source becomes empty (capacity 0).
void test_move_construction() {
    Table a(1024);
    const std::size_t cap = a.capacity();

    Table b = std::move(a);
    assert(b.capacity() == cap);
    assert(a.capacity() == 0);
    // Both `a` and `b` destruct without double-freeing the buckets array.
}

// (E2) Move assignment frees the destination's existing allocation and
// takes over the source's.
void test_move_assignment() {
    Table a(1024);
    Table b(32);  // different capacity
    const std::size_t cap_a = a.capacity();

    b = std::move(a);
    assert(b.capacity() == cap_a);
    assert(a.capacity() == 0);
}

// (E4) Minimum-size table: capacity equals bucket_size (one bucket).
// Tests degenerate values for num_buckets and bucket_mask in the probe
// loops.
void test_minimum_size_table() {
    Table table(Table::bucket_size);
    assert(table.capacity() == Table::bucket_size);

    // Insert and retrieve a single key — exercises the full probe loop in
    // a one-bucket table.
    const std::uint32_t k = 7;
    const std::uint32_t v = 70;
    bulk_insert(table, &k, &v, 1);

    std::uint32_t got_value = 0;
    int got_found = 0;
    bulk_get(table, &k, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == v);
}

// (E4) Requesting a capacity smaller than bucket_size is rounded up to
// bucket_size (one full bucket is the minimum).
void test_capacity_below_bucket_size_clamps() {
    Table table(1);
    assert(table.capacity() == Table::bucket_size);
}

// (E5) Get on a moved-from (capacity-zero) table returns not-found
// without crashing — View::get's probe loop bound is 0 so the body never
// executes.
void test_get_on_moved_from_table_returns_not_found() {
    Table a(64);

    // Insert a real entry so the test would catch a bug where the get
    // accidentally accessed `a`'s old storage after the move.
    const std::uint32_t k = 7;
    const std::uint32_t v = 700;
    bulk_insert(a, &k, &v, 1);

    Table b = std::move(a);

    std::uint32_t got_value = 12345;  // poison
    int got_found = 1;                 // poison
    bulk_get(a, &k, &got_value, &got_found, 1);
    assert(got_found == 0);

    // Sanity: `b` still has the original entry.
    got_value = 0;
    got_found = 0;
    bulk_get(b, &k, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == v);
}

} // namespace

int main() {
    test_construct_with_capacity();
    test_destruct_releases_memory();
    test_move_construction();
    test_move_assignment();
    test_minimum_size_table();
    test_capacity_below_bucket_size_clamps();
    test_get_on_moved_from_table_returns_not_found();
    std::printf("test_construction: all tests passed.\n");
    return 0;
}
