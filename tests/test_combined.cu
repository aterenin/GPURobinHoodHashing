// End-to-end tests combining gpurhh::HashTable::View::insert and View::get.
//
// These tests verify the public API behaves correctly when insert and get
// are used together, without inspecting internal state. Tests for insert
// and get individually live in test_insert.cu and test_get.cu, which seed
// the table directly via cudaMemcpy and inspect its state.

#include "kernels.cuh"

namespace {

void test_insert_and_get_single_key() {
    Table table(1024);
    const std::uint32_t key   = 42;
    const std::uint32_t value = 100;
    bulk_insert(table, &key, &value, 1);

    std::uint32_t got_value = 0;
    int got_found = 0;
    bulk_get(table, &key, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == value);
}

void test_insert_and_get_many_distinct_keys() {
    Table table(1024);
    constexpr std::size_t N = 100;
    std::uint32_t keys[N], values[N];
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);
        values[i] = static_cast<std::uint32_t>(i + 1000);
    }
    bulk_insert(table, keys, values, N);

    std::uint32_t got_value[N] = {};
    int got_found[N] = {};
    bulk_get(table, keys, got_value, got_found, N);
    for (std::size_t i = 0; i < N; ++i) {
        assert(got_found[i] == 1);
        assert(got_value[i] == values[i]);
    }
}

void test_duplicate_key_updates_value() {
    Table table(1024);
    const std::uint32_t key = 7;
    const std::uint32_t v1  = 111;
    const std::uint32_t v2  = 222;
    bulk_insert(table, &key, &v1, 1);
    bulk_insert(table, &key, &v2, 1);

    std::uint32_t got_value = 0;
    int got_found = 0;
    bulk_get(table, &key, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == v2);
}

void test_get_missing_key() {
    Table table(1024);
    const std::uint32_t key = 9999;
    std::uint32_t got_value = 12345;   // poison
    int got_found = 1;                 // poison
    bulk_get(table, &key, &got_value, &got_found, 1);
    assert(got_found == 0);
}

void test_get_strided_subset() {
    Table table(1024);
    constexpr std::size_t N = 50;
    std::uint32_t keys[N], values[N];
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);
        values[i] = static_cast<std::uint32_t>(2 * i + 1);
    }
    bulk_insert(table, keys, values, N);

    constexpr std::size_t M = 10;
    std::uint32_t query[M], got_value[M];
    int got_found[M];
    for (std::size_t i = 0; i < M; ++i) query[i] = keys[5 * i];
    bulk_get(table, query, got_value, got_found, M);
    for (std::size_t i = 0; i < M; ++i) {
        assert(got_found[i] == 1);
        assert(got_value[i] == values[5 * i]);
    }
}

} // namespace

int main() {
    test_insert_and_get_single_key();
    test_insert_and_get_many_distinct_keys();
    test_duplicate_key_updates_value();
    test_get_missing_key();
    test_get_strided_subset();
    std::printf("test_combined: all tests passed.\n");
    return 0;
}
