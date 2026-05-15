// Retrieval tests for gpurhh::HashTable.

#include "common.cuh"

namespace {

void test_get_inserted_key() {
    Table table(1024);
    const std::uint32_t key   = 5;
    const std::uint32_t value = 99;
    bulk_insert(table, &key, &value, 1);

    std::uint32_t got_value = 0;
    int got_found = 0;
    bulk_get(table, &key, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == value);
}

void test_get_missing_key() {
    Table table(1024);
    const std::uint32_t key = 9999;
    std::uint32_t got_value = 12345;   // poison
    int got_found = 1;                 // poison
    bulk_get(table, &key, &got_value, &got_found, 1);
    assert(got_found == 0);
}

void test_get_after_many_inserts() {
    Table table(1024);
    constexpr std::size_t N = 50;
    std::uint32_t keys[N], values[N];
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);
        values[i] = static_cast<std::uint32_t>(2 * i + 1);
    }
    bulk_insert(table, keys, values, N);

    // Query a strided subset.
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
    test_get_inserted_key();
    test_get_missing_key();
    test_get_after_many_inserts();
    std::printf("test_get: all tests passed.\n");
    return 0;
}
