// Insertion tests for gpurhh::HashTable.

#include "common.cuh"

namespace {

void test_insert_single_key() {
    Table table(1024);
    const std::uint32_t keys[]   = {42};
    const std::uint32_t values[] = {100};
    bulk_insert(table, keys, values, 1);

    std::uint32_t got_value = 0;
    int got_found = 0;
    bulk_get(table, keys, &got_value, &got_found, 1);
    assert(got_found == 1);
    assert(got_value == 100);
}

void test_insert_many_distinct_keys() {
    Table table(1024);
    constexpr std::size_t N = 100;
    std::uint32_t keys[N], values[N];
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);     // avoid 0 to keep keys clear of any defaults
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

void test_insert_duplicate_key_updates_value() {
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

} // namespace

int main() {
    test_insert_single_key();
    test_insert_many_distinct_keys();
    test_insert_duplicate_key_updates_value();
    std::printf("test_insert: all tests passed.\n");
    return 0;
}
