// Isolated tests for gpurhh::HashTable::View::insert.
//
// Each test:
//   - Optionally seeds a pre-state via cudaMemcpy (so the test runs against
//     a controlled starting layout, independent of insert correctness).
//   - Runs one or more inserts.
//   - Reads the table state back and either:
//       (A) compares to a hand-computed expected layout, or
//       (B) calls assert_robin_hood_invariant() to verify the structural
//           invariant holds across the whole table.
// Tests use both (A) and (B) — (A) for cases where the parallel insertion
// outcome is hand-predictable, (B) as a safety net everywhere else.

#include "isolated.cuh"

#include <algorithm>

namespace {

// Small table: capacity 64, bucket_size 16, num_buckets 4.
constexpr std::size_t kCapacity = 64;

// (A) single key inserted into empty table lands in its home bucket.
// Note: in bucket-level Robin Hood the slot-within-bucket position is
// decided by __ffs picking the lowest empty lane, not by the key's value.
// So for key 5 (home bucket 0) into an empty bucket, the key lands at
// slot 0 — the first empty lane — not at slot 5.
void test_insert_single_key_into_empty_table() {
    TestTable table(kCapacity);
    do_insert(table, {5}, {50});
    auto state = read_state(table);

    assert(state[0].key == 5);
    assert(state[0].value == 50);
    for (std::size_t i = 1; i < kCapacity; ++i) {
        assert(state[i].key == TestTable::empty_key);
    }
    assert_robin_hood_invariant(table, state);
}

// (A) keys with distinct home buckets all land at their home slots.
void test_insert_keys_into_distinct_home_buckets() {
    TestTable table(kCapacity);
    do_insert(table, {0, 16, 32, 48}, {100, 200, 300, 400});
    auto state = read_state(table);

    assert(state[0].key  ==  0 && state[0].value  == 100);
    assert(state[16].key == 16 && state[16].value == 200);
    assert(state[32].key == 32 && state[32].value == 300);
    assert(state[48].key == 48 && state[48].value == 400);
    assert_robin_hood_invariant(table, state);
}

// (B) fill bucket 0 with 16 home-0 keys. No displacement: every key has
// probe 0. Bucket 1..3 should be untouched. The exact (key → slot) mapping
// within bucket 0 is non-deterministic under concurrent insertion, so we
// check set membership rather than positions.
void test_insert_fills_home_bucket() {
    TestTable table(kCapacity);
    std::vector<std::uint32_t> keys(16), values(16);
    for (std::size_t i = 0; i < 16; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i);
        values[i] = static_cast<std::uint32_t>(i * 10);
    }
    do_insert(table, keys, values);
    auto state = read_state(table);

    std::vector<bool> seen(16, false);
    for (std::size_t s = 0; s < 16; ++s) {
        const auto& slot = state[s];
        assert(slot.key != TestTable::empty_key);
        assert(slot.key < 16);
        assert(slot.value == slot.key * 10);
        seen[slot.key] = true;
    }
    for (std::size_t i = 0; i < 16; ++i) assert(seen[i]);

    for (std::size_t i = 16; i < kCapacity; ++i) {
        assert(state[i].key == TestTable::empty_key);
    }
    assert_robin_hood_invariant(table, state);
}

// (A) deterministic displacement: start from a pre-state where bucket 0 is
// full of home-0 keys (0..15) AND bucket 1 is full of home-1 keys (16..31).
// Then insert key 64 (home bucket 0). It must walk bucket 0 with no
// progress (no match, no empty, equal probe), advance to bucket 1, find a
// richer resident (lane 0 / slot 16, key 16, probe 0 < ours 1), CAS it
// out, take key 16 as the new in-flight pair, advance to bucket 2 (key 16
// has home bucket 1, evicted probe 0 + 1 = 1 → bucket 1+1=2), and land in
// bucket 2's first empty slot (lane 0 → slot 32).
void test_insert_displaces_richer_resident() {
    TestTable table(kCapacity);
    auto state = empty_state(table);
    for (std::size_t i = 0; i < 32; ++i) {
        state[i] = {static_cast<std::uint32_t>(i),
                    static_cast<std::uint32_t>(i * 10)};
    }
    set_state(table, state);

    do_insert(table, {64}, {640});
    auto after = read_state(table);

    // Key 64 displaced key 16 at slot 16.
    assert(after[16].key   == 64);
    assert(after[16].value == 640);

    // Key 16 landed in slot 32 (bucket 2 lane 0), unchanged value.
    assert(after[32].key   == 16);
    assert(after[32].value == 160);

    // Keys 0..15 unchanged.
    for (std::size_t i = 0; i < 16; ++i) {
        assert(after[i].key   == i);
        assert(after[i].value == i * 10);
    }
    // Keys 17..31 unchanged (slot 16 was overwritten, the rest weren't).
    for (std::size_t i = 17; i < 32; ++i) {
        assert(after[i].key   == i);
        assert(after[i].value == i * 10);
    }
    // Bucket 2 slots 33..47 and all of bucket 3 should still be empty.
    for (std::size_t i = 33; i < kCapacity; ++i) {
        assert(after[i].key == TestTable::empty_key);
    }

    assert_robin_hood_invariant(table, after);
}

// (B) near-full table: 45 keys = load factor 0.7. Keys 1..45 have home
// buckets 0, 1, or 2 (no key in bucket 3); each home bucket has at most 16
// keys, so no displacement *out of* any bucket is forced — but several
// bucket-0 and bucket-2 keys may still land at non-home slots within their
// home bucket. Verify that all 45 keys are present and the invariant
// holds.
void test_insert_near_full() {
    TestTable table(kCapacity);
    constexpr std::size_t N = 45;
    std::vector<std::uint32_t> keys(N), values(N);
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);
        values[i] = static_cast<std::uint32_t>(i * 7);
    }
    do_insert(table, keys, values);
    auto state = read_state(table);

    std::vector<std::uint32_t> keys_seen;
    for (const auto& slot : state) {
        if (slot.key != TestTable::empty_key) keys_seen.push_back(slot.key);
    }
    std::sort(keys_seen.begin(), keys_seen.end());

    std::vector<std::uint32_t> expected(N);
    for (std::size_t i = 0; i < N; ++i) expected[i] = static_cast<std::uint32_t>(i + 1);
    assert(keys_seen == expected);

    assert_robin_hood_invariant(table, state);
}

} // namespace

int main() {
    test_insert_single_key_into_empty_table();
    test_insert_keys_into_distinct_home_buckets();
    test_insert_fills_home_bucket();
    test_insert_displaces_richer_resident();
    test_insert_near_full();
    std::printf("test_insert: all tests passed.\n");
    return 0;
}
