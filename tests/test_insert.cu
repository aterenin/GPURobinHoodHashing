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

// (A) probe-cap failure path. Fill the table with keys that all hash to
// home bucket 0 — they spread across buckets 0..3 at probe distances 0..3.
// With capacity 64 (num_buckets = 4) the effective probe bound is
// min(MaxProbeBuckets, num_buckets) = min(8, 4) = 4, so 4 * 16 = 64 keys
// fit exactly. A 65th home-bucket-0 key must fail.
void test_insert_past_probe_cap_fails() {
    TestTable table(kCapacity);

    std::vector<std::uint32_t> keys, values;
    for (std::uint32_t i = 0; i < 64; ++i) {
        keys.push_back(i * 64);  // all multiples of 64 → home bucket 0
        values.push_back(i);
    }
    const auto outcomes = do_insert_with_outcomes(table, keys, values);
    for (int o : outcomes) assert(o == 0);

    {
        const auto state = read_state(table);
        std::size_t occupied = 0;
        for (const auto& slot : state) {
            if (slot.key != TestTable::empty_key) {
                assert(slot.key % 64 == 0);
                ++occupied;
            }
        }
        assert(occupied == 64);
        assert_robin_hood_invariant(table, state);
    }

    // One more home-bucket-0 key. The table is exactly full; the probe loop
    // will walk all 4 buckets, find no match / no empty / no richer
    // resident (all residents have probe distance equal to ours), and
    // return false.
    const std::uint32_t over_cap_key = 64 * 64;  // 4096, also home bucket 0
    const auto failed = do_insert_with_outcomes(table, {over_cap_key}, {999});
    assert(failed.size() == 1);
    assert(failed[0] == 1);

    // The failed insert must not have touched the table.
    {
        const auto after = read_state(table);
        std::size_t occupied = 0;
        bool over_cap_present = false;
        for (const auto& slot : after) {
            if (slot.key != TestTable::empty_key) {
                assert(slot.key % 64 == 0);
                if (slot.key == over_cap_key) over_cap_present = true;
                ++occupied;
            }
        }
        assert(occupied == 64);
        assert(!over_cap_present);
        assert_robin_hood_invariant(table, after);
    }
}

// (B) concurrent inserts of the same key. With the default replace_op
// reduction, exactly one slot should hold the key after all inserts
// complete, and the surviving value should be one of the inputs.
void test_concurrent_same_key_inserts() {
    TestTable table(kCapacity);

    // 100 tiles all racing to insert key 7 with values 1..100.
    constexpr std::size_t N = 100;
    std::vector<std::uint32_t> keys(N, 7);
    std::vector<std::uint32_t> values(N);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = static_cast<std::uint32_t>(i + 1);
    }
    do_insert(table, keys, values);

    const auto state = read_state(table);

    // Uniqueness: exactly one slot holds key 7.
    int count = 0;
    std::uint32_t survivor_value = 0;
    for (const auto& slot : state) {
        if (slot.key == 7) {
            ++count;
            survivor_value = slot.value;
        }
    }
    assert(count == 1);

    // Validity: the surviving value is one of the inputs.
    assert(survivor_value >= 1 && survivor_value <= N);

    assert_robin_hood_invariant(table, state);
}

// (A) wrap-around: insert a key whose home bucket is the last bucket,
// when its home bucket is full. The probe sequence must wrap from bucket
// (num_buckets - 1) to bucket 0 via `(bucket_idx & bucket_mask)`. Verify
// the new key lands in bucket 0 (the wrapped destination) and the rest of
// the table is unchanged.
void test_insert_wraps_around() {
    TestTable table(kCapacity);
    auto state = empty_state(table);

    // Fill bucket 3 (the last bucket) with 16 home-3 keys.
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        const std::uint32_t k = static_cast<std::uint32_t>(48 + i);
        state[48 + i] = {k, k * 10u};
    }
    set_state(table, state);

    // Insert a second home-3 key. It must walk bucket 3 (full of equal-
    // probe residents → no displacement, advance) and then land at bucket
    // 0 lane 0 via the modular wrap.
    const std::uint32_t wrapped_key = 48u + 64u;  // 112; home bucket 3
    do_insert(table, {wrapped_key}, {1120u});
    const auto after = read_state(table);

    // The new key landed at slot 0.
    assert(after[0].key == wrapped_key);
    assert(after[0].value == 1120u);

    // Bucket 3 unchanged.
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        assert(after[48 + i].key   == 48u + i);
        assert(after[48 + i].value == (48u + i) * 10u);
    }

    // Slots 1..47 still empty.
    for (std::size_t i = 1; i < 48; ++i) {
        assert(after[i].key == TestTable::empty_key);
    }

    assert_robin_hood_invariant(table, after);
}

} // namespace

int main() {
    test_insert_single_key_into_empty_table();
    test_insert_keys_into_distinct_home_buckets();
    test_insert_fills_home_bucket();
    test_insert_displaces_richer_resident();
    test_insert_near_full();
    test_insert_past_probe_cap_fails();
    test_concurrent_same_key_inserts();
    test_insert_wraps_around();
    std::printf("test_insert: all tests passed.\n");
    return 0;
}
