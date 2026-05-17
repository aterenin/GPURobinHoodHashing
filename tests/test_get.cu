// Isolated tests for gpurhh::HashTable::View::get.
//
// Each test cudaMemcpys a hand-built state into the table via set_state
// and then verifies that get returns the expected result. This exercises
// View::get independently of View::insert.

#include "isolated.cuh"

namespace {

// Small table for all tests: capacity = 64, bucket_size = 16, num_buckets = 4.
constexpr std::size_t kCapacity = 64;

void test_hit_in_home_bucket() {
    // Place (5, 50) at slot 5 — its home slot. Lookup should find it
    // immediately in the first probe.
    TestTable table(kCapacity);
    auto state = empty_state(table);
    state[5] = {5, 50};
    set_state(table, state);

    auto r = run_get_one(table, 5);
    assert(r.has_value());
    assert(*r == 50);
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

    auto r = run_get_one(table, 64);
    assert(r.has_value());
    assert(*r == 6400);
}

void test_miss_in_empty_table() {
    TestTable table(kCapacity);
    set_state(table, empty_state(table));

    auto r = run_get_one(table, 42);
    assert(!r.has_value());
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

    auto r = run_get_one(table, 5);
    assert(!r.has_value());
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

    auto r = run_get_one(table, 64);
    assert(!r.has_value());
}

// Fill the entire table with multiples of 64 (all home bucket 0). After the
// fill, slot i holds key i*64 — which is at probe distance i / bucket_size
// from its home bucket 0. get should find each key regardless of how
// deeply displaced it is.
//
// Layout:
//   slots  0..15 → bucket 0, probe 0
//   slots 16..31 → bucket 1, probe 1
//   slots 32..47 → bucket 2, probe 2
//   slots 48..63 → bucket 3, probe 3
void test_get_at_various_probe_distances() {
    TestTable table(kCapacity);
    auto state = empty_state(table);
    for (std::size_t i = 0; i < kCapacity; ++i) {
        state[i].key   = static_cast<std::uint32_t>(i * 64);
        state[i].value = static_cast<std::uint32_t>(i);
    }
    set_state(table, state);

    // One probe at each distance. The interesting case is probe distance 3
    // — get must walk past three full buckets of residents with smaller-
    // or-equal probe distances before reaching the match.
    constexpr std::size_t B = TestTable::bucket_size;
    for (std::size_t bucket = 0; bucket < kCapacity / B; ++bucket) {
        const std::size_t slot_idx = bucket * B;
        const std::uint32_t key = static_cast<std::uint32_t>(slot_idx * 64);
        const auto r = run_get_one(table, key);
        assert(r.has_value());
        assert(*r == slot_idx);
    }
}

// Wrap-around test: place a key whose home bucket is the LAST bucket, but
// displaced by 1 (so it lives in bucket 0 — the probe sequence wraps).
// get must walk bucket (num_buckets - 1) first, see no match / no empty /
// no richer (the home bucket is full), advance to bucket 0 via the
// `(bucket_idx & bucket_mask)` wrap, and find the key there.
//
// With capacity 64 and num_buckets 4, bucket 3 is the last. Keys with home
// bucket 3 are those whose K & 63 falls in [48, 64), i.e., K ≡ 48..63
// (mod 64). We fill bucket 3 with keys 48..63 (each at its home slot,
// probe 0), then place key 48 + 64 = 112 at slot 0 (probe 1, wrapped).
void test_get_with_wrapped_probe() {
    TestTable table(kCapacity);
    auto state = empty_state(table);
    for (std::size_t i = 0; i < TestTable::bucket_size; ++i) {
        const std::uint32_t k = static_cast<std::uint32_t>(48 + i);
        state[48 + i] = {k, k * 10u};
    }
    // A second home-3 key, displaced one bucket and wrapped to bucket 0.
    const std::uint32_t wrapped_key = 48u + 64u;  // 112; home bucket 3
    state[0] = {wrapped_key, 1120u};
    set_state(table, state);

    auto r = run_get_one(table, wrapped_key);
    assert(r.has_value());
    assert(*r == 1120u);

    // Sanity: the original home-3 residents are still findable.
    auto r48 = run_get_one(table, 48);
    assert(r48.has_value() && *r48 == 480u);
    auto r63 = run_get_one(table, 63);
    assert(r63.has_value() && *r63 == 630u);
}

} // namespace

int main() {
    test_hit_in_home_bucket();
    test_hit_in_second_bucket();
    test_miss_in_empty_table();
    test_miss_via_empty_slot_in_home_bucket();
    test_miss_via_richer_resident();
    test_get_at_various_probe_distances();
    test_get_with_wrapped_probe();
    std::printf("test_get: all tests passed.\n");
    return 0;
}
