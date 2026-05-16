// Concurrency stress tests for gpurhh::HashTable. Each test launches a
// kernel with many tiles running concurrently against the same table, in
// a configuration designed to exercise a specific code path under
// contention.

#include "isolated.cuh"

#include <set>

namespace {

// --- (C1) adversarial parallel displacement ------------------------------
//
// Pre-fill 8 buckets, each with 16 keys whose home matches that bucket
// (so every resident has probe distance 0). Then insert 16 new keys all
// with home bucket 0 in parallel: each insert walks bucket 0 (no
// displacement — equal probe), advances to bucket 1, displaces a richer
// (probe-0) home-1 resident, adopts it as in-flight, advances to
// bucket 2, displaces a home-2 resident, ... — a 7-step chain per insert,
// happening simultaneously across 16 tiles. Heavy contention on the
// displacement CAS retry that no other test stresses.
void test_adversarial_parallel_displacement() {
    constexpr std::size_t capacity = 256;  // 16 buckets
    TestTable table(capacity);

    constexpr std::size_t B = TestTable::bucket_size;
    auto state = empty_state(table);
    for (std::size_t b = 0; b < 8; ++b) {
        for (std::size_t lane = 0; lane < B; ++lane) {
            const std::size_t slot_idx = b * B + lane;
            state[slot_idx] = {static_cast<std::uint32_t>(slot_idx),
                               static_cast<std::uint32_t>(slot_idx * 10u)};
        }
    }
    set_state(table, state);

    constexpr std::size_t N = 16;
    std::vector<std::uint32_t> keys(N), values(N);
    for (std::size_t i = 0; i < N; ++i) {
        // Each new key has K & 255 < 16, i.e., home bucket 0, and is
        // distinct from the pre-state's 0..15 (which also have home 0).
        keys[i]   = static_cast<std::uint32_t>(256 + i);
        values[i] = static_cast<std::uint32_t>(100000 + i);
    }
    const auto outcomes = do_insert_with_outcomes(table, keys, values);
    for (int o : outcomes) assert(o == 0);

    const auto after = read_state(table);

    // 128 + 16 = 144 occupied slots after the inserts.
    std::size_t occupied = 0;
    for (const auto& slot : after) {
        if (slot.key != TestTable::empty_key) ++occupied;
    }
    assert(occupied == 144);

    // Every original key is still in the table (none lost during the
    // cascading displacements).
    std::vector<std::uint32_t> all_keys;
    all_keys.reserve(144);
    for (std::size_t i = 0; i < 128; ++i) {
        all_keys.push_back(static_cast<std::uint32_t>(i));
    }
    for (auto k : keys) all_keys.push_back(k);

    std::vector<std::uint32_t> got_values;
    std::vector<int> got_found;
    do_get(table, all_keys, got_values, got_found);
    for (std::size_t i = 0; i < all_keys.size(); ++i) {
        assert(got_found[i] == 1);
    }
    // Original keys retain their original values; new keys retain theirs.
    for (std::size_t i = 0; i < 128; ++i) {
        assert(got_values[i] == i * 10u);
    }
    for (std::size_t i = 0; i < N; ++i) {
        assert(got_values[128 + i] == values[i]);
    }

    assert_robin_hood_invariant(table, after);
}

// --- (C2) concurrent gets on a populated table ---------------------------
//
// Insert N distinct keys, then run N parallel gets in a single kernel.
// Read-only operations have no data races in principle; this verifies the
// implementation doesn't have any hidden writes that would break under
// parallel reads.
void test_concurrent_gets() {
    constexpr std::size_t capacity = 4096;
    constexpr std::size_t N = 1000;
    TestTable table(capacity);

    std::vector<std::uint32_t> keys(N), values(N);
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i + 1);
        values[i] = static_cast<std::uint32_t>((i + 1) * 7u);
    }
    do_insert(table, keys, values);

    std::vector<std::uint32_t> got_values;
    std::vector<int> got_found;
    do_get(table, keys, got_values, got_found);

    for (std::size_t i = 0; i < N; ++i) {
        assert(got_found[i] == 1);
        assert(got_values[i] == values[i]);
    }
}

// --- (C3) mixed new + duplicate inserts in parallel ----------------------
//
// 100 unique keys, each inserted twice (with two different values) in the
// same parallel batch — 200 inserts total. After: exactly 100 occupied
// slots (uniqueness invariant), and each unique key holds one of its two
// inserted values (replace_op semantics: some writer wins, but the
// surviving value is always one of the inputs).
void test_mixed_new_and_duplicate_inserts() {
    constexpr std::size_t capacity = 4096;
    TestTable table(capacity);

    constexpr std::size_t U = 100;
    std::vector<std::uint32_t> keys, values;
    keys.reserve(2 * U);
    values.reserve(2 * U);
    for (std::size_t i = 0; i < U; ++i) {
        const std::uint32_t k = static_cast<std::uint32_t>(i + 1);
        keys.push_back(k); values.push_back(k * 10u);
        keys.push_back(k); values.push_back(k * 100u);
    }
    do_insert(table, keys, values);

    // For each unique key, get returns one of the two inserted values.
    std::vector<std::uint32_t> unique_keys(U);
    for (std::size_t i = 0; i < U; ++i) unique_keys[i] = static_cast<std::uint32_t>(i + 1);

    std::vector<std::uint32_t> got_values;
    std::vector<int> got_found;
    do_get(table, unique_keys, got_values, got_found);
    for (std::size_t i = 0; i < U; ++i) {
        assert(got_found[i] == 1);
        const std::uint32_t k = unique_keys[i];
        assert(got_values[i] == k * 10u || got_values[i] == k * 100u);
    }

    // Exactly U occupied slots — no duplicate-key entries in the table.
    const auto state = read_state(table);
    std::size_t occupied = 0;
    for (const auto& slot : state) {
        if (slot.key != TestTable::empty_key) ++occupied;
    }
    assert(occupied == U);

    assert_robin_hood_invariant(table, state);
}

// --- (C4) parallel inserts at probe-cap saturation -----------------------
//
// Insert N home-bucket-0 keys in parallel, where N exceeds the table's
// effective capacity for home-0 keys (= num_buckets * bucket_size). The
// excess inserts must fail; the surviving inserts must form a consistent
// table.
void test_parallel_probe_cap_saturation() {
    constexpr std::size_t capacity = 64;  // 4 buckets, probe_bound = 4
    TestTable table(capacity);

    // 80 keys, all multiples of 64 → all home bucket 0. The table can fit
    // 64 (probe distances 0..3); the other 16 must hit the probe-cap
    // failure path.
    constexpr std::size_t N = 80;
    std::vector<std::uint32_t> keys(N), values(N);
    for (std::size_t i = 0; i < N; ++i) {
        keys[i]   = static_cast<std::uint32_t>(i * 64);
        values[i] = static_cast<std::uint32_t>(i);
    }
    const auto outcomes = do_insert_with_outcomes(table, keys, values);

    std::size_t succeeded = 0, failed = 0;
    for (int o : outcomes) {
        if (o == 0) ++succeeded;
        else        ++failed;
    }
    assert(succeeded == 64);
    assert(failed    == 16);

    // Final state: 64 occupied slots, all multiples of 64, all from the
    // input set. Which 64 won the race is non-deterministic.
    const auto state = read_state(table);
    std::set<std::uint32_t> surviving;
    for (const auto& slot : state) {
        if (slot.key != TestTable::empty_key) {
            assert(slot.key % 64 == 0);
            assert(slot.key / 64 < N);
            surviving.insert(slot.key);
        }
    }
    assert(surviving.size() == 64);
    assert_robin_hood_invariant(table, state);

    // Each surviving key is findable via get.
    std::vector<std::uint32_t> survivors_vec(surviving.begin(), surviving.end());
    std::vector<std::uint32_t> got_values;
    std::vector<int> got_found;
    do_get(table, survivors_vec, got_values, got_found);
    for (std::size_t i = 0; i < survivors_vec.size(); ++i) {
        assert(got_found[i] == 1);
    }
}

} // namespace

int main() {
    test_adversarial_parallel_displacement();
    test_concurrent_gets();
    test_mixed_new_and_duplicate_inserts();
    test_parallel_probe_cap_saturation();
    std::printf("test_concurrency: all tests passed.\n");
    return 0;
}
