# Testing

The test suite lives in `tests/`, with each `test_*.cu` compiled into its own executable.
Run the full suite with `make test`, which builds each binary under `build/tests/` and executes them in sequence.

- **`test_construction.cu`** — construct / destruct, move semantics, edge sizes (capacity == bucket_size, capacity below bucket_size), moved-from state behavior, and `clear()` returning the bucket array to the empty-byte pattern.
- **`test_combined.cu`** — end-to-end insert + get round-trips through the public API.
- **`test_get.cu`** — `View::get` in isolation. Pre-states are hand-built with `IdentityHash` so the probe sequence and home buckets are deterministic; the test then verifies `get`'s match / empty / richer-resident / wrap-around behavior.
- **`test_insert.cu`** — `View::insert` in isolation. Some tests use hand-computed expected layouts; all run a Robin Hood invariant scanner over the final state as a safety net. Covers the probe-cap failure path and concurrent same-key inserts.
- **`test_concurrency.cu`** — heavy parallel contention: adversarial displacement chains, many concurrent gets, mixed new / duplicate inserts, parallel saturation of the probe cap.
- **`test_randomized.cu`** — randomized stress. Uniform-key inserts at small (4 KB table) and gigabyte (1 GB table) scales; a sum-reduction stress test with random keys and deliberate duplicates whose expected per-key sum is computed on the host.

Three shared infrastructure headers:

- **`tests/tests.cuh`** — the `CUDA_CHECK` postfix error-check operator (`expr >> CUDA_CHECK`).
- **`tests/kernels.cuh`** — the `Table` alias (with `default_hash`), bulk insert/get host helpers, and end-to-end driver kernels. Used by the "combined" and construction tests.
- **`tests/isolated.cuh`** — the `TestTable` alias (with `IdentityHash` for predictable home buckets), templated `run_insert` / `run_get` / `run_get_one` / `run_insert_with_outcomes`, the `assert_robin_hood_invariant` scanner, and `set_state` / `read_state` for direct table-memory manipulation via `HashTable::data()`.

We use an identity hash function (`IdentityHash`) to facilitate isolated tests: with `hash(K) = K`, the home bucket of a key is fully predictable from its value, so we can construct any valid Robin Hood state with one `cudaMemcpy` and hand-predict the outcome of small insert sequences.
The randomized tests apply a bit-mixing function (the same `fmix32` finalizer used by `default_hash`) as a permutation on sequential integers — this gives distinct random-looking keys without the slow host-side deduplication overhead that an `unordered_set` would require, which is what makes the gigabyte-scale tests run reasonably fast.

The test suite runs about 40 tests across ~1500 lines of test code (plus ~500 lines of shared test infrastructure under `tests.cuh` / `kernels.cuh` / `isolated.cuh`), against ~750 lines of library code under `include/gpurhh/`.
The high ratio is deliberate: this is a concurrent data structure, and the failure modes for concurrent CAS retries, displacement chains, and probe-cap interactions are subtle enough that we put significant effort into checking each path.
