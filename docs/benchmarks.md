# Benchmarks

## Running the benchmarks

`scripts/benchmark.sh` runs the full grid. Flags select which experiment to run:

- `--timing` — gpurhh's timing sweep (insert + get).
- `--memory-bandwidth` — the bandwidth sweep (memcpy ceiling + counter-instrumented gpurhh insert + get).
- `--cuco` — cuCollections baseline, default linear probing (timing only).
- `--cuco-dh` — cuCollections baseline re-specialized with double hashing (timing only).
- `--warpcore` — WarpCore baseline (timing only).
- `--all` — every flag above; missing binaries are skipped with a notice rather than erroring.

Output goes to `output/<timestamp>/timing/{insert,get}.csv` and `output/<timestamp>/memory_bandwidth/{memcpy,insert,get}.csv`, with a `run_info.txt` sidecar (`nvcc --version`, `uname`, `nvidia-smi` GPU summary) and a `benchmark.log` transcript at the top of the output dir.
The seed is fixed at 1 across the sweep; `reps = 16` per invocation, with the cuRAND stream refreshing keys before every rep so each rep is an independent input draw.

`make benchmarks` builds the binaries at `-O3` (the default for the benchmark pattern rule).
The sweep is launched separately from `scripts/benchmark.sh` because runs are heavy.
Numbers from unoptimized builds are meaningless, so the `-O3` default is effectively non-overridable.

## Code structure

Benchmarks come in two kinds — timing and memory bandwidth — and live in `benchmarks/`, split into two studies with their own directory subtree and their own CSV output tree.

- **`benchmarks/timing/`** — apples-to-apples kernel throughput against optional baselines (`cuCollections::static_map`, `WarpCore::SingleValueHashTable`).
  gpurhh's binaries here use `replace_op` reduction and `MaxProbeBuckets = 1 << 20` (effectively uncapped), matching the baselines' unbounded-probe semantics.
  Two executables: `benchmark_insert`, `benchmark_get`.
  Baseline equivalents live under `benchmarks/timing/baselines/{cuco,cuco_dh,warpcore}/` and append to the same per-workload CSVs; the `library` column distinguishes their rows.
  The `cuco/` baseline uses cuCollections's default linear probing (CG=4); the `cuco_dh/` baseline reuses the same headers but specializes `static_map` with `double_hashing<8, ...>`, isolating the probing-scheme variable within a single library so the family-vs-implementation axis is testable separately.
- **`benchmarks/memory_bandwidth/`** — the memory bandwidth experiment.
  `benchmark_memcpy` provides the empirical DRAM ceiling, measured both via the `cudaMemcpyAsync(... D2D)` API and a minimalist `uint4`-vectorized copy kernel; `benchmark_insert` and `benchmark_get` are gpurhh's design under instrumentation (`sum_op` reduction, default `MaxProbeBuckets = 8`, per-tile probe / failure / hit counters via `GPURHH_BENCHMARK_COUNTERS`).
  The CSVs from this experiment let you compute precise per-op DRAM traffic — `total_probes × sizeof(Bucket) / time_ms` — and compare it against memcpy's ceiling.
  This is also where we can probe occupancies above 1, which is only possible because `sum_op` collapses duplicate-key inserts into the existing slot and because the design tolerates (and records) probe-cap failures rather than dropping them silently.
  No library baselines, since neither cuco nor WarpCore exposes the per-probe counters needed for the bandwidth analysis.

The split exists because the two studies want fundamentally different configurations: timing needs the most permissive probe behavior to make per-library performance comparable; the memory bandwidth experiment considers a different setup with capped probes and real reduction work so we can read the probe / failure / hit characteristics back out and convert them into bandwidth numbers.

`benchmark_memcpy` is the peak-bandwidth reference.
Times `cudaMemcpyAsync(... D2D)` and a trivial `uint4`-vectorized copy kernel on a buffer of configurable size.
The CSV stores `dram_bytes = 2 * payload_bytes` (one read of src + one write of dst), so the downstream bandwidth computation is simply `dram_bytes / time_ms` — and this is directly comparable to the hash-table benchmarks' `total_probes × sizeof(Bucket) / time_ms`, since both metrics are bytes through the DRAM controller per second.
Recording the 2× factor at write time rather than relying on downstream scripts to apply it keeps the convention impossible to misuse in analysis.

## Workload hyperparameters

- **`capacity`** is held at 1 GiB by default — well past the L2 cache on any sm_89 device, putting the kernel firmly in the DRAM-bound regime.
- **Key distribution** is raw `Uniform(0, 2^32)` — every uint32 is equally likely, no clamping.
  With capacity = 2^27, this gives `α = key_range / capacity ≈ 32`: keys are effectively unique.
  This is the collision-resolution regime where table designs actually differ; at much smaller α the table fills very slowly with duplicates and linear probing's pathological clustering never engages.
- The _load factor_ **`F = n_ops / capacity`** is the swept axis.
  The timing sweep uses `F ∈ {0.5, 0.75, 0.85, 0.95, 1.0}` — capped at F=1 because gpurhh's timing build sets `MaxProbeBuckets = 1 << 20` (matching the baselines' effectively-unbounded probing), and at F > 1 with α ≈ 32 each doomed insert would walk its full probe budget, this could cause the table to stall, as the kernel would have to loop over the whole table, which would be extremely slow.
  A redraw guard in the timing-insert binary aborts if a cuRAND draw produces `n_unique > capacity` after a few retries.
  The memory-bandwidth sweep extends through `F ∈ {0.5, 0.75, 0.85, 0.95, 1.0, 1.5, 2.0, 3.0}` — with `MaxProbeBuckets = 8` (the design default) failed inserts give up cheaply, so the over-subscription regime is tractable and the failure / hit / probe counters become informative.
- **`block_size ∈ {64, 128, 256, 512, 1024}`** is swept for the gpurhh binaries to spot-check that launch shape doesn't dominate.
  cuco / WarpCore pick their own internal launch shapes.

## Experiment phases

Every insert benchmark structures its work via `run_benchmark_loop`, which carves each rep into three callbacks.
**Only the middle phase (`launch`) sits inside the timing window** — `EventTimer::begin()` / `end_ms()` bracket exactly that callback's content, nothing more.

**Once at startup (untimed, before the loop).**

- Allocate `d_keys`, `d_values`, and `d_sorted` device buffers, each `n_ops × uint32`.
- `cudaMemset(d_values, 0, ...)` once. Under `replace_op` the value content doesn't affect kernel work, so a zero buffer suffices; gpurhh, cuco, and WarpCore all do this so the kernel pays an apples-to-apples `d_values` read.
- Construct the table (`gpurhh::HashTable`, `cuco::static_map`, or `warpcore::SingleValueHashTable`).
- Construct a single `UniformKeyGenerator` (cuRAND Philox4_32_10) seeded with `--seed`.
- For cuco: build a `thrust::counting_iterator → cuco::pair{d_keys[i], d_values[i]}` transform iterator for the bulk-insert call.
- For the get benchmark only: pre-fill the table once via the same bulk-insert path. Pre-fill input variability is a second-order effect for the get throughput we're measuring, so it's paid once rather than per-rep.

**`setup` callback (untimed, runs before each warmup and each timed rep).**

- Empty the table (`table.clear()` / `map.clear_async()` / `table.init()`) — insert only; the get benchmark keeps its pre-filled state across reps.
- Generate fresh input keys: `fill_uniform_keys(d_keys, n_ops, gen)` advances the cuRAND stream by `n_ops` draws.
  Across reps the generator state continues, so the 16 reps see iid draws rather than 16 timings of one frozen input — statistically equivalent to 16 independent seeds but with CUDA / table / generator setup paid once.
- Count distinct inputs for drop accounting (insert only): copy `d_keys → d_sorted`, `thrust::sort`, `thrust::unique`, record `n_unique`.
  This is the ground-truth denominator for the drops measurement: at α ≈ 32 most inputs are unique but ~1–2% are collision duplicates (governed by the mathematics of the birthday problem), and we don't want input duplicates to show up as drops.

**`launch` callback (timed window).**

- One bulk insert: gpurhh's grid-stride insert kernel, cuco's `map.insert_async(pair_begin, pair_begin + n_ops)`, or WarpCore's `table.insert(d_keys, d_values, n_ops)`.
- The get benchmark is analogous: a single bulk get through gpurhh's grid-stride get kernel, cuco's `map.find_async(...)`, or WarpCore's `table.retrieve(...)`.

**`after` callback (untimed, runs after each timed rep).**

- Count occupied slots in the table (insert only):
  - gpurhh: `count_occupied_slots(reinterpret_cast<Slot*>(table.data()), capacity, empty_key)` — a `thrust::count_if` scan in `benchmarks.cuh`. We use the table's `data()` accessor to view the bucket storage as a flat slot array.
  - cuco: `map.size()` (synchronous, built-in).
  - warpcore: `table.size()` (synchronous, built-in).
- Compute `drops = n_unique - occupied` (insert only). This is the count of keys the library tried to insert and couldn't fit, with input duplicates already factored out via `n_unique`.
- Emit one CSV row with `time_ms`, the configuration columns, and `n_unique` / `drops` on the insert side.

## Per-tile probe / failure / hit counters

The `GPURHH_BENCHMARK_COUNTERS` macro (see [Usage: Preprocessor-gated benchmark counters](usage.md#preprocessor-gated-benchmark-counters)) adds a trailing `std::uint32_t&` parameter to `View::insert` / `View::get`.
The memory-benchmark kernels declare a per-tile register accumulator, pass it through every `view.insert` / `view.get` call in the grid-stride loop, and flush one value per tile to a global array at end of kernel.
No atomics fire in the probe hot loop.
The host sums per-tile values into a grand total per rep.

The memory-bandwidth CSVs (`memory_bandwidth/insert.csv`, `memory_bandwidth/get.csv`) add `total_probes` and `total_failures` (insert) or `total_probes`, `total_hits`, `total_misses` (get) on top of the timing CSV schema.
Downstream analysis derives metrics like average probe length (`total_probes / n_ops`) and bandwidth-corrected throughput (`total_probes × sizeof(Bucket) / time_ms`), the latter being directly comparable to the memcpy ceiling in the same directory.

At low F the corrected and uncorrected throughput numbers converge (average probe length ≈ 1).
At high F the corrected number diverges upward — that gap is itself a useful signal about how hard the Robin Hood machinery is working.
