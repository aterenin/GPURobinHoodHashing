#!/usr/bin/env bash
#
# Run the full gpurhh benchmark sweep.
#
# Writes per-workload CSV files (memcpy.csv, insert.csv, get.csv) plus
# a run_info.txt sidecar with environment metadata and a benchmark.log
# transcript to the output dir. The output dir defaults to
# output/<timestamp>/ under the repo root, so consecutive runs don't
# collide; override by passing a path as the first argument.
#
# Expects the benchmark binaries to be built at $REPO_ROOT/build/benchmarks/.
# Run `make benchmarks` first; or `make benchmark` to build and sweep in
# one step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/benchmarks"

# Output dir: optional positional arg, else a fresh timestamped dir
# under output/. Date-then-time, sortable, no colons (which would be
# annoying inside paths).
if [[ $# -ge 1 ]]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${REPO_ROOT}/output/$(date +%Y-%m-%d_%H-%M-%S)"
fi
mkdir -p "${OUTPUT_DIR}"

# Mirror everything we print from here on to a log file in the output
# dir, so the run is reproducible from disk afterwards. `tee -a` appends,
# matching the CSV writer's behavior across reruns into the same dir.
LOG_FILE="${OUTPUT_DIR}/benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Make sure the binaries are built.
for bin in benchmark_memcpy benchmark_insert benchmark_get; do
    if [[ ! -x "${BUILD_DIR}/${bin}" ]]; then
        echo "Missing executable: ${BUILD_DIR}/${bin}" >&2
        echo "Build first with:   make benchmarks" >&2
        exit 1
    fi
done

echo "==> Writing CSV output and run_info to ${OUTPUT_DIR}"

# Environment sidecar — captures the machine and toolchain a CSV came
# from. The gpurhh CSV columns themselves stay machine-agnostic.
{
    echo "date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname:       $(uname -srm)"
    echo "nvcc:        $(nvcc --version 2>/dev/null | tail -n1 || echo 'not found')"
    echo "nvidia-smi:  $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader 2>/dev/null || echo 'not found')"
} > "${OUTPUT_DIR}/run_info.txt"

# --- Sweep grid ----------------------------------------------------------
#
# Capacity is held fixed at 1 GiB (DRAM-resident, well past the 4090's
# 72 MB L2). α is implicitly 1 (key range = capacity). F sweeps over
# the regime where Robin Hood actually has work to do; block_size
# sweeps to spot-check whether launch shape moves the number.
CAPACITY_BYTES=$((1 << 30))
LOAD_FACTORS=(0.5 1.0 1.5 2.0 3.0)
BLOCK_SIZES=(64 128 256 512 1024)
TAG="sweep"

# --- Memcpy peak-bandwidth reference -----------------------------------
echo "==> memcpy baseline (${CAPACITY_BYTES} B)"
"${BUILD_DIR}/benchmark_memcpy" \
    --output-dir "${OUTPUT_DIR}" \
    --bytes "${CAPACITY_BYTES}" \
    --tag "${TAG}"

# --- Insert and get sweeps ---------------------------------------------
for f in "${LOAD_FACTORS[@]}"; do
    for b in "${BLOCK_SIZES[@]}"; do
        echo "==> insert F=${f} block=${b}"
        "${BUILD_DIR}/benchmark_insert" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --block-size "${b}" \
            --tag "${TAG}"

        echo "==> get    F=${f} block=${b}"
        "${BUILD_DIR}/benchmark_get" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --block-size "${b}" \
            --tag "${TAG}"
    done
done

echo "==> Done. Results in ${OUTPUT_DIR}"
