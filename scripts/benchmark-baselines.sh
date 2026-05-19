#!/usr/bin/env bash
#
# Run benchmark sweeps against the external baseline libraries that
# scripts/setup-baselines.sh has populated under external/include/.
#
# Same output-dir convention as scripts/benchmark.sh: defaults to a
# fresh output/<timestamp>/ under the repo root, or whatever path is
# passed as the first positional arg. Pass the timestamped dir of an
# existing gpurhh run if you want the baseline CSV rows to land in the
# same insert.csv / get.csv as the gpurhh rows for overlaid plotting.
#
# Each baseline library is run only if its binaries built — controlled
# at make time by whether external/include/<namespace>/ exists. If
# neither built, the script does nothing (and says so).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/benchmarks"

# Output dir: optional positional arg, else a fresh timestamped dir.
if [[ $# -ge 1 ]]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${REPO_ROOT}/output/$(date +%Y-%m-%d_%H-%M-%S)"
fi
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/benchmark-baselines.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==> Writing CSV output and run_info to ${OUTPUT_DIR}"

# Environment sidecar (overwrites any from scripts/benchmark.sh — same
# machine, so the values are identical).
{
    echo "date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname:       $(uname -srm)"
    echo "nvcc:        $(nvcc --version 2>/dev/null | tail -n1 || echo 'not found')"
    echo "nvidia-smi:  $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader 2>/dev/null || echo 'not found')"
} > "${OUTPUT_DIR}/run_info.txt"

# Shared workload knobs. Capacity matches scripts/benchmark.sh exactly
# so baseline rows land in the same regime.
CAPACITY_BYTES=$((1 << 30))
TAG="sweep"

ran_any=0

# --- cuCollections -----------------------------------------------------
#
# cuco::static_map has a fixed capacity and can't sustain load factor
# above 1.0, so its F sweep stops short of the gpurhh range. cuco's
# bulk API picks its own launch configuration; block_size is not swept.
CUCO_DIR="${BUILD_DIR}/baselines/cuco"
if [[ -x "${CUCO_DIR}/benchmark_insert" && -x "${CUCO_DIR}/benchmark_get" ]]; then
    CUCO_LOAD_FACTORS=(0.5 0.7 0.85 0.95)
    for f in "${CUCO_LOAD_FACTORS[@]}"; do
        echo "==> cuco insert F=${f}"
        "${CUCO_DIR}/benchmark_insert" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --tag "${TAG}"

        echo "==> cuco get    F=${f}"
        "${CUCO_DIR}/benchmark_get" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --tag "${TAG}"
    done
    ran_any=1
fi

# --- WarpCore (placeholder) -------------------------------------------
#
# WARPCORE_DIR="${BUILD_DIR}/baselines/warpcore"
# if [[ -x "${WARPCORE_DIR}/benchmark_insert" && -x "${WARPCORE_DIR}/benchmark_get" ]]; then
#     ... (to be added once the warpcore binaries are written)
#     ran_any=1
# fi

if [[ "${ran_any}" -eq 0 ]]; then
    echo "No baseline binaries found under ${BUILD_DIR}/baselines/." >&2
    echo "Run scripts/setup-baselines.sh to populate external/include/, then 'make benchmarks'." >&2
    exit 1
fi

echo "==> Done. Results in ${OUTPUT_DIR}"
