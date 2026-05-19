#!/usr/bin/env bash
#
# Run the gpurhh benchmark sweep, optionally followed by the external
# baseline sweeps (cuCollections, WarpCore).
#
# Writes per-workload CSV files (memcpy.csv, insert.csv, get.csv) and
# their counter-instrumented siblings (insert_counters.csv,
# get_counters.csv) plus a run_info.txt sidecar and a benchmark.log
# transcript to the output directory. The output dir defaults to
# output/<timestamp>/ under the repo root, so consecutive runs don't
# collide.
#
# Usage:
#     scripts/benchmark.sh <one-or-more-libraries> [output-dir]
#
# Pick at least one library to run. Combining flags runs each
# selected library's sweep in turn. Passing no flag prints this usage
# and exits — no default, since benchmark runs are heavy.
#
#   --gpurhh           Run the gpurhh comparison sweep (memcpy + insert + get).
#   --gpurhh-counters  Run the gpurhh counter-instrumented sweep
#                      (insert_counters + get_counters; produces probe
#                      / failure / hit columns).
#   --cuco             Run the cuCollections baseline.
#   --warpcore         Run the WarpCore baseline.
#   --all              Shortcut for every library whose binaries built;
#                      missing baselines are skipped with a notice
#                      rather than erroring.
#
# Library-specific flags are strict — they error if the corresponding
# binaries weren't built. --all is lenient. Comparison rows from all
# libraries land in the same insert.csv / get.csv; the `library` column
# distinguishes them. Counter rows go to separate _counters.csv files.

set -euo pipefail

# --- argument parsing ------------------------------------------------------

print_usage() {
    sed -n '2,32p' "$0" | sed 's/^# //; s/^#//'
}

RUN_GPURHH=0
RUN_GPURHH_COUNTERS=0
RUN_CUCO=0
RUN_WARPCORE=0
LENIENT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpurhh)           RUN_GPURHH=1;          shift ;;
        --gpurhh-counters)  RUN_GPURHH_COUNTERS=1; shift ;;
        --cuco)             RUN_CUCO=1;            shift ;;
        --warpcore)         RUN_WARPCORE=1;        shift ;;
        --all)              RUN_GPURHH=1; RUN_GPURHH_COUNTERS=1
                            RUN_CUCO=1; RUN_WARPCORE=1
                            LENIENT=1;             shift ;;
        --help|-h)          print_usage; exit 0 ;;
        --*)
            echo "Unknown flag: $1" >&2
            echo "Try '$0 --help' for usage." >&2
            exit 1
            ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $((RUN_GPURHH + RUN_GPURHH_COUNTERS + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
    print_usage >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/benchmarks"

if [[ $# -ge 1 ]]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${REPO_ROOT}/output/$(date +%Y-%m-%d_%H-%M-%S)"
fi
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

require_or_drop() {
    local label="$1"; shift
    for bin in "$@"; do
        if [[ ! -x "${bin}" ]]; then
            if [[ "${LENIENT}" -eq 1 ]]; then
                echo "==> ${label} binaries not built; skipping (--all is lenient)" >&2
                return 1
            fi
            echo "Missing executable: ${bin}" >&2
            echo "Build first with:   make benchmarks  (and scripts/setup-baselines.sh for baselines)" >&2
            exit 1
        fi
    done
    return 0
}

if [[ "${RUN_GPURHH}" -eq 1 ]]; then
    require_or_drop "gpurhh" \
        "${BUILD_DIR}/benchmark_memcpy" \
        "${BUILD_DIR}/benchmark_insert" \
        "${BUILD_DIR}/benchmark_get" \
        || RUN_GPURHH=0
fi
if [[ "${RUN_GPURHH_COUNTERS}" -eq 1 ]]; then
    require_or_drop "gpurhh-counters" \
        "${BUILD_DIR}/benchmark_insert_counters" \
        "${BUILD_DIR}/benchmark_get_counters" \
        || RUN_GPURHH_COUNTERS=0
fi
if [[ "${RUN_CUCO}" -eq 1 ]]; then
    require_or_drop "cuco" \
        "${BUILD_DIR}/baselines/cuco/benchmark_insert" \
        "${BUILD_DIR}/baselines/cuco/benchmark_get" \
        || RUN_CUCO=0
fi
if [[ "${RUN_WARPCORE}" -eq 1 ]]; then
    require_or_drop "warpcore" \
        "${BUILD_DIR}/baselines/warpcore/benchmark_insert" \
        "${BUILD_DIR}/baselines/warpcore/benchmark_get" \
        || RUN_WARPCORE=0
fi

if [[ $((RUN_GPURHH + RUN_GPURHH_COUNTERS + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
    echo "Nothing to run." >&2
    exit 1
fi

echo "==> Writing CSV output and run_info to ${OUTPUT_DIR}"

{
    echo "date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname:       $(uname -srm)"
    echo "nvcc:        $(nvcc --version 2>/dev/null | tail -n1 || echo 'not found')"
    echo "nvidia-smi:  $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader 2>/dev/null || echo 'not found')"
} > "${OUTPUT_DIR}/run_info.txt"

# --- sweep grid ----------------------------------------------------------
#
# Capacity is fixed at 1 GiB worth of slots (DRAM-resident). α = 1 by
# default: key_range = capacity. We sweep n_ops via convenient
# multipliers, computed once below into a list of raw integer counts.
# block_size is swept for gpurhh; cuco/warpcore have their own internal
# launch shapes.
SLOT_BYTES=8
CAPACITY=$(( (1024 * 1024 * 1024) / SLOT_BYTES ))   # 1 GiB / 8 = 2^27
KEY_RANGE=${CAPACITY}                                # α = 1
BLOCK_SIZES=(64 128 256 512 1024)
TAG="sweep"

# Each binary invocation runs $REPS timed reps, with the keys refilled
# from the continuing cuRAND stream before every rep. Statistically
# equivalent to $REPS independent seeds but with all CUDA / table /
# generator setup paid once instead of $REPS times.
REPS=16
SEED=1

# n_ops multipliers: f * capacity. We compute the raw integers once so
# the rest of the script passes them as-is. Multipliers for gpurhh
# (which can run past 1.0) and for cuco/warpcore (capped at < 1.0 to
# stay below their fixed-capacity limits).
GPURHH_NOPS=()
for mul in 0.5 1.0 1.5 2.0 3.0; do
    GPURHH_NOPS+=($(python3 -c "print(int($mul * $CAPACITY))"))
done
BASELINE_NOPS=()
for mul in 0.5 0.7 0.85 0.95; do
    BASELINE_NOPS+=($(python3 -c "print(int($mul * $CAPACITY))"))
done

# --- gpurhh comparison (memcpy + insert + get sweeps) --------------------
if [[ "${RUN_GPURHH}" -eq 1 ]]; then
    BYTES=$(( CAPACITY * SLOT_BYTES ))
    echo "==> memcpy baseline (${BYTES} B)"
    "${BUILD_DIR}/benchmark_memcpy" \
        --output-dir "${OUTPUT_DIR}" \
        --bytes "${BYTES}" \
        --reps "${REPS}" \
        --seed "${SEED}" \
        --tag "${TAG}"

    for n_ops in "${GPURHH_NOPS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> gpurhh insert n_ops=${n_ops} block=${b}"
            "${BUILD_DIR}/benchmark_insert" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity "${CAPACITY}" \
                --key-range "${KEY_RANGE}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"

            echo "==> gpurhh get    n_ops=${n_ops} block=${b}"
            "${BUILD_DIR}/benchmark_get" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity "${CAPACITY}" \
                --key-range "${KEY_RANGE}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"
        done
    done
fi

# --- gpurhh counters (study the design with instrumentation on) ----------
if [[ "${RUN_GPURHH_COUNTERS}" -eq 1 ]]; then
    for n_ops in "${GPURHH_NOPS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> gpurhh insert_counters n_ops=${n_ops} block=${b}"
            "${BUILD_DIR}/benchmark_insert_counters" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity "${CAPACITY}" \
                --key-range "${KEY_RANGE}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"

            echo "==> gpurhh get_counters    n_ops=${n_ops} block=${b}"
            "${BUILD_DIR}/benchmark_get_counters" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity "${CAPACITY}" \
                --key-range "${KEY_RANGE}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"
        done
    done
fi

# --- cuCollections (fixed-capacity static_map; no block sweep) -----------
if [[ "${RUN_CUCO}" -eq 1 ]]; then
    CUCO_DIR="${BUILD_DIR}/baselines/cuco"
    for n_ops in "${BASELINE_NOPS[@]}"; do
        echo "==> cuco insert n_ops=${n_ops}"
        "${CUCO_DIR}/benchmark_insert" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity "${CAPACITY}" \
            --key-range "${KEY_RANGE}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"

        echo "==> cuco get    n_ops=${n_ops}"
        "${CUCO_DIR}/benchmark_get" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity "${CAPACITY}" \
            --key-range "${KEY_RANGE}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"
    done
fi

# --- WarpCore -----------------------------------------------------------
if [[ "${RUN_WARPCORE}" -eq 1 ]]; then
    WARPCORE_DIR="${BUILD_DIR}/baselines/warpcore"
    for n_ops in "${BASELINE_NOPS[@]}"; do
        echo "==> warpcore insert n_ops=${n_ops}"
        "${WARPCORE_DIR}/benchmark_insert" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity "${CAPACITY}" \
            --key-range "${KEY_RANGE}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"

        echo "==> warpcore get    n_ops=${n_ops}"
        "${WARPCORE_DIR}/benchmark_get" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity "${CAPACITY}" \
            --key-range "${KEY_RANGE}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"
    done
fi

echo "==> Done. Results in ${OUTPUT_DIR}"
