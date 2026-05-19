#!/usr/bin/env bash
#
# Run the gpurhh benchmark sweeps and optional external baselines.
#
# The benchmark suite is split into two studies, each writing its own
# CSV directory tree under the output dir:
#
#   timing/   apples-to-apples kernel throughput (memcpy.csv, insert.csv,
#             get.csv). gpurhh + optional cuco / warpcore baselines all
#             append to the same per-workload CSV; the `library` column
#             distinguishes their rows.
#
#   memory/   gpurhh's counter-instrumented probe / failure / hit study
#             (insert.csv, get.csv with extra count columns). No
#             baselines — cuco / warpcore don't expose these counters.
#
# Also written: benchmark.log (transcript) and run_info.txt (host info),
# both at the top of the output dir. Output defaults to
# output/<timestamp>/ under the repo root, so consecutive runs don't
# collide.
#
# Usage:
#     scripts/benchmark.sh <one-or-more-flags> [output-dir]
#
# Pick at least one flag. Passing none prints this usage and exits.
#
#   --timing     Run gpurhh's timing sweep (memcpy + insert + get).
#   --memory     Run gpurhh's memory-utilization sweep
#                (probe / failure / hit counters).
#   --cuco       Run the cuCollections baseline (timing only).
#   --warpcore   Run the WarpCore baseline (timing only).
#   --all        Shortcut for every flag above; missing binaries are
#                skipped with a notice rather than erroring.
#
# Individual flags are strict — they error if the corresponding binaries
# weren't built. --all is lenient.

set -euo pipefail

# --- argument parsing ------------------------------------------------------

print_usage() {
    sed -n '2,38p' "$0" | sed 's/^# //; s/^#//'
}

RUN_TIMING=0
RUN_MEMORY=0
RUN_CUCO=0
RUN_WARPCORE=0
LENIENT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timing)    RUN_TIMING=1;   shift ;;
        --memory)    RUN_MEMORY=1;   shift ;;
        --cuco)      RUN_CUCO=1;     shift ;;
        --warpcore)  RUN_WARPCORE=1; shift ;;
        --all)       RUN_TIMING=1; RUN_MEMORY=1
                     RUN_CUCO=1;   RUN_WARPCORE=1
                     LENIENT=1;     shift ;;
        --help|-h)   print_usage; exit 0 ;;
        --*)
            echo "Unknown flag: $1" >&2
            echo "Try '$0 --help' for usage." >&2
            exit 1
            ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $((RUN_TIMING + RUN_MEMORY + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
    print_usage >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/benchmarks"
TIMING_BIN="${BUILD_DIR}/timing"
MEMORY_BIN="${BUILD_DIR}/memory"
CUCO_BIN="${TIMING_BIN}/baselines/cuco"
WARPCORE_BIN="${TIMING_BIN}/baselines/warpcore"

if [[ $# -ge 1 ]]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${REPO_ROOT}/output/$(date +%Y-%m-%d_%H-%M-%S)"
fi
TIMING_OUT="${OUTPUT_DIR}/timing"
MEMORY_OUT="${OUTPUT_DIR}/memory"
mkdir -p "${OUTPUT_DIR}" "${TIMING_OUT}" "${MEMORY_OUT}"

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

if [[ "${RUN_TIMING}" -eq 1 ]]; then
    require_or_drop "timing" \
        "${TIMING_BIN}/benchmark_memcpy" \
        "${TIMING_BIN}/benchmark_insert" \
        "${TIMING_BIN}/benchmark_get" \
        || RUN_TIMING=0
fi
if [[ "${RUN_MEMORY}" -eq 1 ]]; then
    require_or_drop "memory" \
        "${MEMORY_BIN}/benchmark_insert" \
        "${MEMORY_BIN}/benchmark_get" \
        || RUN_MEMORY=0
fi
if [[ "${RUN_CUCO}" -eq 1 ]]; then
    require_or_drop "cuco" \
        "${CUCO_BIN}/benchmark_insert" \
        "${CUCO_BIN}/benchmark_get" \
        || RUN_CUCO=0
fi
if [[ "${RUN_WARPCORE}" -eq 1 ]]; then
    require_or_drop "warpcore" \
        "${WARPCORE_BIN}/benchmark_insert" \
        "${WARPCORE_BIN}/benchmark_get" \
        || RUN_WARPCORE=0
fi

if [[ $((RUN_TIMING + RUN_MEMORY + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
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
# Capacity is fixed at 1 GiB worth of slots (DRAM-resident). Keys are
# raw uint32 samples (Uniform on [0, 2^32)), so α = 2^32 / capacity ≈ 32
# — essentially unique keys, the collision-resolution regime where table
# designs differ. We sweep n_ops via convenient multipliers below.
# block_size is swept for gpurhh; cuco/warpcore have their own internal
# launch shapes.
SLOT_BYTES=8
CAPACITY=$(( (1024 * 1024 * 1024) / SLOT_BYTES ))   # 1 GiB / 8 = 2^27
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

# --- timing sweep (memcpy + insert + get for gpurhh) ---------------------
if [[ "${RUN_TIMING}" -eq 1 ]]; then
    BYTES=$(( CAPACITY * SLOT_BYTES ))
    echo "==> memcpy baseline (${BYTES} B)"
    "${TIMING_BIN}/benchmark_memcpy" \
        --output-dir "${TIMING_OUT}" \
        --bytes "${BYTES}" \
        --reps "${REPS}" \
        --seed "${SEED}" \
        --tag "${TAG}"

    for n_ops in "${GPURHH_NOPS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> timing insert n_ops=${n_ops} block=${b}"
            "${TIMING_BIN}/benchmark_insert" \
                --output-dir "${TIMING_OUT}" \
                --capacity "${CAPACITY}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"

            echo "==> timing get    n_ops=${n_ops} block=${b}"
            "${TIMING_BIN}/benchmark_get" \
                --output-dir "${TIMING_OUT}" \
                --capacity "${CAPACITY}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"
        done
    done
fi

# --- memory sweep (counter-instrumented gpurhh study) --------------------
if [[ "${RUN_MEMORY}" -eq 1 ]]; then
    for n_ops in "${GPURHH_NOPS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> memory insert n_ops=${n_ops} block=${b}"
            "${MEMORY_BIN}/benchmark_insert" \
                --output-dir "${MEMORY_OUT}" \
                --capacity "${CAPACITY}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"

            echo "==> memory get    n_ops=${n_ops} block=${b}"
            "${MEMORY_BIN}/benchmark_get" \
                --output-dir "${MEMORY_OUT}" \
                --capacity "${CAPACITY}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"
        done
    done
fi

# --- cuCollections baseline (timing only) --------------------------------
if [[ "${RUN_CUCO}" -eq 1 ]]; then
    for n_ops in "${BASELINE_NOPS[@]}"; do
        echo "==> cuco insert n_ops=${n_ops}"
        "${CUCO_BIN}/benchmark_insert" \
            --output-dir "${TIMING_OUT}" \
            --capacity "${CAPACITY}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"

        echo "==> cuco get    n_ops=${n_ops}"
        "${CUCO_BIN}/benchmark_get" \
            --output-dir "${TIMING_OUT}" \
            --capacity "${CAPACITY}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"
    done
fi

# --- WarpCore baseline (timing only) ------------------------------------
if [[ "${RUN_WARPCORE}" -eq 1 ]]; then
    for n_ops in "${BASELINE_NOPS[@]}"; do
        echo "==> warpcore insert n_ops=${n_ops}"
        "${WARPCORE_BIN}/benchmark_insert" \
            --output-dir "${TIMING_OUT}" \
            --capacity "${CAPACITY}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"

        echo "==> warpcore get    n_ops=${n_ops}"
        "${WARPCORE_BIN}/benchmark_get" \
            --output-dir "${TIMING_OUT}" \
            --capacity "${CAPACITY}" \
            --n-ops "${n_ops}" \
            --reps "${REPS}" \
            --seed "${SEED}" \
            --tag "${TAG}"
    done
fi

echo "==> Done. Results in ${OUTPUT_DIR}"
