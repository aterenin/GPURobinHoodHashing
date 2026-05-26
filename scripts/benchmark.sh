#!/usr/bin/env bash
#
# Run the gpurhh benchmark sweeps and optional external baselines.
#
# The benchmark suite is split into two studies, each writing its own
# CSV directory tree under the output dir:
#
#   timing/            apples-to-apples kernel throughput (insert.csv,
#                      get.csv). gpurhh + optional cuco / warpcore
#                      baselines all append to the same per-workload
#                      CSV; the `library` column distinguishes rows.
#
#   memory_bandwidth/  the bandwidth study. memcpy.csv gives the DRAM
#                      ceiling; insert.csv / get.csv are gpurhh's
#                      counter-instrumented probe / failure / hit
#                      study, from which precise per-op DRAM traffic
#                      (total_probes × sizeof(Bucket) / time_ms) is
#                      computed and compared against memcpy's ceiling.
#                      No library baselines — cuco / warpcore don't
#                      expose the counters.
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
#   --timing             Run gpurhh's timing sweep (insert + get).
#   --memory-bandwidth   Run the bandwidth sweep (memcpy ceiling +
#                        counter-instrumented gpurhh insert + get).
#   --cuco               Run the cuCollections baseline (timing only).
#   --warpcore           Run the WarpCore baseline (timing only).
#   --all                Shortcut for every flag above; missing binaries
#                        are skipped with a notice rather than erroring.
#
# Individual flags are strict — they error if the corresponding binaries
# weren't built. --all is lenient.

set -euo pipefail

# --- argument parsing ------------------------------------------------------

print_usage() {
    sed -n '2,38p' "$0" | sed 's/^# //; s/^#//'
}

RUN_TIMING=0
RUN_MEMBW=0
RUN_CUCO=0
RUN_WARPCORE=0
LENIENT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timing)            RUN_TIMING=1;   shift ;;
        --memory-bandwidth)  RUN_MEMBW=1;    shift ;;
        --cuco)              RUN_CUCO=1;     shift ;;
        --warpcore)          RUN_WARPCORE=1; shift ;;
        --all)               RUN_TIMING=1; RUN_MEMBW=1
                             RUN_CUCO=1;   RUN_WARPCORE=1
                             LENIENT=1;     shift ;;
        --help|-h)           print_usage; exit 0 ;;
        --*)
            echo "Unknown flag: $1" >&2
            echo "Try '$0 --help' for usage." >&2
            exit 1
            ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $((RUN_TIMING + RUN_MEMBW + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
    print_usage >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/benchmarks"
TIMING_BIN="${BUILD_DIR}/timing"
MEMBW_BIN="${BUILD_DIR}/memory_bandwidth"
CUCO_BIN="${TIMING_BIN}/baselines/cuco"
WARPCORE_BIN="${TIMING_BIN}/baselines/warpcore"

if [[ $# -ge 1 ]]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${REPO_ROOT}/output/$(date +%Y-%m-%d_%H-%M-%S)"
fi
TIMING_OUT="${OUTPUT_DIR}/timing"
MEMBW_OUT="${OUTPUT_DIR}/memory_bandwidth"
mkdir -p "${OUTPUT_DIR}" "${TIMING_OUT}" "${MEMBW_OUT}"

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
        "${TIMING_BIN}/benchmark_insert" \
        "${TIMING_BIN}/benchmark_get" \
        || RUN_TIMING=0
fi
if [[ "${RUN_MEMBW}" -eq 1 ]]; then
    require_or_drop "memory_bandwidth" \
        "${MEMBW_BIN}/benchmark_memcpy" \
        "${MEMBW_BIN}/benchmark_insert" \
        "${MEMBW_BIN}/benchmark_get" \
        || RUN_MEMBW=0
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

if [[ $((RUN_TIMING + RUN_MEMBW + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
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

# n_ops multipliers (F = n_ops / capacity). The timing and
# memory-bandwidth sweeps use different grids because their cost
# profiles differ.
#
# Timing: gpurhh uses MaxProbeBuckets = 1 << 20 to match cuco/warpcore's
# effectively-unbounded probing. At F > 1 with α ≈ 32 (raw uint32 keys),
# the expected n_unique exceeds capacity and each doomed insert walks
# its full probe budget — which translates to multi-hour wall time per
# config. We cap timing at F ≤ 1; the achievable occupancy at F = 1 is
# ~98.5% (132M unique keys in 134M slots due to birthday collisions),
# high enough to exercise the regime where Robin Hood vs. linear probing
# differ. The over-subscription story is studied in the bandwidth sweep.
# Multipliers are expressed as integer percentages of capacity so we can
# stay in bash arithmetic; e.g. 85 means F = 0.85.
TIMING_NOPS=()
for mul_pct in 50 70 85 95 100; do
    TIMING_NOPS+=($(( mul_pct * CAPACITY / 100 )))
done

# Memory-bandwidth: gpurhh uses MaxProbeBuckets = 8 (the design default),
# so failed inserts give up after 8 probe iterations rather than 2^20.
# F > 1 is tractable here, and the per-tile probe / failure counters
# are the whole point of this sweep — they let us derive precise DRAM
# traffic per op and compare against the memcpy ceiling. We include the
# same F ≤ 1 points as the timing sweep (so bandwidth panels line up
# with timing panels at low F) and then extend into the F > 1
# over-subscription regime where failure counters become informative.
MEMBW_NOPS=()
for mul_pct in 50 70 85 95 100 150 200 300; do
    MEMBW_NOPS+=($(( mul_pct * CAPACITY / 100 )))
done

# --- timing sweep (insert + get for gpurhh) ------------------------------
if [[ "${RUN_TIMING}" -eq 1 ]]; then
    for n_ops in "${TIMING_NOPS[@]}"; do
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

# --- memory-bandwidth sweep ---------------------------------------------
# memcpy gives the DRAM ceiling; the counter-instrumented gpurhh inserts /
# gets give precise per-op bandwidth via total_probes × sizeof(Bucket) /
# time_ms. Comparing the two yields the bandwidth story.
if [[ "${RUN_MEMBW}" -eq 1 ]]; then
    BYTES=$(( CAPACITY * SLOT_BYTES ))
    echo "==> memcpy ceiling (${BYTES} B)"
    "${MEMBW_BIN}/benchmark_memcpy" \
        --output-dir "${MEMBW_OUT}" \
        --bytes "${BYTES}" \
        --reps "${REPS}" \
        --seed "${SEED}" \
        --tag "${TAG}"

    for n_ops in "${MEMBW_NOPS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> memory_bandwidth insert n_ops=${n_ops} block=${b}"
            "${MEMBW_BIN}/benchmark_insert" \
                --output-dir "${MEMBW_OUT}" \
                --capacity "${CAPACITY}" \
                --n-ops "${n_ops}" \
                --block-size "${b}" \
                --reps "${REPS}" \
                --seed "${SEED}" \
                --tag "${TAG}"

            echo "==> memory_bandwidth get    n_ops=${n_ops} block=${b}"
            "${MEMBW_BIN}/benchmark_get" \
                --output-dir "${MEMBW_OUT}" \
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
    for n_ops in "${TIMING_NOPS[@]}"; do
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
    for n_ops in "${TIMING_NOPS[@]}"; do
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
