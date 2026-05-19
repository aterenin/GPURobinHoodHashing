#!/usr/bin/env bash
#
# Run the gpurhh benchmark sweep, optionally followed by the external
# baseline sweeps (cuCollections, WarpCore).
#
# Writes per-workload CSV files (memcpy.csv, insert.csv, get.csv) plus
# a run_info.txt sidecar and a benchmark.log transcript to the output
# directory. The output dir defaults to output/<timestamp>/ under the
# repo root, so consecutive runs don't collide.
#
# Usage:
#     scripts/benchmark.sh <one-or-more-libraries> [output-dir]
#
# Pick at least one library to run. Combining flags runs each
# selected library's sweep in turn. Passing no flag prints this usage
# and exits — no default, since benchmark runs are heavy.
#
#   --gpurhh    Run the gpurhh sweep (memcpy reference + insert + get).
#   --cuco      Run the cuCollections baseline (must be set up via
#               scripts/setup-baselines.sh).
#   --warpcore  Run the WarpCore baseline (likewise).
#   --all       Shortcut for "every library whose binaries built";
#               missing baselines are skipped with a warning rather
#               than erroring.
#
# Library-specific flags are strict — they error if the corresponding
# binaries weren't built. Use --all if you want lenient "run what's
# there" behavior. All rows land in the same insert.csv / get.csv;
# the `library` column distinguishes them.

set -euo pipefail

# --- argument parsing ------------------------------------------------------

print_usage() {
    sed -n '2,30p' "$0" | sed 's/^# //; s/^#//'
}

RUN_GPURHH=0
RUN_CUCO=0
RUN_WARPCORE=0
LENIENT=0       # set by --all: skip missing baselines instead of erroring
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpurhh)   RUN_GPURHH=1;                                shift ;;
        --cuco)     RUN_CUCO=1;                                  shift ;;
        --warpcore) RUN_WARPCORE=1;                              shift ;;
        --all)      RUN_GPURHH=1; RUN_CUCO=1; RUN_WARPCORE=1
                    LENIENT=1;                                   shift ;;
        --help|-h)  print_usage; exit 0 ;;
        --*)
            echo "Unknown flag: $1" >&2
            echo "Try '$0 --help' for usage." >&2
            exit 1
            ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $((RUN_GPURHH + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
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

# Mirror stdout + stderr to a log file in the output dir.
LOG_FILE="${OUTPUT_DIR}/benchmark.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Per-library binary existence check. For an explicit library flag,
# missing binaries are an error; for --all (LENIENT=1) we drop the
# library silently with a notice and continue.
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

if [[ $((RUN_GPURHH + RUN_CUCO + RUN_WARPCORE)) -eq 0 ]]; then
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

# --- sweep grid (gpurhh) ---------------------------------------------------
#
# Capacity is held fixed at 1 GiB (DRAM-resident, well past the 4090's
# 72 MB L2). α is implicitly 1 (key range = capacity). F sweeps over
# the regime where Robin Hood actually has work to do; block_size
# sweeps to spot-check whether launch shape moves the number.
CAPACITY_BYTES=$((1024 * 1024 * 1024))   # 1 GiB
LOAD_FACTORS=(0.5 1.0 1.5 2.0 3.0)
BLOCK_SIZES=(64 128 256 512 1024)
TAG="sweep"

# --- gpurhh (memcpy reference + insert + get sweeps) ---------------------
if [[ "${RUN_GPURHH}" -eq 1 ]]; then
    echo "==> memcpy baseline (${CAPACITY_BYTES} B)"
    "${BUILD_DIR}/benchmark_memcpy" \
        --output-dir "${OUTPUT_DIR}" \
        --bytes "${CAPACITY_BYTES}" \
        --tag "${TAG}"

    for f in "${LOAD_FACTORS[@]}"; do
        for b in "${BLOCK_SIZES[@]}"; do
            echo "==> gpurhh insert F=${f} block=${b}"
            "${BUILD_DIR}/benchmark_insert" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity-bytes "${CAPACITY_BYTES}" \
                --load-factor "${f}" \
                --block-size "${b}" \
                --tag "${TAG}"

            echo "==> gpurhh get    F=${f} block=${b}"
            "${BUILD_DIR}/benchmark_get" \
                --output-dir "${OUTPUT_DIR}" \
                --capacity-bytes "${CAPACITY_BYTES}" \
                --load-factor "${f}" \
                --block-size "${b}" \
                --tag "${TAG}"
        done
    done
fi

# --- cuCollections (fixed-capacity static_map; F < 1, no block sweep) -----
if [[ "${RUN_CUCO}" -eq 1 ]]; then
    CUCO_DIR="${BUILD_DIR}/baselines/cuco"
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
fi

# --- WarpCore (same shape as cuco) ----------------------------------------
if [[ "${RUN_WARPCORE}" -eq 1 ]]; then
    WARPCORE_DIR="${BUILD_DIR}/baselines/warpcore"
    WARPCORE_LOAD_FACTORS=(0.5 0.7 0.85 0.95)
    for f in "${WARPCORE_LOAD_FACTORS[@]}"; do
        echo "==> warpcore insert F=${f}"
        "${WARPCORE_DIR}/benchmark_insert" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --tag "${TAG}"

        echo "==> warpcore get    F=${f}"
        "${WARPCORE_DIR}/benchmark_get" \
            --output-dir "${OUTPUT_DIR}" \
            --capacity-bytes "${CAPACITY_BYTES}" \
            --load-factor "${f}" \
            --tag "${TAG}"
    done
fi

echo "==> Done. Results in ${OUTPUT_DIR}"
