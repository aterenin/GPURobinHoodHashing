#!/usr/bin/env bash
#
# Populate external/ with the header subtrees of the optional baseline
# libraries the benchmark suite compares against. Neither is required
# to build or use gpurhh itself — these exist only for the benchmark
# comparison plots.
#
# Pins are inlined below. To upgrade a baseline, bump its commit hash
# and re-run this script.
#
# Implementation: partial clone with --filter=blob:none plus a
# sparse-checkout limited to `include/`. The full upstream tree is
# never materialized.
#
# Idempotent — safe to re-run after a pin change to refresh.

set -euo pipefail

# --- pinned versions -------------------------------------------------------
# Format per entry: "<name> <url> <commit>"
#
# hpc_helpers is a transitive dependency of warpcore (provides
# <helpers/cuda_helpers.cuh>, <helpers/packed_types.cuh>). It is hosted
# on a self-hosted GitLab; the `restructure` branch is what warpcore's
# CMakeLists pins via CPM. We mirror that pin here.
LIBS=(
    "cuCollections https://github.com/NVIDIA/cuCollections.git    d4e84ee20b9185a3aa279ce184416bd41e53287f"
    "warpcore      https://github.com/sleeepyjack/warpcore.git    1a2fe03d438e8dfa1dbd8d9da73f183bd2d72051"
    "hpc_helpers   https://gitlab.rlp.net/pararch/hpc_helpers.git 232b928cd69bc29ad3d9c65d661bd1073cf4b779"
)

# --- script body -----------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="${REPO_ROOT}/external"
INCLUDE_DIR="${EXTERNAL}/include"
mkdir -p "${INCLUDE_DIR}"

for entry in "${LIBS[@]}"; do
    read -r name url commit <<< "${entry}"
    clone="${EXTERNAL}/${name}"

    echo "==> ${name} @ ${commit}"

    rm -rf "${clone}"
    git clone --quiet --filter=blob:none --no-checkout "${url}" "${clone}"
    # Non-cone sparse-checkout with an anchored pattern. Cone mode would
    # also pull every root-level file (LICENSE, CMakeLists.txt, README,
    # etc.) alongside include/; the pattern below restricts the working
    # tree to just the include subtree.
    git -C "${clone}" sparse-checkout set --no-cone '/include/'
    git -C "${clone}" checkout --quiet "${commit}"

    # Move each namespace subdir (e.g. cuco/, warpcore/) into the shared
    # external/include/ tree, replacing any prior copy. Then delete the
    # clone — including its .git — since we only need the headers.
    for ns_dir in "${clone}/include"/*; do
        ns="$(basename "${ns_dir}")"
        rm -rf "${INCLUDE_DIR}/${ns}"
        mv "${ns_dir}" "${INCLUDE_DIR}/${ns}"
    done
    rm -rf "${clone}"
done

echo "Done."
