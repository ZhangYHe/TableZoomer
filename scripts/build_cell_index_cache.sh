#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate tablezoomer
fi

mkdir -p logs cache/cell_index

python scripts/build_cell_index_cache.py \
  --dataset_path output/wtq_test_random_50.jsonl \
  --task wtq \
  --cache_dir cache/cell_index \
  --cell_retrieval_method bm25 \
  --overwrite_cache \
  "$@" 2>&1 | tee logs/build_cell_index_cache.log
