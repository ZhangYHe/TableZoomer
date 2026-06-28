#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate tablezoomer
fi

export METAGPT_PROJECT_ROOT="${METAGPT_PROJECT_ROOT:-${HOME}/.metagpt/tablezoomer_runtime}"

mkdir -p logs output

python scripts/run_tablezoomer_dataset.py \
  --env_file .env \
  --model_name gpt-5.4 \
  --dataset_path output/wtq_test_random_50.jsonl \
  --limit 2 \
  --react_round 2 \
  --task wtq \
  --use_cell_guided_zoom \
  --cell_retrieval_method bm25 \
  --top_k_cells 20 \
  --top_k_rows 10 \
  --top_k_cols 10 \
  --output_path output/wtq_cell_guided_zoom_test.jsonl \
  "$@" 2>&1 | tee logs/cell_guided_zoom_test.log
