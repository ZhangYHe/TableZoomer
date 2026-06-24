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

python scripts/run_tablezoomer_dataset.py \
  --env_file .env \
  --model_name gpt-5.4 \
  --dataset_path output/wtq_test_random_50.jsonl \
  --limit 2 \
  --react_round 2 \
  --output_path output/wtq_limit_test.jsonl
