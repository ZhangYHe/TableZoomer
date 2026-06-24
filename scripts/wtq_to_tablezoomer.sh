#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python scripts/convert_to_tablezoomer.py \
  --task wtq \
  --dataset_path /home/zhangyunhe/nas/dataset/WikiTableQuestions \
  --split test \
  --output_path output/wtq_test_random_50.jsonl \
  --limit 50 \
  --shuffle \
  --include_metadata
