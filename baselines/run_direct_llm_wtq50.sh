#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate tablezoomer
fi

DATASET_PATH="${DATASET_PATH:-output/wtq_test_random_50.jsonl}"
MODEL_NAME="gpt-5.4"
EVALUATE="${EVALUATE:-1}"
RUN_DIR="${RUN_DIR:-baselines/output/gpt_5.4_direct_llm_wtq50_$(date +%m%d%H%M)}"
RESULT_PATH="${RUN_DIR}/results.jsonl"

# Modify this value directly to change the number of concurrent requests.
WORKERS=4

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Model: ${MODEL_NAME}"
echo "Workers: ${WORKERS}"

python baselines/direct_llm_baseline.py \
  --env_file .env \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${RESULT_PATH}" \
  --model_name "${MODEL_NAME}" \
  --workers "${WORKERS}" \
  "$@" 2>&1 | tee "${RUN_DIR}/run.log"

echo "Saved results to ${RESULT_PATH}"

if [[ "${EVALUATE}" == "1" ]]; then
  scripts/evaluate_wtq50_tablezoomer.sh "${RESULT_PATH}"
fi
