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

WORKERS="${WORKERS:-4}"
DATASET_PATH="/home/zhangyunhe/nas/code/table/TableZoomer/output/wtq_test_random_50_error.jsonl"
RUN_DIR="output/wtq50_$(date +%m%d%H%M)"
SHARD_DIR="${RUN_DIR}/shards"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "${arg}" == "--limit" || "${arg}" == --limit=* ]]; then
    echo "--limit is not supported by this parallel script. Use scripts/run_cell_guided_zoom_test.sh for limited non-parallel runs." >&2
    exit 1
  fi
done

mkdir -p "${SHARD_DIR}"

for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
  : > "${shard_path}"
done

awk -v workers="${WORKERS}" -v out_dir="${SHARD_DIR}" '
  NF {
    shard = count % workers
    path = sprintf("%s/shard_%02d.jsonl", out_dir, shard)
    print > path
    count++
  }
  END {
    print count
  }
' "${DATASET_PATH}" > "${RUN_DIR}/shard_count.txt"

echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Workers: ${WORKERS}"
echo "Examples: $(cat "${RUN_DIR}/shard_count.txt")"

pids=()
for ((worker = 0; worker < WORKERS; worker++)); do
  shard_path=$(printf "%s/shard_%02d.jsonl" "${SHARD_DIR}" "${worker}")
  worker_dir="${RUN_DIR}/worker_${worker}"
  mkdir -p "${worker_dir}"

  (
    python scripts/run_tablezoomer_dataset.py \
      --env_file .env \
      --model_name gpt-5.4 \
      --dataset_path "${shard_path}" \
      --react_round 2 \
      --task wtq \
      --use_cell_guided_zoom \
      --cell_retrieval_method bm25 \
      --top_k_cells 20 \
      --top_k_rows 10 \
      --top_k_cols 10 \
      --output_path "${worker_dir}/results.jsonl" \
      "$@" 2>&1 \
      | tee "${worker_dir}/run.log" \
      | grep --line-buffered -E '^\[PROGRESS\]|^Saved results' \
      | sed -u "s/^/[W${worker}] /"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one worker failed. Check ${RUN_DIR}/worker_*/run.log" >&2
  exit 1
fi

: > "${RUN_DIR}/results.jsonl"
for ((worker = 0; worker < WORKERS; worker++)); do
  worker_result="${RUN_DIR}/worker_${worker}/results.jsonl"
  if [[ ! -f "${worker_result}" ]]; then
    echo "Missing worker result file: ${worker_result}" >&2
    exit 1
  fi
  cat "${worker_result}" >> "${RUN_DIR}/results.jsonl"
done

echo "Saved merged results to ${RUN_DIR}/results.jsonl"
