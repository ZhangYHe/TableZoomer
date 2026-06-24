#!/usr/bin/env python3
"""Run TableZoomer on a JSONL dataset produced by convert_to_tablezoomer.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TableZoomer on a JSONL dataset.")
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--model_name", default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--react_round", type=int, default=2)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        raise FileNotFoundError(f"Env file does not exist: {env_file}")

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def read_jsonl(dataset_path: Path, limit: int | None) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

    rows: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_runtime_configs(runtime_dir: Path, model_name: str, api_key: str, base_url: str) -> Path:
    llm_config = runtime_dir / "llm_api.yaml"
    tablezoomer_config = runtime_dir / "tablezoomer_api.yaml"

    llm_config.write_text(
        "\n".join(
            [
                "llm:",
                "  api_type: openai",
                f'  base_url: "{base_url}"',
                f'  model: "{model_name}"',
                "  temperature: 0",
                "  calc_usage: false",
                f'  api_key: "{api_key}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    tablezoomer_config.write_text(
        "\n".join(
            [
                "prompt_template:",
                f'  react: "{PROJECT_ROOT / "prompts/react_prompt_en_v4.txt"}"',
                f'  table_desc: "{PROJECT_ROOT / "prompts/table_desc_prompt_en_v3.txt"}"',
                f'  query_expansion: "{PROJECT_ROOT / "prompts/query_refine_en_v1.txt"}"',
                f'  code_generation: "{PROJECT_ROOT / "prompts/code_generate_prompt_en_v11.txt"}"',
                f'  answer_summary: "{PROJECT_ROOT / "prompts/final_answer_prompt_en_v5.txt"}"',
                "",
                "llm_config:",
                f'  react: "{llm_config}"',
                f'  table_desc: "{llm_config}"',
                f'  query_expansion: "{llm_config}"',
                f'  code_generation: "{llm_config}"',
                f'  answer_summary: "{llm_config}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tablezoomer_config


def ensure_metagpt_root() -> None:
    runtime_root = os.environ.get("METAGPT_PROJECT_ROOT")
    if not runtime_root:
        runtime_root = str(Path.home() / ".metagpt" / "tablezoomer_runtime")
        os.environ["METAGPT_PROJECT_ROOT"] = runtime_root
    Path(runtime_root).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY is required in {env_file}")
    if not base_url:
        raise ValueError(f"OPENAI_BASE_URL is required in {env_file}")

    ensure_metagpt_root()
    rows = read_jsonl(dataset_path, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from table_agent import TableZoomer

    with tempfile.TemporaryDirectory(prefix="tablezoomer_run_") as tmp:
        runtime_dir = Path(tmp)
        config_file = write_runtime_configs(runtime_dir, args.model_name, api_key, base_url)
        schema_dir = runtime_dir / "table_schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        agent = TableZoomer(config_file=str(config_file), max_react_round=args.react_round)

        with output_path.open("w", encoding="utf-8") as output_file:
            for idx, item in enumerate(rows):
                question = item["question"]
                table_file = item["table_file"]
                schema_key = item.get("id") or f"row_{idx}"
                table_desc_file = schema_dir / f"{schema_key}.json"
                result = dict(item)

                try:
                    answer, log_item = agent.execute_qa(question, table_file, str(table_desc_file))
                    result["response"] = answer
                    result["pred_answer"] = answer
                    result["log_item"] = log_item
                    result["execute_status"] = "success"
                except Exception as exc:
                    result["response"] = "fail"
                    result["error"] = str(exc)
                    result["execute_status"] = "fail"

                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"[{idx + 1}/{len(rows)}] {result['execute_status']} - {question}")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
