#!/usr/bin/env python3
"""Direct LLM baseline for WikiTableQuestions JSONL files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "output" / "wtq_test_random_50.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "baselines" / "output" / "direct_llm_results.jsonl"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct LLM WTQ baseline.")
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "gpt-5.4"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument(
        "--max_table_chars",
        type=int,
        default=0,
        help="Maximum CSV characters to send. 0 means send the full table.",
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def read_jsonl(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_table_csv(table_file: str, max_table_chars: int) -> tuple[str, bool]:
    table_text = Path(table_file).read_text(encoding="utf-8")
    if max_table_chars > 0 and len(table_text) > max_table_chars:
        return table_text[:max_table_chars], True
    return table_text, False


def build_prompt(question: str, table_text: str) -> str:
    return (
        "Table (CSV):\n"
        f"{table_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question based on the table. Output only the final answer."
    )


def normalize_answer(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff.]+", " ", text)
    return text.strip()


def is_correct_answer(prediction: str, gold_answer: object) -> bool:
    if gold_answer is None:
        return False

    if isinstance(gold_answer, list):
        gold_values = gold_answer
    else:
        gold_values = [gold_answer]

    gold_norms = [normalize_answer(value) for value in gold_values if normalize_answer(value)]
    pred_norm = normalize_answer(prediction)
    if not gold_norms or not pred_norm:
        return False

    if pred_norm in gold_norms:
        return True
    return all(gold in pred_norm for gold in gold_norms)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def parse_chat_response(response_body: bytes) -> str:
    data = json.loads(response_body.decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"LLM response has no choices: {data}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError(f"LLM response has no message content: {data}")
    return str(content).strip()


def call_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    request_timeout: float,
    max_retries: int,
) -> str:
    url = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return parse_chat_response(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"LLM request failed after {max_retries + 1} attempts: {last_error}")


def run_one(
    idx: int,
    item: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    max_table_chars: int,
    request_timeout: float,
    max_retries: int,
) -> tuple[int, dict[str, Any]]:
    result = dict(item)
    result["model_name"] = model_name

    try:
        table_text, truncated = read_table_csv(str(item["table_file"]), max_table_chars)
        prompt = build_prompt(str(item["question"]), table_text)
        answer = call_chat_completion(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        result["response"] = answer
        result["pred_answer"] = answer
        result["execute_status"] = "success"
        result["table_truncated"] = truncated
    except Exception as exc:
        result["response"] = ""
        result["pred_answer"] = ""
        result["execute_status"] = "fail"
        result["error"] = str(exc)

    return idx, result


def write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.max_retries < 0:
        raise ValueError(f"--max_retries must be non-negative, got {args.max_retries}")
    if args.max_table_chars < 0:
        raise ValueError(f"--max_table_chars must be non-negative, got {args.max_table_chars}")

    env_file = Path(args.env_file).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY is required in {env_file} or the environment")

    rows = read_jsonl(dataset_path, args.limit)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    completed = 0
    failed = 0
    correct = 0
    incorrect = 0

    print(
        f"Running direct LLM baseline: examples={len(rows)} workers={args.workers} "
        f"model={args.model_name}",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                idx,
                item,
                api_key=api_key,
                base_url=base_url,
                model_name=args.model_name,
                max_table_chars=args.max_table_chars,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            for idx, item in enumerate(rows)
        ]

        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if result.get("execute_status") != "success":
                failed += 1
                progress_result = "error"
            elif "answer" in result:
                progress_result = (
                    "correct"
                    if is_correct_answer(str(result.get("pred_answer", "")), result["answer"])
                    else "incorrect"
                )
            else:
                progress_result = "unknown"
            if progress_result == "correct":
                correct += 1
            elif progress_result == "incorrect":
                incorrect += 1
            example_id = result.get("id") or f"row_{idx}"
            acc = correct / completed if completed else 0.0
            print(
                f"[PROGRESS] {completed}/{len(rows)} "
                f"id={example_id} result={progress_result} "
                f"status={result.get('execute_status')} "
                f"correct={correct} incorrect={incorrect} failed={failed} "
                f"acc={acc:.4f}",
                flush=True,
            )

    final_results = [row for row in results if row is not None]
    write_jsonl(output_path, final_results)
    print(f"Saved results to {output_path}", flush=True)

    if failed:
        print(f"Completed with {failed} failed examples.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
