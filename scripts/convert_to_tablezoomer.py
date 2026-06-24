#!/usr/bin/env python3
"""Convert table QA datasets to the JSONL format consumed by TableZoomer."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_WTQ_PATH = "/home/zhangyunhe/nas/dataset/WikiTableQuestions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert datasets to TableZoomer JSONL.")
    parser.add_argument("--task", default="wtq", choices=["wtq"])
    parser.add_argument("--dataset_path", default=DEFAULT_WTQ_PATH)
    parser.add_argument(
        "--split",
        required=True,
        help="WTQ split: train, test, dev, pristine-unseen-tables, pristine-seen-tables, training-before300, or random-split-N-train/dev.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_metadata", action="store_true")
    return parser.parse_args()


def resolve_wtq_split_file(dataset_path: Path, split: str) -> Path:
    split_map = {
        "train": "training.tsv",
        "training": "training.tsv",
        "test": "pristine-unseen-tables.tsv",
        "pristine-unseen-tables": "pristine-unseen-tables.tsv",
        "dev": "pristine-seen-tables.tsv",
        "validation": "pristine-seen-tables.tsv",
        "pristine-seen-tables": "pristine-seen-tables.tsv",
        "training-before300": "training-before300.tsv",
    }

    if split.startswith("random-split-"):
        parts = split.split("-")
        if len(parts) == 4 and parts[0] == "random" and parts[1] == "split":
            seed, subset = parts[2], parts[3]
            if seed in {"1", "2", "3", "4", "5"} and subset in {"train", "dev"}:
                split_map[split] = f"{split}.tsv"

    if split not in split_map:
        random_splits = [
            f"random-split-{seed}-{subset}"
            for seed in range(1, 6)
            for subset in ("train", "dev")
        ]
        valid = sorted(list(split_map) + random_splits)
        raise ValueError(f"Unsupported WTQ split '{split}'. Valid splits: {', '.join(valid)}")

    split_file = dataset_path / "data" / split_map[split]
    if not split_file.is_file():
        raise FileNotFoundError(f"WTQ split file does not exist: {split_file}")
    return split_file


def unescape_wtq_value(value: str) -> str:
    return value.replace(r"\n", " ").replace(r"\p", "|").replace(r"\\", "\\").strip()


def split_wtq_answer(answer: str) -> list[str]:
    if answer == "":
        return []
    return [unescape_wtq_value(part) for part in answer.split("|")]


def load_wtq_examples(dataset_path: Path, split: str) -> list[dict]:
    split_file = resolve_wtq_split_file(dataset_path, split)
    examples: list[dict] = []

    with split_file.open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file, delimiter="\t")
        required_fields = {"id", "utterance", "context", "targetValue"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"Unexpected WTQ fields in {split_file}: {reader.fieldnames}. "
                f"Expected at least: {sorted(required_fields)}"
            )

        for row in reader:
            table_id = row["context"]
            table_file = dataset_path / table_id
            if not table_file.is_file():
                raise FileNotFoundError(f"WTQ table file does not exist: {table_file}")

            examples.append(
                {
                    "question": unescape_wtq_value(row["utterance"]),
                    "table_file": str(table_file.resolve()),
                    "answer": split_wtq_answer(row["targetValue"]),
                    "dataset": "wtq",
                    "split": split,
                    "id": row["id"],
                    "table_id": table_id,
                }
            )

    return examples


def select_examples(examples: list[dict], limit: int | None, shuffle: bool, seed: int) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

    selected = list(examples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return selected


def write_jsonl(examples: list[dict], output_path: Path, include_metadata: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            item = dict(example)
            if not include_metadata:
                for key in ("dataset", "split", "id", "table_id"):
                    item.pop(key, None)
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_path}")

    if args.task != "wtq":
        raise ValueError(f"Unsupported task: {args.task}")

    examples = load_wtq_examples(dataset_path, args.split)
    selected = select_examples(examples, args.limit, args.shuffle, args.seed)
    output_path = Path(args.output_path).expanduser()
    write_jsonl(selected, output_path, args.include_metadata)

    print(f"Converted {len(selected)} {args.task} examples from split '{args.split}' to {output_path}")


if __name__ == "__main__":
    main()
