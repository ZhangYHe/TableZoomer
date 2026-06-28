#!/usr/bin/env python3
"""Prebuild cell index caches for TableZoomer datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cell index cache files.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache_dir", default=str(PROJECT_ROOT / "cache" / "cell_index"))
    parser.add_argument(
        "--cell_retrieval_method",
        default="bm25",
        choices=["bm25", "hybrid", "embed"],
    )
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--max_row_context_cols", type=int, default=4)
    parser.add_argument("--max_cell_text_chars", type=int, default=180)
    return parser.parse_args()


def read_dataset_tables(dataset_path: Path, limit: int | None) -> list[dict[str, str]]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")

    table_items = []
    seen = set()
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue
            item = json.loads(line)
            table_file = item.get("table_file")
            if not table_file:
                continue
            resolved = str(Path(table_file).expanduser().resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            table_items.append(
                {
                    "table_file": resolved,
                    "table_id": str(item.get("table_id", "") or ""),
                }
            )
            if limit is not None and len(table_items) >= limit:
                break
    return table_items


def build_schema_from_table(table_file: str, table_id: str = "") -> dict:
    from actions.cell_retrieval import read_table

    df = read_table(table_file)
    return {
        "file_path": table_file,
        "table_id": table_id,
        "table_name": Path(table_file).parent.name or Path(table_file).stem,
        "column_list": [str(col) for col in df.columns.tolist()],
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset JSONL does not exist: {dataset_path}")
    if args.cell_retrieval_method != "bm25":
        raise NotImplementedError(
            f"cell_retrieval_method={args.cell_retrieval_method!r} is reserved but not implemented. "
            "Please use 'bm25' for now."
        )

    from actions.cell_retrieval import (
        build_or_load_cell_index,
        get_cell_index_cache_path,
        load_cell_index_cache,
    )

    table_items = read_dataset_tables(dataset_path, args.limit)
    print(f"Found {len(table_items)} unique tables for task={args.task}")

    built = 0
    skipped = 0
    failed = 0
    for idx, table_item in enumerate(table_items, start=1):
        table_file = table_item["table_file"]
        table_id = table_item.get("table_id", "")
        try:
            cache_path = get_cell_index_cache_path(
                table_file=table_file,
                task=args.task,
                cache_dir=args.cache_dir,
                max_row_context_cols=args.max_row_context_cols,
                max_cell_text_chars=args.max_cell_text_chars,
                cell_retrieval_method=args.cell_retrieval_method,
                table_id=table_id,
            )

            if not args.overwrite_cache:
                cached_items = load_cell_index_cache(
                    table_file=table_file,
                    task=args.task,
                    cache_dir=args.cache_dir,
                    max_row_context_cols=args.max_row_context_cols,
                    max_cell_text_chars=args.max_cell_text_chars,
                    cell_retrieval_method=args.cell_retrieval_method,
                    table_id=table_id,
                )
                if cached_items is not None:
                    skipped += 1
                    print(f"[{idx}/{len(table_items)}] skip/load: {len(cached_items)} cells -> {cache_path}")
                    continue

            table_schema = build_schema_from_table(table_file, table_id=table_id)
            cell_items = build_or_load_cell_index(
                table_file=table_file,
                table_schema=table_schema,
                task=args.task,
                cache_dir=args.cache_dir,
                cell_retrieval_method=args.cell_retrieval_method,
                table_id=table_id,
                overwrite_cache=args.overwrite_cache,
                max_row_context_cols=args.max_row_context_cols,
                max_cell_text_chars=args.max_cell_text_chars,
            )
            built += 1
            print(f"[{idx}/{len(table_items)}] build: {len(cell_items)} cells -> {cache_path}")
        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(table_items)}] fail: {table_file} ({exc})")

    print(f"Done. Built: {built}, Skipped/Loaded: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
