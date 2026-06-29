"""
Cell-level retrieval utilities for cell-guided table zooming.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import csv
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

CELL_INDEX_VERSION = "cell_index_v1"


def read_table(table_file: str) -> pd.DataFrame:
    if table_file.endswith(".csv"):
        try:
            return pd.read_csv(table_file, encoding="utf8")
        except pd.errors.ParserError:
            return pd.read_csv(
                table_file,
                encoding="utf8",
                engine="python",
                escapechar="\\",
                quoting=csv.QUOTE_MINIMAL,
            )
    if table_file.endswith((".xlsx", ".xls")):
        return pd.read_excel(table_file)
    raise ValueError(f"Unsupported table file type: {table_file}")


def _sanitize_task(task: str) -> str:
    task = str(task or "default").strip().lower()
    task = re.sub(r"[^0-9a-zA-Z_.-]+", "_", task)
    return task or "default"


def _sanitize_cell_retrieval_method(cell_retrieval_method: str) -> str:
    method = str(cell_retrieval_method or "bm25").strip().lower()
    method = re.sub(r"[^0-9a-zA-Z_.-]+", "_", method)
    return method or "bm25"


def _safe_cache_stem(value: str) -> str:
    stem = str(value or "").strip().strip("/")
    stem = re.sub(r"[^0-9a-zA-Z_.-]+", "_", stem)
    stem = stem.strip("._")
    return stem[:120] if stem else ""


def _ensure_supported_cell_index_method(cell_retrieval_method: str) -> str:
    method = _sanitize_cell_retrieval_method(cell_retrieval_method)
    if method != "bm25":
        raise NotImplementedError(
            f"cell_retrieval_method={method!r} is reserved but not implemented for cell index cache. "
            "Please use 'bm25' for now."
        )
    return method


def _table_file_metadata(table_file: str) -> dict[str, Any]:
    stat = os.stat(table_file)
    return {
        "table_file": str(Path(table_file).expanduser().resolve()),
        "table_size": int(stat.st_size),
        "table_mtime": float(stat.st_mtime),
    }


def _cell_index_metadata(
    table_file: str,
    task: str,
    max_row_context_cols: int,
    max_cell_text_chars: int,
    cell_retrieval_method: str = "bm25",
    table_id: str | None = None,
) -> dict[str, Any]:
    metadata = _table_file_metadata(table_file)
    metadata.update(
        {
            "task": _sanitize_task(task),
            "table_id": str(table_id) if table_id else "",
            "cell_retrieval_method": _sanitize_cell_retrieval_method(cell_retrieval_method),
            "max_row_context_cols": int(max_row_context_cols),
            "max_cell_text_chars": int(max_cell_text_chars),
            "index_version": CELL_INDEX_VERSION,
        }
    )
    return metadata


def get_cell_index_cache_path(
    table_file: str,
    task: str = "default",
    cache_dir: str = "cache/cell_index",
    max_row_context_cols: int = 4,
    max_cell_text_chars: int = 180,
    cell_retrieval_method: str = "bm25",
    table_id: str | None = None,
) -> Path:
    resolved_table_file = str(Path(table_file).expanduser().resolve())
    table_key = _safe_cache_stem(table_id)
    if not table_key:
        table_key = hashlib.sha1(resolved_table_file.encode("utf-8")).hexdigest()[:16]
    filename = f"{table_key}_ctx{max_row_context_cols}_chars{max_cell_text_chars}.json"
    return (
        Path(cache_dir).expanduser()
        / _sanitize_task(task)
        / _sanitize_cell_retrieval_method(cell_retrieval_method)
        / filename
    )


def _is_valid_cell_index_cache(
    cache_data: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> bool:
    metadata = cache_data.get("metadata")
    cell_items = cache_data.get("cell_items")
    if not isinstance(metadata, dict) or not isinstance(cell_items, list):
        return False
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            return False
    return True


def load_cell_index_cache(
    table_file: str,
    task: str = "default",
    cache_dir: str = "cache/cell_index",
    max_row_context_cols: int = 4,
    max_cell_text_chars: int = 180,
    cell_retrieval_method: str = "bm25",
    table_id: str | None = None,
) -> list[dict[str, Any]] | None:
    cache_path = get_cell_index_cache_path(
        table_file=table_file,
        task=task,
        cache_dir=cache_dir,
        max_row_context_cols=max_row_context_cols,
        max_cell_text_chars=max_cell_text_chars,
        cell_retrieval_method=cell_retrieval_method,
        table_id=table_id,
    )
    if not cache_path.exists():
        return None

    expected_metadata = _cell_index_metadata(
        table_file=table_file,
        task=task,
        max_row_context_cols=max_row_context_cols,
        max_cell_text_chars=max_cell_text_chars,
        cell_retrieval_method=cell_retrieval_method,
        table_id=table_id,
    )
    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not _is_valid_cell_index_cache(cache_data, expected_metadata):
        return None
    return cache_data["cell_items"]


def save_cell_index_cache(
    table_file: str,
    cell_items: list[dict[str, Any]],
    task: str = "default",
    cache_dir: str = "cache/cell_index",
    max_row_context_cols: int = 4,
    max_cell_text_chars: int = 180,
    cell_retrieval_method: str = "bm25",
    table_id: str | None = None,
) -> Path:
    cache_path = get_cell_index_cache_path(
        table_file=table_file,
        task=task,
        cache_dir=cache_dir,
        max_row_context_cols=max_row_context_cols,
        max_cell_text_chars=max_cell_text_chars,
        cell_retrieval_method=cell_retrieval_method,
        table_id=table_id,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "metadata": _cell_index_metadata(
            table_file=table_file,
            task=task,
            max_row_context_cols=max_row_context_cols,
            max_cell_text_chars=max_cell_text_chars,
            cell_retrieval_method=cell_retrieval_method,
            table_id=table_id,
        ),
        "cell_items": cell_items,
    }
    cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
    return cache_path


def get_table_title(table_schema: dict[str, Any], table_file: str | None = None) -> str:
    title = table_schema.get("table_title") or table_schema.get("table_name") or ""
    if title:
        return str(title)
    if table_file:
        return Path(table_file).stem
    file_path = table_schema.get("file_path")
    return Path(file_path).stem if file_path else ""


def table_preview_records(table_file: str, max_rows: int = 2) -> list[dict[str, Any]]:
    df = read_table(table_file)
    return df.head(max_rows).to_dict(orient="records")


def _is_empty_cell(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def _safe_text(value: Any) -> str:
    if _is_empty_cell(value):
        return ""
    return str(value).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def build_cell_index(
    table_file: str,
    table_schema: dict[str, Any],
    max_row_context_cols: int = 4,
    max_cell_text_chars: int = 180,
) -> list[dict[str, Any]]:
    """Read the table and convert each non-empty cell into a retrieval item."""
    df = read_table(table_file)
    table_title = get_table_title(table_schema, table_file)
    cell_items: list[dict[str, Any]] = []

    for row_id, row in df.iterrows():
        row_pairs = []
        for col in df.columns:
            text_value = _safe_text(row[col])
            if text_value:
                row_pairs.append((str(col), text_value))

        for col_id, col in enumerate(df.columns):
            cell_value = _safe_text(row[col])
            if not cell_value:
                continue

            context_pairs = [(str(col), cell_value)]
            for ctx_col, ctx_value in row_pairs:
                if ctx_col == str(col):
                    continue
                context_pairs.append((ctx_col, ctx_value))
                if len(context_pairs) >= max_row_context_cols:
                    break

            row_context = "; ".join(
                f"{ctx_col}={ctx_value}" for ctx_col, ctx_value in context_pairs
            )
            text_parts = []
            if table_title:
                text_parts.append(f"Table title: {table_title}.")
            text_parts.extend(
                [
                    f"Column: {col}.",
                    f"Cell value: {cell_value}.",
                    f"Row context: {row_context}.",
                ]
            )
            item_text = _truncate_text(" ".join(text_parts), max_cell_text_chars)

            cell_items.append(
                {
                    "cell_id": f"r{int(row_id)}::c{int(col_id)}",
                    "row_id": int(row_id),
                    "col_id": int(col_id),
                    "col_name": str(col),
                    "cell_value": cell_value,
                    "row_context": row_context,
                    "text": item_text,
                }
            )

    return cell_items


def build_or_load_cell_index(
    table_file: str,
    table_schema: dict[str, Any] | None = None,
    task: str = "default",
    cache_dir: str = "cache/cell_index",
    cell_retrieval_method: str = "bm25",
    table_id: str | None = None,
    overwrite_cache: bool = False,
    max_row_context_cols: int = 4,
    max_cell_text_chars: int = 180,
) -> list[dict[str, Any]]:
    cell_retrieval_method = _ensure_supported_cell_index_method(cell_retrieval_method)
    if table_id is None and table_schema is not None:
        table_id = table_schema.get("table_id")
    if not overwrite_cache:
        cached_items = load_cell_index_cache(
            table_file=table_file,
            task=task,
            cache_dir=cache_dir,
            max_row_context_cols=max_row_context_cols,
            max_cell_text_chars=max_cell_text_chars,
            cell_retrieval_method=cell_retrieval_method,
            table_id=table_id,
        )
        if cached_items is not None:
            return cached_items

    if table_schema is None:
        df = read_table(table_file)
        table_schema = {
            "file_path": table_file,
            "table_id": table_id or "",
            "table_name": os.path.basename(os.path.dirname(table_file)) or Path(table_file).stem,
            "column_list": [str(col) for col in df.columns.tolist()],
        }

    cell_items = build_cell_index(
        table_file=table_file,
        table_schema=table_schema,
        max_row_context_cols=max_row_context_cols,
        max_cell_text_chars=max_cell_text_chars,
    )
    save_cell_index_cache(
        table_file=table_file,
        cell_items=cell_items,
        task=task,
        cache_dir=cache_dir,
        max_row_context_cols=max_row_context_cols,
        max_cell_text_chars=max_cell_text_chars,
        cell_retrieval_method=cell_retrieval_method,
        table_id=table_id,
    )
    return cell_items


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def bm25_retrieve_cells(
    cell_items: list[dict[str, Any]],
    query: str,
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Retrieve cells for one query with a small dependency-free BM25 scorer."""
    if top_k <= 0 or not cell_items:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_tokens = [tokenize(item.get("text", "")) for item in cell_items]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avgdl == 0:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    query_counts = Counter(query_terms)
    scores: list[tuple[float, int]] = []
    total_docs = len(cell_items)

    for idx, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        score = 0.0
        doc_len = doc_lengths[idx]
        for term, query_tf in query_counts.items():
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += query_tf * idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scores.append((score, idx))

    scores.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for score, idx in scores[:top_k]:
        result = dict(cell_items[idx])
        result["score"] = round(float(score), 6)
        result["source_query"] = query
        results.append(result)
    return results


def retrieve_cells_by_queries(
    cell_items: list[dict[str, Any]],
    queries: list[str],
    top_k: int = 20,
    method: str = "bm25",
) -> dict[str, list[dict[str, Any]]]:
    method = method.lower()
    if method != "bm25":
        raise NotImplementedError(
            f"cell_retrieval_method={method!r} is reserved but not implemented. "
            "Please use 'bm25' for now."
        )

    per_query_results = {}
    for query in queries:
        if not str(query).strip():
            continue
        per_query_results[query] = bm25_retrieve_cells(cell_items, query, top_k=top_k)
    return per_query_results


def merge_cell_results(
    per_query_results: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for query, cells in per_query_results.items():
        for cell in cells:
            key = cell.get("cell_id") or f"r{int(cell['row_id'])}::c{cell.get('col_id', cell.get('col_name'))}"
            if key not in merged or cell.get("score", 0.0) > merged[key].get("score", 0.0):
                merged[key] = dict(cell)
                merged[key]["source_queries"] = [query]
            elif cell.get("score", 0.0) == merged[key].get("score", 0.0):
                merged[key].setdefault("source_queries", [])
                if query not in merged[key]["source_queries"]:
                    merged[key]["source_queries"].append(query)

    return sorted(
        merged.values(),
        key=lambda cell: (-cell.get("score", 0.0), cell.get("row_id", 0), cell.get("col_name", "")),
    )


def normalize_columns(candidate_cols: list[str], all_columns: list[str]) -> list[str]:
    normalized = []
    lower_map = {str(col).lower(): str(col) for col in all_columns}

    for candidate in candidate_cols or []:
        if not candidate:
            continue
        candidate = str(candidate).strip()
        match = None
        if candidate in all_columns:
            match = candidate
        elif candidate.lower() in lower_map:
            match = lower_map[candidate.lower()]
        else:
            candidate_lower = candidate.lower()
            for col in all_columns:
                col_lower = str(col).lower()
                if candidate_lower in col_lower or col_lower in candidate_lower:
                    match = str(col)
                    break
            if match is None:
                close = get_close_matches(candidate, [str(col) for col in all_columns], n=1, cutoff=0.75)
                if close:
                    match = close[0]

        if match and match not in normalized:
            normalized.append(match)

    return normalized


def deterministic_expand_cells_to_subtable(
    top_k_cells: list[dict[str, Any]],
    rewrite_profile: dict[str, Any],
    all_columns: list[str],
    top_k_rows: int = 10,
    top_k_cols: int = 10,
) -> tuple[list[int], list[str]]:
    expansion = deterministic_expand_cells_to_subtable_with_meta(
        top_k_cells=top_k_cells,
        rewrite_profile=rewrite_profile,
        all_columns=all_columns,
        top_k_rows=top_k_rows,
        top_k_cols=top_k_cols,
    )
    return expansion["selected_rows"], expansion["selected_columns"]


def deterministic_expand_cells_to_subtable_with_meta(
    top_k_cells: list[dict[str, Any]],
    rewrite_profile: dict[str, Any],
    all_columns: list[str],
    top_k_rows: int = 10,
    top_k_cols: int = 10,
) -> dict[str, Any]:
    target_cols = normalize_columns(rewrite_profile.get("target_columns", []), all_columns)
    constraint_cols = normalize_columns(rewrite_profile.get("constraint_columns", []), all_columns)

    row_best_score: dict[int, float] = {}
    for cell in top_k_cells:
        row_id = int(cell["row_id"])
        score = float(cell.get("score", 0.0))
        row_best_score[row_id] = max(row_best_score.get(row_id, 0.0), score)

    selected_rows_ranked = sorted(
        row_best_score.keys(),
        key=lambda row_id: (-row_best_score[row_id], row_id),
    )[:top_k_rows]
    selected_rows = sorted(selected_rows_ranked)

    col_hit_count: defaultdict[str, int] = defaultdict(int)
    col_best_score: defaultdict[str, float] = defaultdict(float)
    col_id_map: dict[str, int] = {str(col): idx for idx, col in enumerate(all_columns)}
    for cell in top_k_cells:
        col = str(cell["col_name"])
        score = float(cell.get("score", 0.0))
        col_hit_count[col] += 1
        col_best_score[col] = max(col_best_score[col], score)
        if col not in col_id_map and "col_id" in cell:
            col_id_map[col] = int(cell["col_id"])

    hit_cols = sorted(
        col_hit_count.keys(),
        key=lambda col: (-col_hit_count[col], -col_best_score[col], col_id_map.get(col, 10**9)),
    )

    selected_columns = []
    for col in target_cols + constraint_cols + hit_cols:
        if col in all_columns and col not in selected_columns:
            selected_columns.append(col)

    row_fallback = False
    col_fallback = False
    if not selected_rows:
        row_fallback = True
    selected_columns = selected_columns[:top_k_cols]
    if not selected_columns:
        col_fallback = True
        selected_columns = [str(col) for col in all_columns[:top_k_cols]]

    stats_cols = set(target_cols + constraint_cols + hit_cols + selected_columns)
    column_stats = {}
    for col in all_columns:
        if col not in stats_cols:
            continue
        if col not in all_columns:
            continue
        column_stats[col] = {
            "hit_count": int(col_hit_count.get(col, 0)),
            "best_score": round(float(col_best_score.get(col, 0.0)), 6),
            "col_id": int(col_id_map.get(col, all_columns.index(col))),
            "is_target_column": col in target_cols,
            "is_constraint_column": col in constraint_cols,
            "selected": col in selected_columns,
        }

    return {
        "selected_rows": selected_rows,
        "selected_rows_ranked": selected_rows_ranked,
        "selected_columns": selected_columns,
        "row_scores": {str(row_id): round(float(score), 6) for row_id, score in sorted(row_best_score.items())},
        "column_stats": column_stats,
        "target_columns_normalized": target_cols,
        "constraint_columns_normalized": constraint_cols,
        "hit_columns_ranked": hit_cols,
        "expansion_fallback": {
            "row_fallback": row_fallback,
            "column_fallback": col_fallback,
        },
    }


def build_cell_guided_table_zoom(
    table_file: str,
    table_schema: dict[str, Any],
    selected_rows: list[int],
    selected_columns: list[str],
    top_k_cells: list[dict[str, Any]] | None = None,
    fallback_rows: int = 10,
    expansion_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    df = read_table(table_file)
    all_columns = [str(col) for col in df.columns.tolist()]
    selected_columns = [col for col in selected_columns if col in all_columns]
    column_fallback = False
    if not selected_columns:
        column_fallback = True
        selected_columns = all_columns[:10]

    valid_rows = [row for row in selected_rows if 0 <= row < len(df)]
    row_fallback = False
    if not valid_rows:
        row_fallback = True
        valid_rows = list(range(min(fallback_rows, len(df))))

    selected_df = df.iloc[valid_rows][selected_columns].copy()
    table_zoom = {
        "header": selected_columns,
        "rows": selected_df.values.tolist(),
    }

    refined_table_schema = {
        "file_path": table_schema.get("file_path", table_file),
        "table_id": table_schema.get("table_id", ""),
        "table_title": get_table_title(table_schema, table_file),
        "table_name": table_schema.get("table_name", os.path.basename(os.path.dirname(table_file))),
        "table_description": table_schema.get("table_description", table_schema.get("description", "")),
        "number_of_rows": len(valid_rows),
        "column_list": selected_columns,
        "cell_example": table_zoom,
        "column_description": [
            col_desc for col_desc in table_schema.get("column_description", [])
            if col_desc.get("column_name") in selected_columns
        ],
        "table_zoom": table_zoom,
        "evidence_cells": (top_k_cells or [])[:20],
        "selected_rows": valid_rows,
        "selected_columns": selected_columns,
        "selected_rows_ranked": (expansion_metadata or {}).get("selected_rows_ranked", valid_rows),
        "row_scores": (expansion_metadata or {}).get("row_scores", {}),
        "column_stats": (expansion_metadata or {}).get("column_stats", {}),
        "hit_columns_ranked": (expansion_metadata or {}).get("hit_columns_ranked", []),
        "subtable_shape": [len(valid_rows), len(selected_columns)],
        "expansion_fallback": {
            "row_fallback": row_fallback or (expansion_metadata or {}).get("expansion_fallback", {}).get("row_fallback", False),
            "column_fallback": column_fallback or (expansion_metadata or {}).get("expansion_fallback", {}).get("column_fallback", False),
        },
    }
    return refined_table_schema, table_zoom


def compute_compression_ratio(
    total_rows: int,
    total_cols: int,
    selected_rows: list[int],
    selected_columns: list[str],
) -> float:
    total_cells = max(total_rows * total_cols, 1)
    selected_cells = len(selected_rows) * len(selected_columns)
    return round(selected_cells / total_cells, 6)
