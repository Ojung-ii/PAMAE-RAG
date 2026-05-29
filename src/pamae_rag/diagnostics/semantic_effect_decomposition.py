from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

DECOMPOSITION_METRIC_FIELDS = (
    "answer_in_context",
    "rendered_recall",
    "context_f1",
    "qa_f1",
)

VARIANT_RENDERERS = (
    "current_renderer",
    "metric_path_carrier",
    "tree_shell1_graph_order",
    "tree_shell1_semantic_query_order",
    "tree_shell1_semantic_tree_order",
    "semantic_weighted_tree_diagnostic",
)


def _numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def row_by_renderer(rows: Sequence[dict[str, Any]], renderer_mode: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("renderer_mode") == renderer_mode), {})


def delta_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Return `left - right` for the fixed decomposition fields."""

    payload = {field: _numeric(left, field) - _numeric(right, field) for field in DECOMPOSITION_METRIC_FIELDS}
    payload["tokens"] = _numeric(left, "avg_context_tokens") - _numeric(right, "avg_context_tokens")
    return payload


def decomposition_deltas(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_renderer = {str(row.get("renderer_mode")): dict(row) for row in rows}
    a1 = by_renderer.get("metric_path_carrier", {})
    b1 = by_renderer.get("tree_shell1_graph_order", {})
    b2 = by_renderer.get("tree_shell1_semantic_query_order", {})
    b3 = by_renderer.get("tree_shell1_semantic_tree_order", {})
    return {
        "delta_shell_B1_minus_A1": delta_metrics(b1, a1),
        "delta_query_semantic_B2_minus_B1": delta_metrics(b2, b1),
        "delta_tree_semantic_B3_minus_B1": delta_metrics(b3, b1),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def query_ids_in_order(path: Path) -> list[str]:
    return [str(row.get("query_id")) for row in read_jsonl(path)]


def same_query_sample(paths: Iterable[Path]) -> dict[str, Any]:
    sequences = [query_ids_in_order(path) for path in paths if path.exists()]
    if not sequences:
        return {"same_sample": False, "sample_size": 0, "reason": "no qa rows found"}
    first = sequences[0]
    same = all(sequence == first for sequence in sequences[1:])
    return {
        "same_sample": same,
        "sample_size": len(first),
        "variant_count": len(sequences),
        "reason": "identical query IDs and order" if same else "query IDs or order differ",
    }


def prompt_protocol_status(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prompt_names = {row.get("qa_prompt_name") for row in rows}
    prompt_hashes = {row.get("qa_prompt_hash") for row in rows}
    exact_values = {row.get("qa_prompt_text_exact_match") for row in rows}
    return {
        "qa_prompt_name": next(iter(prompt_names)) if len(prompt_names) == 1 else None,
        "qa_prompt_hash": next(iter(prompt_hashes)) if len(prompt_hashes) == 1 else None,
        "qa_prompt_text_exact_match": exact_values == {True},
        "qa_prompt_consistent": len(prompt_names) == 1
        and prompt_names == {"common_qa"}
        and len(prompt_hashes) == 1
        and exact_values == {True},
    }


__all__ = [
    "DECOMPOSITION_METRIC_FIELDS",
    "VARIANT_RENDERERS",
    "decomposition_deltas",
    "delta_metrics",
    "load_json",
    "prompt_protocol_status",
    "query_ids_in_order",
    "read_jsonl",
    "row_by_renderer",
    "same_query_sample",
]
