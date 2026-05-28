from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


PATH_CARRIER_FAILURES = (
    "A_answer_not_projected",
    "B_answer_not_on_support_tree",
    "C_answer_on_tree_not_rendered",
    "D_answer_rendered_by_path_carrier",
    "E_current_only_hidden_recovery",
    "F_metric_path_adds_answer",
    "G_answer_rendered_qa_fail",
    "H_success",
)


def classify_path_carrier_failure(rows: Sequence[dict[str, Any]]) -> str:
    real_rows = [row for row in rows if row.get("answer_chunk_id") is not None]
    if not real_rows or not any(bool(row.get("answer_chunk_in_projected")) for row in real_rows):
        return "A_answer_not_projected"
    if not any(bool(row.get("answer_chunk_on_refined_support_tree")) for row in real_rows):
        return "B_answer_not_on_support_tree"

    current_rendered = any(bool(row.get("answer_chunk_current_rendered")) for row in real_rows)
    metric_rendered = any(bool(row.get("answer_chunk_metric_path_rendered")) for row in real_rows)
    if current_rendered and not metric_rendered:
        return "E_current_only_hidden_recovery"
    if metric_rendered and not current_rendered:
        return "F_metric_path_adds_answer"
    if any(bool(row.get("answer_chunk_dropped_by_budget")) for row in real_rows) and not metric_rendered:
        return "C_answer_on_tree_not_rendered"
    if not metric_rendered:
        return "C_answer_on_tree_not_rendered"

    max_f1 = max(float(row.get("qa_f1", 0.0) or 0.0) for row in rows)
    if max_f1 > 0.0:
        return "H_success"
    return "G_answer_rendered_qa_fail"


def aggregate_path_carrier_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id"))].append(row)
    counts = Counter(classify_path_carrier_failure(query_rows) for query_rows in grouped.values())
    return {
        "path_carrier_failure_counts": {
            key: int(counts.get(key, 0)) for key in PATH_CARRIER_FAILURES
        }
    }


__all__ = [
    "PATH_CARRIER_FAILURES",
    "aggregate_path_carrier_taxonomy",
    "classify_path_carrier_failure",
]
