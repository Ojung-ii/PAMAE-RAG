from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


SUPPORT_TREE_ORDER_FAILURES = (
    "A_answer_not_projected",
    "B_answer_not_on_tree",
    "C_answer_on_tree_cut_by_budget",
    "D_answer_on_tree_bad_order",
    "E_current_hidden_non_tree",
    "F_tree_contains_answer_current_misses",
    "G_answer_rendered_qa_fail",
    "H_success",
)


def classify_support_tree_order_failure(rows: Sequence[dict[str, Any]]) -> str:
    real_rows = [row for row in rows if row.get("answer_chunk_id") is not None]
    if not real_rows or not any(bool(row.get("answer_chunk_in_projected")) for row in real_rows):
        return "A_answer_not_projected"
    if any(
        bool(row.get("current_renderer_rendered")) and not bool(row.get("answer_chunk_on_support_tree"))
        for row in real_rows
    ):
        return "E_current_hidden_non_tree"
    if not any(bool(row.get("answer_chunk_on_support_tree")) for row in real_rows):
        return "B_answer_not_on_tree"
    if any(bool(row.get("metric_budget_cutoff_before_answer")) for row in real_rows):
        return "C_answer_on_tree_cut_by_budget"
    if any(
        bool(row.get("answer_chunk_on_support_tree")) and not bool(row.get("metric_path_carrier_rendered"))
        for row in real_rows
    ):
        return "D_answer_on_tree_bad_order"
    if any(
        bool(row.get("answer_chunk_on_support_tree")) and not bool(row.get("current_renderer_rendered"))
        for row in real_rows
    ):
        return "F_tree_contains_answer_current_misses"
    rendered = any(
        bool(row.get("current_renderer_rendered")) or bool(row.get("metric_path_carrier_rendered"))
        for row in real_rows
    )
    qa_success = any(
        float(row.get("qa_f1_current", 0.0) or 0.0) > 0.0
        or float(row.get("qa_f1_metric", 0.0) or 0.0) > 0.0
        for row in real_rows
    )
    if rendered and not qa_success:
        return "G_answer_rendered_qa_fail"
    return "H_success"


def aggregate_support_tree_order_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id"))].append(row)
    counts = Counter(classify_support_tree_order_failure(query_rows) for query_rows in grouped.values())
    return {
        "support_tree_order_failure_counts": {
            key: int(counts.get(key, 0)) for key in SUPPORT_TREE_ORDER_FAILURES
        }
    }


__all__ = [
    "SUPPORT_TREE_ORDER_FAILURES",
    "aggregate_support_tree_order_taxonomy",
    "classify_support_tree_order_failure",
]
