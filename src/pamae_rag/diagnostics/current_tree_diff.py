from __future__ import annotations

from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _chunk_ids(nodes: Sequence[EvidenceNode], values: Iterable[Any]) -> set[str]:
    chunks = {str(node.node_id) for node in nodes if str(getattr(node, "node_type", "chunk")) == "chunk"}
    return {node_id for node_id in _ids(values) if node_id in chunks}


def current_tree_diff_row(
    *,
    example: QueryExample,
    current_row: dict[str, Any],
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    current_diag = _diagnostics(current_row)
    current_chunks = _chunk_ids(example.nodes, current_row.get("context_node_ids", []))
    tree_chunks = _chunk_ids(example.nodes, current_diag.get("refined_support_tree_node_ids", []))
    metric_chunks = _chunk_ids(example.nodes, metric_row.get("context_node_ids", []))

    current_tree_intersection = current_chunks & tree_chunks
    current_only = current_chunks - tree_chunks
    tree_only = tree_chunks - current_chunks
    metric_missed_current = current_chunks - metric_chunks
    metric_extra = metric_chunks - current_chunks

    answer_ids = set(answer_containing_chunk_ids(example, example.nodes))
    gold_ids = {str(node_id) for node_id in example.gold_node_ids}

    def any_in(values: set[str], targets: set[str]) -> bool:
        return bool(values & targets)

    return {
        "query_id": example.query_id,
        "current_chunk_count": len(current_chunks),
        "support_tree_chunk_count": len(tree_chunks),
        "metric_chunk_count": len(metric_chunks),
        "current_tree_intersection_count": len(current_tree_intersection),
        "current_only_count": len(current_only),
        "tree_only_count": len(tree_only),
        "metric_missed_current_count": len(metric_missed_current),
        "metric_extra_count": len(metric_extra),
        "current_tree_intersection_chunk_ids": sorted(current_tree_intersection),
        "current_only_chunk_ids": sorted(current_only),
        "tree_only_chunk_ids": sorted(tree_only),
        "metric_missed_current_chunk_ids": sorted(metric_missed_current),
        "metric_extra_chunk_ids": sorted(metric_extra),
        "answer_in_current_tree_intersection": any_in(current_tree_intersection, answer_ids),
        "answer_in_current_only": any_in(current_only, answer_ids),
        "answer_in_tree_only": any_in(tree_only, answer_ids),
        "answer_in_metric_missed_current": any_in(metric_missed_current, answer_ids),
        "answer_in_metric_extra": any_in(metric_extra, answer_ids),
        "gold_in_current_tree_intersection": any_in(current_tree_intersection, gold_ids),
        "gold_in_current_only": any_in(current_only, gold_ids),
        "gold_in_tree_only": any_in(tree_only, gold_ids),
    }


def aggregate_current_tree_diff(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    denom = max(len(rows), 1)

    def rate(key: str) -> float:
        return float(sum(1 for row in rows if bool(row.get(key))) / denom)

    return {
        "num_queries": len(rows),
        "answer_in_current_tree_intersection_rate": rate("answer_in_current_tree_intersection"),
        "answer_in_current_only_rate": rate("answer_in_current_only"),
        "answer_in_tree_only_rate": rate("answer_in_tree_only"),
        "answer_in_metric_missed_current_rate": rate("answer_in_metric_missed_current"),
        "answer_in_metric_extra_rate": rate("answer_in_metric_extra"),
        "gold_in_current_tree_intersection_rate": rate("gold_in_current_tree_intersection"),
        "gold_in_current_only_rate": rate("gold_in_current_only"),
        "gold_in_tree_only_rate": rate("gold_in_tree_only"),
    }


__all__ = ["aggregate_current_tree_diff", "current_tree_diff_row"]
