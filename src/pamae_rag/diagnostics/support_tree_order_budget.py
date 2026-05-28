from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids

DistanceLookup = Callable[[str, str], float | None]


CURRENT_RENDER_ROLES = (
    "selected_medoid",
    "post_refine_medoid",
    "support_tree_bridge",
    "path_closure",
    "basin_member",
    "extra_nonmedoid",
    "fallback_or_unknown",
    "not_rendered",
)
METRIC_RENDER_ROLES = (
    "selected_medoid",
    "anchor_medoid_path",
    "medoid_medoid_path",
    "not_rendered",
)


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _dataset_name(example: QueryExample) -> str:
    value = example.metadata.get("dataset")
    if value is not None:
        return str(value)
    nested = example.metadata.get("metadata")
    if isinstance(nested, dict) and nested.get("dataset") is not None:
        return str(nested["dataset"])
    return ""


def _node_by_id(nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
    return {str(node.node_id): node for node in nodes}


def _chunk_ids(nodes: Sequence[EvidenceNode], values: Iterable[Any]) -> set[str]:
    chunks = {str(node.node_id) for node in nodes if str(getattr(node, "node_type", "chunk")) == "chunk"}
    return {node_id for node_id in _ids(values) if node_id in chunks}


def _token_spans(nodes: Sequence[EvidenceNode], context_ids: Iterable[Any]) -> dict[str, tuple[int, int, int]]:
    by_id = _node_by_id(nodes)
    spans: dict[str, tuple[int, int, int]] = {}
    cursor = 0
    for order, node_id in enumerate(_ids(context_ids), start=1):
        node = by_id.get(node_id)
        tokens = max(1, int(node.token_count)) if node is not None else 1
        spans[node_id] = (order, cursor, cursor + tokens)
        cursor += tokens
    return spans


def _sets(
    *,
    example: QueryExample,
    current_row: dict[str, Any],
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    current_diag = _diagnostics(current_row)
    metric_diag = _diagnostics(metric_row)
    active = set(_ids(current_diag.get("active_universe_node_ids", []))) or {str(node.node_id) for node in example.nodes}
    candidate = set(_ids(current_diag.get("candidate_node_ids", []))) or active
    projected = set(_ids(current_diag.get("projected_node_ids", []))) or active
    refined_medoids = set(_ids(current_row.get("anchor_node_ids", [])))
    tree = _chunk_ids(example.nodes, current_diag.get("refined_support_tree_node_ids", []))
    anchor_path = _chunk_ids(example.nodes, current_diag.get("refined_anchor_medoid_path_node_ids", []))
    medoid_path = _chunk_ids(example.nodes, current_diag.get("refined_medoid_medoid_path_node_ids", []))
    path_closure = set(_ids(current_diag.get("path_closure_node_ids", [])))
    selected_basin = set(_ids(current_diag.get("diagnostic_selected_basin_node_ids", [])))
    metric_order = _ids(metric_diag.get("path_carrier_order_node_ids", metric_diag.get("renderer_budget_order_node_ids", [])))
    if not metric_order:
        metric_order = _ids(metric_row.get("context_node_ids", []))
    return {
        "active": active,
        "candidate": candidate,
        "projected": projected,
        "refined_medoids": refined_medoids,
        "tree": tree,
        "anchor_path": anchor_path,
        "medoid_path": medoid_path,
        "path_closure": path_closure,
        "selected_basin": selected_basin,
        "current_context": _chunk_ids(example.nodes, current_row.get("context_node_ids", [])),
        "metric_context": _chunk_ids(example.nodes, metric_row.get("context_node_ids", [])),
        "current_spans": _token_spans(example.nodes, current_row.get("context_node_ids", [])),
        "metric_spans": _token_spans(example.nodes, metric_row.get("context_node_ids", [])),
        "metric_order": metric_order,
        "metric_rank": {node_id: rank for rank, node_id in enumerate(metric_order, start=1)},
        "metric_cutoff": set(_ids(metric_diag.get("budget_cutoff_node_ids", []))),
    }


def _qa_f1(qa_row: dict[str, Any] | None) -> float:
    if not qa_row:
        return 0.0
    value = qa_row.get("f1")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _answer_in_context(qa_row: dict[str, Any] | None) -> bool:
    if not qa_row:
        return False
    value = qa_row.get("answer_coverage")
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0)


def _current_role(chunk_id: str | None, stage: dict[str, Any]) -> str:
    if chunk_id is None or chunk_id not in stage["current_context"]:
        return "not_rendered"
    if chunk_id in stage["refined_medoids"]:
        return "post_refine_medoid"
    if chunk_id in stage["tree"]:
        return "support_tree_bridge"
    if chunk_id in stage["path_closure"]:
        return "path_closure"
    if chunk_id in stage["selected_basin"]:
        return "extra_nonmedoid"
    return "fallback_or_unknown"


def _metric_role(chunk_id: str | None, stage: dict[str, Any]) -> str:
    if chunk_id is None or chunk_id not in stage["metric_context"]:
        return "not_rendered"
    if chunk_id in stage["refined_medoids"]:
        return "selected_medoid"
    if chunk_id in stage["anchor_path"]:
        return "anchor_medoid_path"
    if chunk_id in stage["medoid_path"]:
        return "medoid_medoid_path"
    return "not_rendered"


def _nearest_tree(
    chunk_id: str | None,
    tree_chunks: set[str],
    distance_lookup: DistanceLookup | None,
) -> tuple[str | None, float | None]:
    if chunk_id is None or not tree_chunks:
        return None, None
    if chunk_id in tree_chunks:
        return chunk_id, 0.0
    if distance_lookup is None:
        return None, None
    candidates: list[tuple[float, str]] = []
    for tree_id in sorted(tree_chunks):
        value = distance_lookup(chunk_id, tree_id)
        if isinstance(value, (int, float)):
            candidates.append((float(value), tree_id))
    if not candidates:
        return None, None
    distance, tree_id = min(candidates, key=lambda item: (item[0], item[1]))
    return tree_id, distance


def _row_for_chunk(
    *,
    example: QueryExample,
    chunk_id: str | None,
    prefix: str,
    stage: dict[str, Any],
    current_qa: dict[str, Any] | None,
    metric_qa: dict[str, Any] | None,
    distance_lookup: DistanceLookup | None,
) -> dict[str, Any]:
    current_span = stage["current_spans"].get(chunk_id) if chunk_id is not None else None
    metric_span = stage["metric_spans"].get(chunk_id) if chunk_id is not None else None
    nearest_id, nearest_distance = _nearest_tree(chunk_id, stage["tree"], distance_lookup)
    on_tree = bool(chunk_id is not None and chunk_id in stage["tree"])
    metric_rendered = bool(chunk_id is not None and chunk_id in stage["metric_context"])
    current_rendered = bool(chunk_id is not None and chunk_id in stage["current_context"])
    metric_rank = stage["metric_rank"].get(chunk_id) if chunk_id is not None else None
    row = {
        "query_id": example.query_id,
        "dataset": _dataset_name(example),
        f"{prefix}_id": chunk_id,
        f"{prefix}_in_candidate": bool(chunk_id is not None and chunk_id in stage["candidate"]),
        f"{prefix}_in_projected": bool(chunk_id is not None and chunk_id in stage["projected"]),
        f"{prefix}_in_active_universe": bool(chunk_id is not None and chunk_id in stage["active"]),
        f"{prefix}_is_refined_medoid": bool(chunk_id is not None and chunk_id in stage["refined_medoids"]),
        f"{prefix}_on_support_tree": on_tree,
        f"{prefix}_on_anchor_medoid_path": bool(chunk_id is not None and chunk_id in stage["anchor_path"]),
        f"{prefix}_on_medoid_medoid_path": bool(chunk_id is not None and chunk_id in stage["medoid_path"]),
        f"d_{prefix}_to_support_tree": nearest_distance,
        f"nearest_support_tree_chunk_id": nearest_id,
        f"nearest_support_tree_distance": nearest_distance,
        "current_renderer_rendered": current_rendered,
        "metric_path_carrier_rendered": metric_rendered,
        "current_render_role": _current_role(chunk_id, stage),
        "metric_render_role": _metric_role(chunk_id, stage),
        "current_render_order": current_span[0] if current_span else None,
        "metric_render_order": metric_span[0] if metric_span else None,
        "current_token_start": current_span[1] if current_span else None,
        "current_token_end": current_span[2] if current_span else None,
        "metric_token_start": metric_span[1] if metric_span else None,
        "metric_token_end": metric_span[2] if metric_span else None,
        "metric_budget_cutoff_before_answer": bool(chunk_id is not None and chunk_id in stage["metric_cutoff"]),
        "metric_rank_before_budget": metric_rank,
        "metric_rank_after_budget": metric_span[0] if metric_span else None,
        "answer_in_current_context": _answer_in_context(current_qa),
        "answer_in_metric_context": _answer_in_context(metric_qa),
        "qa_f1_current": _qa_f1(current_qa),
        "qa_f1_metric": _qa_f1(metric_qa),
    }
    if prefix == "answer_chunk":
        row["answer"] = example.answer or ""
    return row


def support_tree_order_budget_rows(
    *,
    example: QueryExample,
    current_row: dict[str, Any],
    metric_row: dict[str, Any],
    current_qa: dict[str, Any] | None = None,
    metric_qa: dict[str, Any] | None = None,
    distance_lookup: DistanceLookup | None = None,
) -> list[dict[str, Any]]:
    stage = _sets(example=example, current_row=current_row, metric_row=metric_row)
    answer_ids = answer_containing_chunk_ids(example, example.nodes) or [None]  # type: ignore[list-item]
    return [
        _row_for_chunk(
            example=example,
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            prefix="answer_chunk",
            stage=stage,
            current_qa=current_qa,
            metric_qa=metric_qa,
            distance_lookup=distance_lookup,
        )
        for chunk_id in answer_ids
    ]


def gold_support_tree_order_budget_rows(
    *,
    example: QueryExample,
    current_row: dict[str, Any],
    metric_row: dict[str, Any],
    current_qa: dict[str, Any] | None = None,
    metric_qa: dict[str, Any] | None = None,
    distance_lookup: DistanceLookup | None = None,
) -> list[dict[str, Any]]:
    stage = _sets(example=example, current_row=current_row, metric_row=metric_row)
    return [
        _row_for_chunk(
            example=example,
            chunk_id=str(chunk_id),
            prefix="gold_chunk",
            stage=stage,
            current_qa=current_qa,
            metric_qa=metric_qa,
            distance_lookup=distance_lookup,
        )
        for chunk_id in sorted(str(value) for value in example.gold_node_ids)
    ]


def _group_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id"))].append(row)
    return grouped


def _query_rate(grouped: dict[str, list[dict[str, Any]]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    if not grouped:
        return 0.0
    return float(sum(1 for query_rows in grouped.values() if any(predicate(row) for row in query_rows)) / len(grouped))


def aggregate_support_tree_order_budget(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_query(rows)
    current_rate = _query_rate(grouped, lambda row: bool(row.get("current_renderer_rendered")))
    metric_rate = _query_rate(grouped, lambda row: bool(row.get("metric_path_carrier_rendered")))
    current_orders = [
        float(row["current_render_order"])
        for row in rows
        if isinstance(row.get("current_render_order"), (int, float))
    ]
    metric_orders = [
        float(row["metric_render_order"])
        for row in rows
        if isinstance(row.get("metric_render_order"), (int, float))
    ]
    return {
        "num_queries": len(grouped),
        "answer_on_support_tree_rate": _query_rate(grouped, lambda row: bool(row.get("answer_chunk_on_support_tree"))),
        "answer_on_anchor_medoid_path_rate": _query_rate(
            grouped, lambda row: bool(row.get("answer_chunk_on_anchor_medoid_path"))
        ),
        "answer_on_medoid_medoid_path_rate": _query_rate(
            grouped, lambda row: bool(row.get("answer_chunk_on_medoid_medoid_path"))
        ),
        "answer_current_rendered_rate": current_rate,
        "answer_metric_rendered_rate": metric_rate,
        "answer_on_tree_but_metric_not_rendered_rate": _query_rate(
            grouped,
            lambda row: bool(row.get("answer_chunk_on_support_tree"))
            and not bool(row.get("metric_path_carrier_rendered")),
        ),
        "answer_metric_budget_cutoff_rate": _query_rate(
            grouped, lambda row: bool(row.get("metric_budget_cutoff_before_answer"))
        ),
        "answer_current_only_non_tree_rate": _query_rate(
            grouped,
            lambda row: bool(row.get("current_renderer_rendered"))
            and not bool(row.get("answer_chunk_on_support_tree")),
        ),
        "answer_near_tree_distance_1_rate": _query_rate(
            grouped,
            lambda row: isinstance(row.get("nearest_support_tree_distance"), (int, float))
            and float(row["nearest_support_tree_distance"]) <= 1.0,
        ),
        "answer_near_tree_distance_2_rate": _query_rate(
            grouped,
            lambda row: isinstance(row.get("nearest_support_tree_distance"), (int, float))
            and float(row["nearest_support_tree_distance"]) <= 2.0,
        ),
        "current_minus_metric_answer_gap": current_rate - metric_rate,
        "mean_current_answer_render_order": float(sum(current_orders) / len(current_orders)) if current_orders else 0.0,
        "mean_metric_answer_render_order": float(sum(metric_orders) / len(metric_orders)) if metric_orders else 0.0,
    }


__all__ = [
    "CURRENT_RENDER_ROLES",
    "METRIC_RENDER_ROLES",
    "aggregate_support_tree_order_budget",
    "gold_support_tree_order_budget_rows",
    "support_tree_order_budget_rows",
]
