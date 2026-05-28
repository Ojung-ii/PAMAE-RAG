from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids

PATH_CARRIER_RENDERER_MODES = {
    "metric_path_carrier",
    "metric_path_carrier_no_medoids",
    "metric_path_carrier_medoids_first",
}


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


def _answer_in_context(row: dict[str, Any]) -> bool:
    diagnostics = _diagnostics(row)
    stage = diagnostics.get("stage_diagnostics")
    if isinstance(stage, dict):
        context = stage.get("context_rendering")
        if isinstance(context, dict):
            extra = context.get("extra")
            if isinstance(extra, dict) and extra.get("answer_in_context") is not None:
                return bool(extra.get("answer_in_context"))
    path = diagnostics.get("path_realizability")
    if isinstance(path, dict):
        trace = path.get("answer_trace")
        if isinstance(trace, dict):
            return bool(trace.get("answer_chunk_rendered", False))
    return False


def _stage_sets(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
) -> dict[str, set[str] | dict[str, int] | bool]:
    diagnostics = _diagnostics(retrieval_row)
    active_ids = set(_ids(diagnostics.get("active_universe_node_ids", [])))
    if not active_ids:
        active_ids = {str(node.node_id) for node in example.nodes}
    projected_ids = set(_ids(diagnostics.get("projected_node_ids", []))) or active_ids
    current_context = set(_ids(retrieval_row.get("context_node_ids", [])))
    renderer_order = _ids(diagnostics.get("path_carrier_order_node_ids", diagnostics.get("renderer_budget_order_node_ids", [])))
    budget_cutoff = set(_ids(diagnostics.get("budget_cutoff_node_ids", [])))
    return {
        "active": active_ids,
        "candidate": set(_ids(diagnostics.get("candidate_node_ids", []))) or active_ids,
        "projected": projected_ids,
        "phase1_medoids": set(_ids(diagnostics.get("pre_refinement_anchor_ids", []))),
        "refined_medoids": set(_ids(retrieval_row.get("anchor_node_ids", []))),
        "selected_basin": set(_ids(diagnostics.get("diagnostic_selected_basin_node_ids", []))),
        "phase1_tree": set(_ids(diagnostics.get("phase1_support_tree_node_ids", []))),
        "refined_tree": set(_ids(diagnostics.get("refined_support_tree_node_ids", []))),
        "anchor_medoid_path": set(_ids(diagnostics.get("refined_anchor_medoid_path_node_ids", []))),
        "medoid_medoid_path": set(_ids(diagnostics.get("refined_medoid_medoid_path_node_ids", []))),
        "current_context": current_context,
        "metric_context": current_context,
        "budget_cutoff": budget_cutoff,
        "rank": {node_id: pos for pos, node_id in enumerate(renderer_order, start=1)},
    }


def _carrier_role(row: dict[str, Any]) -> str | None:
    if bool(row.get("answer_chunk_is_refined_medoid")):
        return "medoid"
    if bool(row.get("answer_chunk_on_anchor_medoid_path")):
        return "anchor_medoid_path"
    if bool(row.get("answer_chunk_on_medoid_medoid_path")):
        return "medoid_medoid_path"
    if bool(row.get("answer_chunk_dropped_by_budget")):
        return "budget_cutoff"
    if bool(row.get("answer_chunk_current_rendered")) and not bool(row.get("answer_chunk_metric_path_rendered")):
        return "current_only_hidden_recovery"
    return None


def support_tree_carrier_trace_rows(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
    renderer_mode: str,
    qa_f1: float = 0.0,
) -> list[dict[str, Any]]:
    stage = _stage_sets(example=example, retrieval_row=retrieval_row)
    answer_ids = answer_containing_chunk_ids(example, example.nodes)
    current_answer = _answer_in_context(retrieval_row) if renderer_mode == "current_renderer" else False
    metric_answer = _answer_in_context(retrieval_row) if renderer_mode in PATH_CARRIER_RENDERER_MODES else False
    rows: list[dict[str, Any]] = []
    if not answer_ids:
        answer_ids = [None]  # type: ignore[list-item]
    for chunk_id in answer_ids:
        chunk = str(chunk_id) if chunk_id is not None else None
        on_refined_tree = bool(chunk is not None and chunk in stage["refined_tree"])
        on_phase1_tree = bool(chunk is not None and chunk in stage["phase1_tree"])
        current_rendered = bool(chunk is not None and renderer_mode == "current_renderer" and chunk in stage["current_context"])
        metric_rendered = bool(chunk is not None and renderer_mode in PATH_CARRIER_RENDERER_MODES and chunk in stage["metric_context"])
        row = {
            "query_id": example.query_id,
            "dataset": _dataset_name(example),
            "answer": example.answer or "",
            "renderer_mode": renderer_mode,
            "answer_chunk_id": chunk,
            "answer_chunk_in_candidate": bool(chunk is not None and chunk in stage["candidate"]),
            "answer_chunk_in_projected": bool(chunk is not None and chunk in stage["projected"]),
            "answer_chunk_in_active_universe": bool(chunk is not None and chunk in stage["active"]),
            "answer_chunk_is_phase1_medoid": bool(chunk is not None and chunk in stage["phase1_medoids"]),
            "answer_chunk_is_refined_medoid": bool(chunk is not None and chunk in stage["refined_medoids"]),
            "answer_chunk_in_selected_basin": bool(chunk is not None and chunk in stage["selected_basin"]),
            "answer_chunk_on_phase1_support_tree": on_phase1_tree,
            "answer_chunk_on_refined_support_tree": on_refined_tree,
            "answer_chunk_on_anchor_medoid_path": bool(chunk is not None and chunk in stage["anchor_medoid_path"]),
            "answer_chunk_on_medoid_medoid_path": bool(chunk is not None and chunk in stage["medoid_medoid_path"]),
            "answer_chunk_current_rendered": current_rendered,
            "answer_chunk_metric_path_rendered": metric_rendered,
            "answer_chunk_dropped_by_budget": bool(chunk is not None and chunk in stage["budget_cutoff"]),
            "answer_chunk_rank_before_budget": stage["rank"].get(chunk) if chunk is not None else None,
            "d_answer_to_support_tree": 0.0 if on_refined_tree else None,
            "nearest_support_tree_node": chunk if on_refined_tree else None,
            "current_answer_in_context": current_answer,
            "metric_path_answer_in_context": metric_answer,
            "qa_f1": float(qa_f1),
        }
        row["answer_carrier_role"] = _carrier_role(row)
        rows.append(row)
    return rows


def _group_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id"))].append(row)
    return grouped


def _query_rate(grouped: dict[str, list[dict[str, Any]]], key: str) -> float:
    if not grouped:
        return 0.0
    return float(sum(1 for rows in grouped.values() if any(bool(row.get(key)) for row in rows)) / len(grouped))


def aggregate_support_tree_carrier_traces(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_query(rows)
    current_answer = _query_rate(grouped, "current_answer_in_context")
    refined_tree = _query_rate(grouped, "answer_chunk_on_refined_support_tree")
    metric_answer = _query_rate(grouped, "metric_path_answer_in_context")
    distances = [
        float(row["d_answer_to_support_tree"])
        for row in rows
        if isinstance(row.get("d_answer_to_support_tree"), (int, float)) and row.get("answer_chunk_id") is not None
    ]
    roles = Counter(
        str(row.get("answer_carrier_role"))
        for query_rows in grouped.values()
        for row in query_rows
        if row.get("answer_carrier_role")
    )
    role_keys = (
        "medoid",
        "anchor_medoid_path",
        "medoid_medoid_path",
        "current_only_hidden_recovery",
        "budget_cutoff",
    )
    return {
        "num_queries": len(grouped),
        "answer_on_phase1_support_tree_rate": _query_rate(grouped, "answer_chunk_on_phase1_support_tree"),
        "answer_on_refined_support_tree_rate": refined_tree,
        "answer_current_rendered_rate": _query_rate(grouped, "answer_chunk_current_rendered"),
        "answer_metric_path_rendered_rate": _query_rate(grouped, "answer_chunk_metric_path_rendered"),
        "current_answer_in_context": current_answer,
        "metric_path_answer_in_context": metric_answer,
        "current_minus_refined_tree_gap": current_answer - refined_tree,
        "current_minus_metric_path_gap": current_answer - metric_answer,
        "mean_d_answer_to_support_tree": float(sum(distances) / len(distances)) if distances else 0.0,
        "answer_path_role_distribution": {key: int(roles.get(key, 0)) for key in role_keys},
    }


__all__ = [
    "aggregate_support_tree_carrier_traces",
    "support_tree_carrier_trace_rows",
]
