from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids

RENDER_ROLES = (
    "selected_medoid",
    "post_refine_medoid",
    "support_tree_bridge",
    "path_closure",
    "basin_member",
    "extra_nonmedoid",
    "fallback_or_unknown",
)


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _path_realizability(row: dict[str, Any]) -> dict[str, Any]:
    value = _diagnostics(row).get("path_realizability")
    return value if isinstance(value, dict) else {}


def _node_by_id(nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
    return {str(node.node_id): node for node in nodes}


def _role_for_chunk(
    chunk_id: str,
    *,
    pre_refine_medoid_ids: set[str],
    post_refine_medoid_ids: set[str],
    support_tree_ids: set[str],
    path_closure_ids: set[str],
    selected_basin_ids: set[str],
) -> str:
    if chunk_id in pre_refine_medoid_ids:
        return "selected_medoid"
    if chunk_id in post_refine_medoid_ids:
        return "post_refine_medoid"
    if chunk_id in support_tree_ids:
        return "support_tree_bridge"
    if chunk_id in path_closure_ids:
        return "path_closure"
    if chunk_id not in selected_basin_ids:
        return "fallback_or_unknown"
    return "extra_nonmedoid"


def renderer_role_trace_rows(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = _diagnostics(retrieval_row)
    path = _path_realizability(retrieval_row)
    nodes = _node_by_id(example.nodes)
    answer_ids = set(answer_containing_chunk_ids(example, example.nodes))
    gold_ids = {str(value) for value in example.gold_node_ids}
    pre_refine_ids = set(_ids(diagnostics.get("pre_refinement_anchor_ids", [])))
    post_refine_ids = set(_ids(retrieval_row.get("anchor_node_ids", [])))
    support_tree_ids = set(_ids(path.get("support_tree_node_ids", [])))
    path_closure_ids = set(_ids(diagnostics.get("path_closure_node_ids", [])))
    selected_basin_ids = set(_ids(diagnostics.get("diagnostic_selected_basin_node_ids", [])))

    rows: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(_ids(retrieval_row.get("context_node_ids", [])), start=1):
        node = nodes.get(chunk_id)
        rows.append(
            {
                "query_id": example.query_id,
                "rendered_chunk_id": chunk_id,
                "render_role": _role_for_chunk(
                    chunk_id,
                    pre_refine_medoid_ids=pre_refine_ids,
                    post_refine_medoid_ids=post_refine_ids,
                    support_tree_ids=support_tree_ids,
                    path_closure_ids=path_closure_ids,
                    selected_basin_ids=selected_basin_ids,
                ),
                "contains_answer": chunk_id in answer_ids,
                "is_gold_support": chunk_id in gold_ids,
                "token_count": int(max(1, node.token_count)) if node is not None else 0,
                "render_order": rank,
            }
        )
    return rows


def aggregate_renderer_role_trace(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answer_counts = Counter(
        str(row.get("render_role"))
        for row in rows
        if bool(row.get("contains_answer")) and str(row.get("render_role")) in RENDER_ROLES
    )
    gold_counts = Counter(
        str(row.get("render_role"))
        for row in rows
        if bool(row.get("is_gold_support")) and str(row.get("render_role")) in RENDER_ROLES
    )
    return {
        "answer_render_role_distribution": {role: int(answer_counts.get(role, 0)) for role in RENDER_ROLES},
        "gold_render_role_distribution": {role: int(gold_counts.get(role, 0)) for role in RENDER_ROLES},
    }


__all__ = [
    "RENDER_ROLES",
    "aggregate_renderer_role_trace",
    "renderer_role_trace_rows",
]
