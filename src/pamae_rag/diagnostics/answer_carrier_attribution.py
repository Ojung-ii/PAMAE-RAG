from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids
from pamae_rag.diagnostics.renderer_role_trace import aggregate_renderer_role_trace

ANSWER_CARRIER_FAILURES = (
    "A_answer_not_in_candidate",
    "B_answer_lost_at_projection",
    "C_answer_lost_at_medoid_selection",
    "D_answer_in_basin_not_medoid",
    "E_answer_on_path_or_bridge",
    "F_answer_rendered_nonmedoid",
    "G_answer_budget_cutoff",
    "H_answer_rendered_qa_fail",
    "I_success",
)


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _path_realizability(row: dict[str, Any]) -> dict[str, Any]:
    value = _diagnostics(row).get("path_realizability")
    return value if isinstance(value, dict) else {}


def _answer_in_context(row: dict[str, Any]) -> bool:
    diagnostics = _diagnostics(row)
    stage = diagnostics.get("stage_diagnostics")
    if isinstance(stage, dict):
        context = stage.get("context_rendering")
        if isinstance(context, dict):
            extra = context.get("extra")
            if isinstance(extra, dict) and extra.get("answer_in_context") is not None:
                return bool(extra.get("answer_in_context"))
    path = _path_realizability(row).get("answer_trace")
    if isinstance(path, dict):
        return bool(path.get("answer_chunk_rendered", False))
    return False


def _dataset_name(example: QueryExample) -> str:
    value = example.metadata.get("dataset")
    if value is not None:
        return str(value)
    nested = example.metadata.get("metadata")
    if isinstance(nested, dict) and nested.get("dataset") is not None:
        return str(nested["dataset"])
    return ""


def _node_order(nodes: Sequence[EvidenceNode]) -> dict[str, int]:
    return {str(node.node_id): idx for idx, node in enumerate(nodes)}


def _stage_sets(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = _diagnostics(retrieval_row)
    path = _path_realizability(retrieval_row)
    active_ids = set(_ids(diagnostics.get("active_universe_node_ids", [])))
    if not active_ids:
        active_ids = {str(node.node_id) for node in example.nodes}
    candidate_ids = set(_ids(diagnostics.get("candidate_node_ids", []))) or active_ids
    projected_ids = set(_ids(diagnostics.get("projected_node_ids", []))) or active_ids
    pre_refine_ids = set(_ids(diagnostics.get("pre_refinement_anchor_ids", [])))
    post_refine_ids = set(_ids(retrieval_row.get("anchor_node_ids", [])))
    selected_basin_ids = set(_ids(diagnostics.get("diagnostic_selected_basin_node_ids", [])))
    rendered_ids = set(_ids(retrieval_row.get("context_node_ids", [])))
    support_tree_ids = set(_ids(path.get("support_tree_node_ids", [])))
    path_closure_ids = set(_ids(diagnostics.get("path_closure_node_ids", [])))
    renderer_order = _ids(diagnostics.get("renderer_budget_order_node_ids", []))
    rank = {node_id: pos for pos, node_id in enumerate(renderer_order, start=1)}
    budget_saturated = bool(
        not diagnostics.get("node_budget_satisfied", True)
        or not diagnostics.get("token_budget_satisfied", True)
        or len(rendered_ids) < len(renderer_order)
    )
    return {
        "active": active_ids,
        "candidate": candidate_ids,
        "projected": projected_ids,
        "phase1_medoids": pre_refine_ids,
        "post_refine_medoids": post_refine_ids,
        "selected_basin": selected_basin_ids,
        "rendered": rendered_ids,
        "support_tree": support_tree_ids,
        "path_closure": path_closure_ids,
        "rank": rank,
        "budget_saturated": budget_saturated,
    }


def _row_for_chunk(
    *,
    example: QueryExample,
    chunk_id: str | None,
    stage: dict[str, Any],
    qa_f1: float,
    current_answer_in_context: bool,
    prefix: str,
) -> dict[str, Any]:
    chunk_id_value = chunk_id if chunk_id is not None else None
    active = bool(chunk_id is not None and chunk_id in stage["active"])
    candidate = bool(chunk_id is not None and chunk_id in stage["candidate"])
    projected = bool(chunk_id is not None and chunk_id in stage["projected"])
    phase1 = bool(chunk_id is not None and chunk_id in stage["phase1_medoids"])
    post_refine = bool(chunk_id is not None and chunk_id in stage["post_refine_medoids"])
    selected_basin = bool(chunk_id is not None and chunk_id in stage["selected_basin"])
    support_tree = bool(chunk_id is not None and chunk_id in stage["support_tree"])
    path_closure = bool(chunk_id is not None and chunk_id in stage["path_closure"])
    rendered = bool(chunk_id is not None and chunk_id in stage["rendered"])
    bridge = bool(support_tree and not post_refine)
    rendered_as_medoid = bool(rendered and (phase1 or post_refine))
    rendered_as_bridge = bool(rendered and bridge)
    rendered_as_extra = bool(rendered and not rendered_as_medoid and not rendered_as_bridge and not path_closure)
    rank = stage["rank"].get(chunk_id) if chunk_id is not None else None
    dropped = bool(
        chunk_id is not None
        and rank is not None
        and not rendered
        and stage["budget_saturated"]
    )
    return {
        "query_id": example.query_id,
        "dataset": _dataset_name(example),
        "answer": example.answer or "",
        f"{prefix}_chunk_id": chunk_id_value,
        f"{prefix}_chunk_in_candidate": candidate,
        f"{prefix}_chunk_in_projected": projected,
        f"{prefix}_chunk_in_active_universe": active,
        f"{prefix}_chunk_phase1_selected_medoid": phase1,
        f"{prefix}_chunk_post_refine_medoid": post_refine,
        f"{prefix}_chunk_in_selected_basin": selected_basin,
        f"{prefix}_chunk_on_support_tree": support_tree,
        f"{prefix}_chunk_is_bridge": bridge,
        f"{prefix}_chunk_is_path_closure": path_closure,
        f"{prefix}_chunk_current_rendered": rendered,
        f"{prefix}_chunk_rendered_as_medoid": rendered_as_medoid,
        f"{prefix}_chunk_rendered_as_bridge": rendered_as_bridge,
        f"{prefix}_chunk_rendered_as_extra_nonmedoid": rendered_as_extra,
        f"{prefix}_chunk_dropped_by_budget": dropped,
        f"{prefix}_chunk_rank_before_budget": rank,
        "current_renderer_answer_in_context": bool(current_answer_in_context),
        "qa_f1": float(qa_f1),
    }


def answer_carrier_attribution_rows(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
    qa_f1: float = 0.0,
    current_answer_in_context: bool | None = None,
) -> list[dict[str, Any]]:
    stage = _stage_sets(example=example, retrieval_row=retrieval_row)
    current_answer = _answer_in_context(retrieval_row) if current_answer_in_context is None else bool(current_answer_in_context)
    answer_ids = answer_containing_chunk_ids(example, example.nodes)
    rows = [
        {
            "carrier_type": "answer",
            **_row_for_chunk(
                example=example,
                chunk_id=chunk_id,
                stage=stage,
                qa_f1=qa_f1,
                current_answer_in_context=current_answer,
                prefix="answer",
            ),
        }
        for chunk_id in answer_ids
    ]
    if not rows:
        rows.append(
            {
                "carrier_type": "answer",
                **_row_for_chunk(
                    example=example,
                    chunk_id=None,
                    stage=stage,
                    qa_f1=qa_f1,
                    current_answer_in_context=current_answer,
                    prefix="answer",
                ),
            }
        )
    return rows


def gold_carrier_attribution_rows(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
    qa_f1: float = 0.0,
    current_answer_in_context: bool | None = None,
) -> list[dict[str, Any]]:
    stage = _stage_sets(example=example, retrieval_row=retrieval_row)
    current_answer = _answer_in_context(retrieval_row) if current_answer_in_context is None else bool(current_answer_in_context)
    node_order = _node_order(example.nodes)
    gold_ids = sorted(
        (str(value) for value in example.gold_node_ids),
        key=lambda node_id: (node_order.get(node_id, 10**9), node_id),
    )
    return [
        {
            "carrier_type": "gold",
            **_row_for_chunk(
                example=example,
                chunk_id=chunk_id,
                stage=stage,
                qa_f1=qa_f1,
                current_answer_in_context=current_answer,
                prefix="gold",
            ),
        }
        for chunk_id in gold_ids
    ]


def classify_answer_carrier_failure(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "A_answer_not_in_candidate"
    real_rows = [row for row in rows if row.get("answer_chunk_id") is not None]
    if not real_rows or not any(bool(row.get("answer_chunk_in_candidate")) for row in real_rows):
        return "A_answer_not_in_candidate"
    if not any(bool(row.get("answer_chunk_in_projected")) for row in real_rows):
        return "B_answer_lost_at_projection"
    if any(bool(row.get("answer_chunk_dropped_by_budget")) for row in real_rows):
        return "G_answer_budget_cutoff"
    if any(bool(row.get("answer_chunk_rendered_as_extra_nonmedoid")) for row in real_rows):
        return "F_answer_rendered_nonmedoid"
    if any(bool(row.get("answer_chunk_rendered_as_bridge")) or bool(row.get("answer_chunk_is_path_closure")) for row in real_rows):
        return "E_answer_on_path_or_bridge"
    if any(bool(row.get("current_renderer_answer_in_context")) for row in rows):
        if max(float(row.get("qa_f1", 0.0) or 0.0) for row in rows) > 0.0:
            return "I_success"
        return "H_answer_rendered_qa_fail"
    if any(bool(row.get("answer_chunk_in_selected_basin")) and not bool(row.get("answer_chunk_post_refine_medoid")) for row in real_rows):
        return "D_answer_in_basin_not_medoid"
    if any(bool(row.get("answer_chunk_in_projected")) for row in real_rows):
        return "C_answer_lost_at_medoid_selection"
    return "A_answer_not_in_candidate"


def _group_by_query(rows: Sequence[dict[str, Any]], *, carrier_type: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("carrier_type") == carrier_type:
            grouped[str(row.get("query_id"))].append(row)
    return grouped


def _query_rate(grouped: dict[str, list[dict[str, Any]]], key: str) -> float:
    if not grouped:
        return 0.0
    return float(sum(1 for rows in grouped.values() if any(bool(row.get(key)) for row in rows)) / len(grouped))


def aggregate_answer_carrier_attribution(
    rows: Sequence[dict[str, Any]],
    *,
    renderer_role_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    answer_grouped = _group_by_query(rows, carrier_type="answer")
    answer_failure_counts = Counter(classify_answer_carrier_failure(query_rows) for query_rows in answer_grouped.values())
    current_answer = _query_rate(answer_grouped, "current_renderer_answer_in_context")
    medoid_availability = _query_rate(answer_grouped, "answer_chunk_post_refine_medoid")
    role_payload = aggregate_renderer_role_trace(renderer_role_rows)
    return {
        "num_queries": len(answer_grouped),
        "answer_chunk_candidate_rate": _query_rate(answer_grouped, "answer_chunk_in_candidate"),
        "answer_chunk_projected_rate": _query_rate(answer_grouped, "answer_chunk_in_projected"),
        "answer_chunk_selected_medoid_rate": _query_rate(answer_grouped, "answer_chunk_phase1_selected_medoid"),
        "answer_chunk_post_refine_medoid_rate": medoid_availability,
        "answer_chunk_selected_basin_rate": _query_rate(answer_grouped, "answer_chunk_in_selected_basin"),
        "answer_chunk_support_tree_rate": _query_rate(answer_grouped, "answer_chunk_on_support_tree"),
        "answer_chunk_bridge_rate": _query_rate(answer_grouped, "answer_chunk_is_bridge"),
        "answer_chunk_current_rendered_rate": _query_rate(answer_grouped, "answer_chunk_current_rendered"),
        "answer_chunk_rendered_nonmedoid_rate": _query_rate(answer_grouped, "answer_chunk_rendered_as_extra_nonmedoid"),
        "answer_chunk_budget_cutoff_rate": _query_rate(answer_grouped, "answer_chunk_dropped_by_budget"),
        "current_answer_in_context": current_answer,
        "selected_medoid_answer_availability": medoid_availability,
        "current_minus_medoid_answer_gap": current_answer - medoid_availability,
        "answer_carrier_failure_counts": {
            key: int(answer_failure_counts.get(key, 0)) for key in ANSWER_CARRIER_FAILURES
        },
        **role_payload,
    }


__all__ = [
    "ANSWER_CARRIER_FAILURES",
    "aggregate_answer_carrier_attribution",
    "answer_carrier_attribution_rows",
    "classify_answer_carrier_failure",
    "gold_carrier_attribution_rows",
]
