from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_carrier_taxonomy import aggregate_path_carrier_taxonomy
from pamae_rag.diagnostics.support_tree_carrier_trace import (
    aggregate_support_tree_carrier_traces,
    support_tree_carrier_trace_rows,
)


def _node(node_id: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        token_count=1,
        node_type="chunk",
    )


def _example() -> QueryExample:
    return QueryExample(
        query_id="q1",
        query="q",
        answer="answer phrase",
        nodes=(
            _node("c_anchor", "anchor text"),
            _node("c_bridge", "answer phrase appears here"),
            _node("c_medoid", "medoid text"),
        ),
        gold_node_ids=frozenset({"c_bridge"}),
        metadata={"dataset": "unit"},
    )


def _row(renderer: str, context: list[str]) -> dict:
    return {
        "query_id": "q1",
        "anchor_node_ids": ["c_medoid"],
        "context_node_ids": context,
        "diagnostics": {
            "active_universe_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "candidate_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "projected_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "pre_refinement_anchor_ids": ["c_medoid"],
            "diagnostic_selected_basin_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "phase1_support_tree_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "refined_support_tree_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "refined_anchor_medoid_path_node_ids": ["c_anchor", "c_bridge", "c_medoid"],
            "refined_medoid_medoid_path_node_ids": [],
            "path_carrier_order_node_ids": ["c_medoid", "c_bridge"],
            "budget_cutoff_node_ids": [],
            "rendered_path_role_node_ids": {
                "medoid": ["c_medoid"],
                "anchor_medoid_path": ["c_bridge"],
                "medoid_medoid_path": [],
            },
            "stage_diagnostics": {
                "context_rendering": {"extra": {"answer_in_context": "c_bridge" in context}},
            },
            "renderer": renderer,
        },
    }


def test_support_tree_trace_marks_metric_path_answer_carrier() -> None:
    rows = support_tree_carrier_trace_rows(
        example=_example(),
        retrieval_row=_row("metric_path_carrier", ["c_medoid", "c_bridge"]),
        renderer_mode="metric_path_carrier",
        qa_f1=0.5,
    )

    assert rows[0]["answer_chunk_on_refined_support_tree"] is True
    assert rows[0]["answer_chunk_on_anchor_medoid_path"] is True
    assert rows[0]["answer_chunk_metric_path_rendered"] is True
    aggregate = aggregate_support_tree_carrier_traces(rows)
    assert aggregate["answer_on_refined_support_tree_rate"] == 1.0
    assert aggregate["answer_metric_path_rendered_rate"] == 1.0


def test_path_carrier_taxonomy_finds_current_only_hidden_recovery() -> None:
    example = _example()
    current = support_tree_carrier_trace_rows(
        example=example,
        retrieval_row=_row("current_renderer", ["c_bridge"]),
        renderer_mode="current_renderer",
        qa_f1=0.5,
    )
    metric = support_tree_carrier_trace_rows(
        example=example,
        retrieval_row=_row("metric_path_carrier", ["c_medoid"]),
        renderer_mode="metric_path_carrier",
        qa_f1=0.0,
    )

    taxonomy = aggregate_path_carrier_taxonomy([*current, *metric])

    assert taxonomy["path_carrier_failure_counts"]["E_current_only_hidden_recovery"] == 1
