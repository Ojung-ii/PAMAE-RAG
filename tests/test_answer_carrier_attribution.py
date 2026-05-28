from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.answer_carrier_attribution import (
    aggregate_answer_carrier_attribution,
    answer_carrier_attribution_rows,
    classify_answer_carrier_failure,
    gold_carrier_attribution_rows,
)
from pamae_rag.diagnostics.renderer_role_trace import renderer_role_trace_rows


def _node(node_id: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        token_count=max(1, len(text.split())),
        node_type="chunk",
    )


def _example() -> QueryExample:
    return QueryExample(
        query_id="q1",
        query="Where is the answer?",
        answer="answer phrase",
        nodes=(
            _node("c0", "selected medoid"),
            _node("c1", "support bridge"),
            _node("c2", "this chunk has the answer phrase"),
        ),
        gold_node_ids=frozenset({"c1"}),
        metadata={"dataset": "unit"},
    )


def _retrieval_row() -> dict:
    return {
        "query_id": "q1",
        "anchor_node_ids": ["c0"],
        "context_node_ids": ["c0", "c2"],
        "diagnostics": {
            "active_universe_node_ids": ["c0", "c1", "c2"],
            "candidate_node_ids": ["c0", "c1", "c2"],
            "projected_node_ids": ["c0", "c1", "c2"],
            "pre_refinement_anchor_ids": ["c0"],
            "diagnostic_selected_basin_node_ids": ["c0", "c2"],
            "renderer_budget_order_node_ids": ["c0", "c2", "c1"],
            "node_budget_satisfied": True,
            "token_budget_satisfied": True,
            "stage_diagnostics": {
                "context_rendering": {"extra": {"answer_in_context": True}},
            },
            "path_realizability": {
                "support_tree_node_ids": ["c0", "c1"],
                "answer_trace": {"answer_chunk_rendered": True},
            },
        },
    }


def test_answer_carrier_rows_mark_extra_nonmedoid_recovery() -> None:
    rows = answer_carrier_attribution_rows(
        example=_example(),
        retrieval_row=_retrieval_row(),
        qa_f1=0.25,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["answer_chunk_id"] == "c2"
    assert row["answer_chunk_in_projected"] is True
    assert row["answer_chunk_post_refine_medoid"] is False
    assert row["answer_chunk_in_selected_basin"] is True
    assert row["answer_chunk_current_rendered"] is True
    assert row["answer_chunk_rendered_as_extra_nonmedoid"] is True
    assert classify_answer_carrier_failure(rows) == "F_answer_rendered_nonmedoid"


def test_answer_carrier_aggregate_reports_current_minus_medoid_gap() -> None:
    example = _example()
    answer_rows = answer_carrier_attribution_rows(example=example, retrieval_row=_retrieval_row(), qa_f1=0.25)
    gold_rows = gold_carrier_attribution_rows(example=example, retrieval_row=_retrieval_row(), qa_f1=0.25)
    role_rows = renderer_role_trace_rows(example=example, retrieval_row=_retrieval_row())

    aggregate = aggregate_answer_carrier_attribution(
        [*answer_rows, *gold_rows],
        renderer_role_rows=role_rows,
    )

    assert aggregate["current_answer_in_context"] == 1.0
    assert aggregate["selected_medoid_answer_availability"] == 0.0
    assert aggregate["current_minus_medoid_answer_gap"] == 1.0
    assert aggregate["answer_render_role_distribution"]["extra_nonmedoid"] == 1
    assert aggregate["gold_render_role_distribution"]["support_tree_bridge"] == 0
