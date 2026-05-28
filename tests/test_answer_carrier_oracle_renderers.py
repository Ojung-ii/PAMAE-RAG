from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.rendering.answer_carrier_oracle_renderers import (
    CURRENT_ANSWER_ROLE_ORACLE,
    GOLD_CHUNK_ROLE_ORACLE,
    PROJECTED_ANSWER_CHUNK_ORACLE,
    SELECTED_BASIN_ANSWER_CHUNK_ORACLE,
    render_answer_carrier_oracle,
)


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
        query="Where is Ada from?",
        answer="London",
        nodes=(
            _node("c0", "Ada was a writer."),
            _node("c1", "Ada was born in London."),
            _node("c2", "Extra context."),
        ),
        gold_node_ids=frozenset({"c0"}),
        metadata={"dataset": "unit"},
    )


def _retrieval_row() -> dict:
    return {
        "query_id": "q1",
        "anchor_node_ids": ["c2"],
        "context_node_ids": ["c0", "c1"],
        "diagnostics": {
            "projected_node_ids": ["c0", "c1", "c2"],
            "diagnostic_selected_basin_node_ids": ["c1", "c2"],
        },
    }


def test_projected_and_selected_basin_answer_chunk_oracles_are_diagnostic_only() -> None:
    example = _example()

    projected = render_answer_carrier_oracle(
        example=example,
        retrieval_row=_retrieval_row(),
        renderer_mode=PROJECTED_ANSWER_CHUNK_ORACLE,
        max_context_tokens=100,
    )
    basin = render_answer_carrier_oracle(
        example=example,
        retrieval_row=_retrieval_row(),
        renderer_mode=SELECTED_BASIN_ANSWER_CHUNK_ORACLE,
        max_context_tokens=100,
    )

    assert projected.context_node_ids == ("c1",)
    assert basin.context_node_ids == ("c1",)
    assert projected.diagnostics["oracle_renderer"] is True
    assert projected.diagnostics["uses_answer_string"] is True


def test_current_answer_and_gold_role_oracles_preserve_current_render_order() -> None:
    example = _example()

    answer = render_answer_carrier_oracle(
        example=example,
        retrieval_row=_retrieval_row(),
        renderer_mode=CURRENT_ANSWER_ROLE_ORACLE,
        max_context_tokens=100,
    )
    gold = render_answer_carrier_oracle(
        example=example,
        retrieval_row=_retrieval_row(),
        renderer_mode=GOLD_CHUNK_ROLE_ORACLE,
        max_context_tokens=100,
    )

    assert answer.context_node_ids == ("c1",)
    assert gold.context_node_ids == ("c0",)
    assert gold.diagnostics["uses_answer_string"] is False
    assert gold.diagnostics["uses_gold_label"] is True
