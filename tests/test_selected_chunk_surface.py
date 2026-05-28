from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.selected_chunk_surface import (
    aggregate_selected_chunk_surface_traces,
    selected_chunk_surface_trace,
)


def _node(node_id: str, title: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        metadata={"title": title},
    )


def test_selected_chunk_surface_traces_answer_and_gold_sentences() -> None:
    example = QueryExample(
        query_id="q0",
        query="Where was Ada born?",
        answer="London",
        nodes=(
            _node("c0", "Ada", "Ada wrote notes. Ada was born in London."),
            _node("c1", "Other", "Other text."),
        ),
        metadata={
            "dataset": "unit",
            "answers": ["London"],
            "support_facts": [{"title": "Ada", "sentence_id": 1}],
        },
    )

    trace = selected_chunk_surface_trace(
        example=example,
        selected_chunk_ids=["c0"],
        rendered_chunk_ids=[],
        explicit_rendered_sentence_ids=["sent:c0:c0:1"],
        qa_f1=1.0,
    )

    assert trace["answer_sentence_in_selected_chunks"] is True
    assert trace["gold_support_sentence_in_selected_chunks"] is True
    assert trace["current_renderer_answer_in_context"] is True
    assert trace["current_renderer_gold_sentence_rendered"] is True
    assert trace["answer_sentence_count_in_selected_chunks"] == 1
    assert trace["gold_sentence_count_in_selected_chunks"] == 1


def test_selected_chunk_surface_aggregate_conditional_recovery() -> None:
    rows = [
        {
            "answer_chunk_in_selected_chunks": True,
            "answer_sentence_in_selected_chunks": True,
            "gold_support_sentence_in_selected_chunks": True,
            "current_renderer_answer_in_context": True,
            "current_renderer_gold_sentence_rendered": False,
        },
        {
            "answer_chunk_in_selected_chunks": False,
            "answer_sentence_in_selected_chunks": False,
            "gold_support_sentence_in_selected_chunks": True,
            "current_renderer_answer_in_context": False,
            "current_renderer_gold_sentence_rendered": True,
        },
    ]

    aggregate = aggregate_selected_chunk_surface_traces(rows)

    assert aggregate["answer_chunk_selected_rate"] == 0.5
    assert aggregate["answer_sentence_available_in_selected_chunks_rate"] == 0.5
    assert aggregate["gold_sentence_available_in_selected_chunks_rate"] == 1.0
    assert aggregate["current_renderer_answer_recovery_given_available"] == 1.0
    assert aggregate["current_renderer_gold_recovery_given_available"] == 0.5
