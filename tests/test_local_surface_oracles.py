from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.local_surface.local_surface_graph import build_local_surface_graph
from pamae_rag.local_surface.local_surface_renderers import (
    ORACLE_RENDERERS,
    SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
    SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
    render_local_surface,
)


def _node(node_id: str, title: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        metadata={"title": title},
    )


def test_local_oracle_renderers_are_diagnostic_only_and_not_adoption_candidates() -> None:
    example = QueryExample(
        query_id="q0",
        query="Where was Ada born?",
        answer="London",
        nodes=(
            _node("c0", "Ada", "Ada wrote notes. Ada was born in London. Ada left."),
        ),
        metadata={
            "answers": ["London"],
            "support_facts": [{"title": "Ada", "sentence_id": 1}],
        },
    )
    graph = build_local_surface_graph(example.nodes, ["c0"])

    answer = render_local_surface(
        example=example,
        graph=graph,
        renderer_mode=SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
    )
    gold = render_local_surface(
        example=example,
        graph=graph,
        renderer_mode=SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
    )

    assert SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE in ORACLE_RENDERERS
    assert SELECTED_CHUNK_GOLD_SENTENCE_ORACLE in ORACLE_RENDERERS
    assert answer.diagnostics["oracle_renderer"] is True
    assert answer.diagnostics["uses_answer_string"] is True
    assert gold.diagnostics["oracle_renderer"] is True
    assert gold.diagnostics["uses_gold_label"] is True
    assert answer.diagnostics["answer_in_context"] is True
    assert gold.diagnostics["gold_sentence_rendered"] is True
