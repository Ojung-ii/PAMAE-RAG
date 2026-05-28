from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.local_surface.local_sentence_medoid import LocalMedoidConfig
from pamae_rag.local_surface.local_surface_graph import build_local_surface_graph
from pamae_rag.local_surface.local_surface_renderers import (
    LOCAL_SENTENCE_MEDOID,
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


def _example() -> QueryExample:
    return QueryExample(
        query_id="q0",
        query="Where was Ada Lovelace born?",
        answer="London",
        nodes=(
            _node("c0", "Ada", "Ada Lovelace wrote notes. Ada Lovelace was born in London."),
        ),
        metadata={
            "answers": ["London"],
            "support_facts": [{"title": "Ada", "sentence_id": 1}],
        },
    )


def test_local_sentence_medoid_renderer_does_not_output_full_chunk() -> None:
    example = _example()
    graph = build_local_surface_graph(example.nodes, ["c0"])

    rendered = render_local_surface(
        example=example,
        graph=graph,
        renderer_mode=LOCAL_SENTENCE_MEDOID,
        medoid_config=LocalMedoidConfig(local_sentence_medoids=1),
    )

    assert rendered.context_nodes
    text = "\n".join(str(node["text"]) for node in rendered.context_nodes)
    assert "Ada Lovelace wrote notes. Ada Lovelace was born in London." not in text
    assert rendered.diagnostics["oracle_renderer"] is False


def test_oracle_renderers_are_marked_oracle_only() -> None:
    example = _example()
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

    assert answer.diagnostics["oracle_renderer"] is True
    assert answer.diagnostics["uses_answer_string"] is True
    assert gold.diagnostics["oracle_renderer"] is True
    assert gold.diagnostics["uses_gold_label"] is True
