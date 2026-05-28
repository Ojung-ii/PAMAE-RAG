from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.local_surface.local_surface_graph import build_local_surface_graph


def _node(node_id: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        metadata={"title": node_id},
    )


def test_local_graph_construction_has_sentence_entity_fact_edges() -> None:
    graph = build_local_surface_graph(
        [
            _node("c0", "Ada Lovelace wrote notes. London hosted Ada Lovelace."),
            _node("c1", "Grace Hopper made COBOL."),
        ],
        ["c0"],
    )

    assert graph.sentences
    assert graph.entity_ids
    assert graph.facts
    assert all(edge.length > 0 for edge in graph.edges)
    edge_types = {edge.edge_type for edge in graph.edges}
    assert "entity_sentence" in edge_types
    assert "sentence_fact" in edge_types
    assert "fact_entity" in edge_types
    assert "sent_adjacent" in edge_types
    assert graph.diagnostics["edge_lengths_positive"] is True
