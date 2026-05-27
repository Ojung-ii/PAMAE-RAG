import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.graph.content_graph import build_content_graph_index, normalize_content_text


def _node(node_id: str, text: str, title: str = "Metadata Trap") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        metadata={"title": title},
    )


def test_content_graph_extracts_text_entities_facts_and_edges():
    index = build_content_graph_index(
        [
            _node("n1", "Ada Lovelace was born in London. London is in England."),
            _node("n2", "Charles Babbage worked with Ada Lovelace."),
        ]
    )

    surfaces = {surface for entity in index.entities for surface in entity.surfaces}
    assert {"Ada Lovelace", "London", "England", "Charles Babbage"} <= surfaces
    assert len(index.facts) == 3
    assert index.triples
    edge_types = {edge.edge_type for edge in index.edges}
    assert {"chunk_entity", "chunk_fact", "fact_entity", "entity_cofact"} <= edge_types
    assert index.diagnostics["content_graph_title_metadata_used"] is False


def test_content_graph_does_not_use_title_metadata_as_entity_source():
    index = build_content_graph_index([_node("n1", "plain lower-case evidence only.", title="Ada Lovelace")])

    surfaces = {normalize_content_text(surface) for entity in index.entities for surface in entity.surfaces}
    assert "ada lovelace" not in surfaces
    assert index.diagnostics["content_graph_num_facts"] == 1
