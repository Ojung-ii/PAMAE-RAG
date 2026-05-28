import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.graph.content_graph import (
    build_content_graph_index,
    normalize_content_text,
    project_content_graph_to_query_graph,
)


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


def test_content_graph_projects_text_edges_to_query_graph():
    nodes = [
        _node("n1", "Ada Lovelace worked with Charles Babbage."),
        _node("n2", "Charles Babbage built engines."),
    ]

    graph, index, projected = project_content_graph_to_query_graph(
        nodes,
        edge_lengths={"shared_entity": 1.0, "entity_fact_bridge": 1.0},
        max_edges_per_node=4,
    )

    assert index.diagnostics["content_graph_title_metadata_used"] is False
    assert graph.num_edges > 0
    assert set(projected) == {0, 1}
    assert graph.edge_counts_by_type["shared_entity"] > 0


def test_content_graph_projection_deduplicates_repeated_bridge_entity_pairs():
    nodes = [
        _node(
            "n1",
            "Ada Lovelace met Charles Babbage. Ada Lovelace advised Charles Babbage.",
        ),
        _node("n2", "Charles Babbage worked with Ada Lovelace."),
    ]

    graph, index, projected = project_content_graph_to_query_graph(
        nodes,
        edge_lengths={"shared_entity": 1.0, "entity_fact_bridge": 1.0},
        max_edges_per_node=8,
    )

    assert graph.num_edges > 0
    assert set(projected) == {0, 1}
    assert index.diagnostics["content_graph_unique_entity_fact_bridge_pairs"] < len(index.triples)
