import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE, ENTITY_SENTENCE_CHUNK_HIER
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index


def _node(node_id: str, text: str, title: str = "Ada") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        relevance=1.0,
        token_count=max(1, len(text.split())),
        metadata={"title": title, "corpus_index": node_id.removeprefix("n")},
    )


def test_entity_sentence_graph_creates_sentence_nodes_and_edges_without_chunks():
    index = build_sentence_graph_index(
        [_node("n1", "Ada Lovelace met Charles Babbage. London hosted Ada Lovelace.")],
        graph_variant=ENTITY_SENTENCE,
    )

    assert len(index.sentence_nodes) == 2
    assert index.entity_ids
    assert not index.chunk_nodes
    edge_types = {edge.edge_type for edge in index.edges}
    assert "entity_sentence_mention" in edge_types
    assert "sent_adjacent" in edge_types
    assert "sent_chunk_parent" not in edge_types
    assert all(sentence.sentence_id.startswith("sent:") for sentence in index.sentence_nodes)


def test_entity_sentence_chunk_hier_adds_metadata_parent_edges_not_metric_edges():
    index = build_sentence_graph_index(
        [_node("n1", "Ada Lovelace met Charles Babbage. London hosted Ada Lovelace.")],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
        use_chunk_parent_edges_in_metric=False,
    )

    assert len(index.chunk_nodes) == 1
    parent_edges = [edge for edge in index.edges if edge.edge_type == "sent_chunk_parent"]
    assert parent_edges
    assert all(not edge.include_in_metric for edge in parent_edges)
    metric_neighbors = index.adjacency(include_chunk_parent_edges=False)
    chunk_ids = {chunk.chunk_node_id for chunk in index.chunk_nodes}
    assert not (set(metric_neighbors) & chunk_ids)
    assert index.diagnostics["chunk_parent_edges_are_metric_edges"] is False


def test_entity_sentence_chunk_hier_can_explicitly_include_parent_edges_for_diagnostics():
    index = build_sentence_graph_index(
        [_node("n1", "Ada Lovelace met Charles Babbage.")],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
        use_chunk_parent_edges_in_metric=True,
    )

    parent_edges = [edge for edge in index.edges if edge.edge_type == "sent_chunk_parent"]
    assert parent_edges
    assert all(edge.include_in_metric for edge in parent_edges)
