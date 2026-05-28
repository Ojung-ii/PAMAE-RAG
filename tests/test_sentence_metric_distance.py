import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE_CHUNK_HIER
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index
from pamae_rag.sentence_graph.sentence_metric_distance import (
    sentence_shortest_path_distances,
    triangle_inequality_violation_count,
)


def _node(node_id: str, text: str, title: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        metadata={"title": title, "corpus_index": node_id},
    )


def test_sentence_metric_excludes_chunk_parent_edges_by_default():
    index = build_sentence_graph_index(
        [
            _node("n1", "Ada met Charles.", "Ada"),
            _node("n2", "Grace met Charles.", "Grace"),
        ],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
        use_chunk_parent_edges_in_metric=False,
    )

    metric = sentence_shortest_path_distances(index, index.sentence_ids)

    assert metric.distance_matrix.shape == (2, 2)
    assert metric.diagnostics["metric_use_chunk_parent_edges"] is False
    assert triangle_inequality_violation_count(metric.distance_matrix) == 0


def test_chunk_parent_edges_do_not_create_metric_shortcut():
    index = build_sentence_graph_index(
        [_node("n1", "Ada met Charles. Grace solved it.", "Ada")],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
        use_chunk_parent_edges_in_metric=False,
    )
    with_parent = build_sentence_graph_index(
        [_node("n1", "Ada met Charles. Grace solved it.", "Ada")],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
        use_chunk_parent_edges_in_metric=True,
    )

    no_parent_metric = sentence_shortest_path_distances(
        index,
        index.sentence_ids,
        use_chunk_parent_edges_in_metric=False,
    )
    parent_metric = sentence_shortest_path_distances(
        with_parent,
        with_parent.sentence_ids,
        use_chunk_parent_edges_in_metric=True,
    )

    assert no_parent_metric.distance_matrix[0, 1] == 1.0
    assert parent_metric.distance_matrix[0, 1] == 1.0
    assert no_parent_metric.diagnostics["metric_use_chunk_parent_edges"] is False
    assert parent_metric.diagnostics["metric_use_chunk_parent_edges"] is True
