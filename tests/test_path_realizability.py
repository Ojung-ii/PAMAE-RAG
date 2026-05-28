from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import compute_path_realizability, path_nodes


def _node(node_id: str, text: str = "") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text or node_id,
        embedding=np.zeros(2),
        token_count=1,
        node_type="chunk",
    )


def test_path_nodes_uses_graph_metric_shortest_path_membership() -> None:
    nodes = tuple(_node(str(i)) for i in range(4))
    distance = np.asarray(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )

    assert path_nodes(distance, 0, 2, nodes, disconnected_distance=9.0) == [0, 1, 2]


def test_path_realizability_marks_gold_in_selected_basin_and_rendering_gap() -> None:
    nodes = (
        _node("anchor", "anchor text"),
        _node("medoid", "medoid text"),
        _node("gold", "the answer phrase appears here"),
    )
    example = QueryExample(
        query_id="q1",
        query="where is the answer?",
        nodes=nodes,
        gold_node_ids=frozenset({"gold"}),
        answer="answer phrase",
        metadata={"metadata": {"dataset": "unit"}},
    )
    distance = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = compute_path_realizability(
        example=example,
        nodes=nodes,
        candidate_indices=[0, 1, 2],
        projected_node_ids=["anchor", "medoid", "gold"],
        selected_medoids=[1],
        context_indices=[1],
        distance_matrix=distance,
        rho=np.asarray([0.7, 0.2, 0.1]),
        selected_mode="current_content",
        renderer_mode="current",
        max_context_tokens=10,
        max_context_nodes=2,
        disconnected_distance=9.0,
        node_to_basin={0: 0, 1: 0, 2: 0},
        query_anchors=[0],
    )

    row = result.gold_rows[0]
    assert row["gold_in_selected_basin"] is True
    assert row["medoid_to_gold_path_exists"] is True
    assert row["gold_on_existing_support_tree"] is False
    assert row["gold_rendered"] is False
    assert result.answer_trace["answer_chunk_in_projected"] is True
    assert result.answer_trace["answer_chunk_rendered"] is False

