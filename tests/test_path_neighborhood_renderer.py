from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.rendering.path_neighborhood_renderer import render_path_neighborhood_indices


def _node(node_id: str, text: str = "") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text or node_id,
        embedding=np.zeros(2),
        token_count=1,
        node_type="chunk",
    )


def test_path_neighborhood_expands_by_distance_to_support_tree_before_mass() -> None:
    nodes = (
        _node("anchor"),
        _node("medoid"),
        _node("near_low_mass", "gold answer"),
        _node("far_high_mass"),
    )
    example = QueryExample(
        query_id="q1",
        query="q",
        nodes=nodes,
        gold_node_ids=frozenset({"near_low_mass"}),
        answer="answer",
    )
    distance = np.asarray(
        [
            [0.0, 1.0, 1.1, 3.0],
            [1.0, 0.0, 0.1, 2.0],
            [1.1, 0.1, 0.0, 2.1],
            [3.0, 2.0, 2.1, 0.0],
        ]
    )

    result = render_path_neighborhood_indices(
        nodes=nodes,
        selected_medoids=[1],
        query_anchors=[0],
        distance_matrix=distance,
        rho=np.asarray([0.4, 0.3, 0.01, 0.99]),
        max_context_tokens=3,
        max_context_nodes=3,
        active_indices=[0, 1, 2, 3],
        disconnected_distance=9.0,
        example=example,
    )

    assert result.indices == [1, 0, 2]
    assert result.diagnostics["renderer_mode"] == "path_neighborhood"
    assert result.diagnostics["neighborhood_chunk_count"] == 1
    assert result.diagnostics["answer_in_context"] is True
    assert result.diagnostics["rendered_recall"] == 1.0

