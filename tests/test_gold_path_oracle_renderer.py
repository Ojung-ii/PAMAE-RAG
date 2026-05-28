from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.rendering.gold_path_oracle_renderer import render_gold_path_oracle_indices


def _node(node_id: str, text: str = "") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text or node_id,
        embedding=np.zeros(2),
        token_count=1,
        node_type="chunk",
    )


def test_gold_path_oracle_adds_reachable_gold_path_only_for_diagnostics() -> None:
    nodes = (
        _node("anchor"),
        _node("medoid"),
        _node("gold", "The answer is visible here."),
    )
    example = QueryExample(
        query_id="q1",
        query="q",
        nodes=nodes,
        gold_node_ids=frozenset({"gold"}),
        answer="answer",
    )
    distance = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = render_gold_path_oracle_indices(
        example=example,
        nodes=nodes,
        selected_medoids=[1],
        query_anchors=[0],
        distance_matrix=distance,
        max_context_tokens=10,
        max_context_nodes=4,
        node_to_basin={0: 0, 1: 0, 2: 0},
        disconnected_distance=9.0,
    )

    assert result.indices == [1, 0, 2]
    assert result.diagnostics["oracle_renderer"] is True
    assert result.diagnostics["gold_path_added"] is True
    assert result.diagnostics["rendered_recall"] == 1.0
    assert result.diagnostics["answer_in_context"] is True

