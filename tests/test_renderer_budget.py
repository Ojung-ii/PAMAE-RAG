import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.retrieval.renderer import render_context_indices


def _node(node_id: str, tokens: int = 1) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=node_id,
        embedding=np.array([1.0, 0.0]),
        token_count=tokens,
        metadata={"title": node_id},
    )


def test_renderer_respects_max_context_nodes():
    nodes = tuple(_node(str(i)) for i in range(8))
    distance_matrix = np.abs(np.subtract.outer(np.arange(8), np.arange(8))).astype(float)
    rho = np.linspace(0.8, 0.1, 8)

    selected = render_context_indices(
        nodes,
        anchors=[0, 5],
        distance_matrix=distance_matrix,
        rho=rho,
        max_context_tokens=100,
        max_context_nodes=4,
        renderer="cell_top_rho",
    )

    assert selected[:2] == [0, 5]
    assert len(selected) <= 4
    assert len(selected) == len(set(selected))


def test_renderer_respects_max_context_tokens():
    nodes = (
        _node("a0", tokens=1),
        _node("a1", tokens=1),
        _node("expensive", tokens=10),
        _node("cheap", tokens=1),
    )
    distance_matrix = np.array(
        [
            [0.0, 1.0, 0.1, 0.2],
            [1.0, 0.0, 0.2, 0.1],
            [0.1, 0.2, 0.0, 0.3],
            [0.2, 0.1, 0.3, 0.0],
        ]
    )
    rho = np.array([0.2, 0.2, 0.9, 0.8])

    selected = render_context_indices(
        nodes,
        anchors=[0, 1],
        distance_matrix=distance_matrix,
        rho=rho,
        max_context_tokens=2,
        max_context_nodes=4,
        renderer="global_top_rho",
    )

    assert selected == [0, 1]
    assert sum(nodes[i].token_count for i in selected) <= 2

