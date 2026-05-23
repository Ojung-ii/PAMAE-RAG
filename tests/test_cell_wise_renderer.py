import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.retrieval.renderer import render_context_indices


def _node(node_id: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=node_id,
        embedding=np.array([1.0, 0.0]),
        token_count=1,
        metadata={"title": node_id},
    )


def test_cell_wise_renderer_includes_top_rho_per_cell():
    nodes = tuple(_node(str(i)) for i in range(6))
    distance_matrix = np.array(
        [
            [0.0, 0.1, 0.2, 0.9, 0.9, 0.9],
            [0.1, 0.0, 0.2, 0.8, 0.8, 0.8],
            [0.2, 0.2, 0.0, 0.7, 0.7, 0.7],
            [0.9, 0.8, 0.7, 0.0, 0.1, 0.2],
            [0.9, 0.8, 0.7, 0.1, 0.0, 0.2],
            [0.9, 0.8, 0.7, 0.2, 0.2, 0.0],
        ]
    )
    rho = np.array([0.05, 0.50, 0.10, 0.05, 0.45, 0.15])

    selected = render_context_indices(
        nodes,
        anchors=[0, 3],
        distance_matrix=distance_matrix,
        rho=rho,
        max_context_tokens=4,
        renderer="cell_top_rho",
    )

    assert selected[:2] == [0, 3]
    assert 1 in selected
    assert 4 in selected
    assert len(selected) == 4

