from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.embedding_store import EmbeddingStore
from pamae_rag.semantic.semantic_weighted_tree import (
    semantic_edge_length,
    semantic_weighted_support_tree_indices,
)


def _node(node_id: str, *, embedding: list[float], node_type: str = "chunk") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=f"text {node_id}",
        embedding=np.asarray(embedding, dtype=float),
        token_count=1,
        node_type=node_type,
    )


def _example() -> QueryExample:
    return QueryExample(
        query_id="q",
        query="q",
        nodes=(
            _node("anchor", embedding=[1.0, 0.0], node_type="entity"),
            _node("c_bridge", embedding=[1.0, 0.0]),
            _node("c_medoid", embedding=[0.0, 1.0]),
        ),
    )


def _distance() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )


def test_semantic_weighted_tree_uses_positive_edge_lengths() -> None:
    example = _example()
    store = EmbeddingStore.from_example(example)

    length = semantic_edge_length(left_idx=0, right_idx=1, nodes=example.nodes, store=store)

    assert length is not None
    assert length > 0.0


def test_semantic_weighted_tree_is_deterministic_and_renders_chunks_only() -> None:
    kwargs = dict(
        example=_example(),
        selected_medoids=[2],
        query_anchors=[0],
        distance_matrix=_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
    )

    first = semantic_weighted_support_tree_indices(**kwargs)
    second = semantic_weighted_support_tree_indices(**kwargs)

    assert first.indices == second.indices
    assert first.indices == [2, 1]
    assert first.diagnostics["positive_edge_lengths"] is True
    assert "anchor" not in first.diagnostics["semantic_weighted_order_node_ids"]
