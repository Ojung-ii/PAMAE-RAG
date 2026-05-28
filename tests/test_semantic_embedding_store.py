from __future__ import annotations

import math

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.angular_distance import angular_distance, normalize_embedding
from pamae_rag.semantic.embedding_store import EmbeddingStore


def _node(node_id: str, embedding: object, *, node_type: str = "chunk") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=f"text {node_id}",
        embedding=np.asarray(embedding, dtype=float),
        node_type=node_type,
    )


def test_angular_distance_basic_metric_properties() -> None:
    x = np.asarray([1.0, 0.0, 0.0])
    y = np.asarray([0.0, 1.0, 0.0])
    z = np.asarray([0.0, 0.0, 1.0])

    assert angular_distance(x, x) == 0.0
    assert 0.0 <= angular_distance(x, y) <= 1.0
    assert angular_distance(x, y) == angular_distance(y, x)
    assert angular_distance(x, z) <= angular_distance(x, y) + angular_distance(y, z) + 1e-12


def test_angular_distance_uses_angle_not_one_minus_cosine() -> None:
    x = np.asarray([1.0, 0.0])
    y = np.asarray([0.5, math.sqrt(3.0) / 2.0])

    assert abs(angular_distance(x, y) - (1.0 / 3.0)) < 1e-12


def test_embedding_store_logs_missing_embeddings_without_random_fallback() -> None:
    example = QueryExample(
        query_id="q",
        query="question",
        nodes=(
            _node("c1", [1.0, 0.0]),
            _node("c_missing", [0.0, 0.0]),
            _node("e1", [0.0, 1.0], node_type="entity"),
        ),
        metadata={},
    )

    store = EmbeddingStore.from_example(example)
    diagnostics = store.diagnostics()

    assert diagnostics.embedding_source == "existing_node_embeddings"
    assert diagnostics.chunk_embedding_coverage == 0.5
    assert diagnostics.query_embedding_available is False
    assert diagnostics.semantic_mode_enabled is False
    assert store.node_embedding("c_missing") is None
    assert store.missing_chunk_ids == ("c_missing",)


def test_embedding_store_reads_existing_query_embedding_only() -> None:
    example = QueryExample(
        query_id="q",
        query="question",
        nodes=(_node("c1", [1.0, 0.0]),),
        metadata={"query_embedding": [2.0, 0.0]},
    )

    store = EmbeddingStore.from_example(example)

    assert store.diagnostics().semantic_mode_enabled is True
    assert np.allclose(store.query_embedding, normalize_embedding([2.0, 0.0]))
