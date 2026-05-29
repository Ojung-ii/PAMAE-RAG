from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.embedding_provenance import audit_existing_example_embeddings


def _node(node_id: str, embedding: list[float]) -> EvidenceNode:
    return EvidenceNode(node_id=node_id, text=node_id, embedding=np.asarray(embedding, dtype=float))


def test_legacy_128d_embeddings_rejected_without_matching_query_provenance() -> None:
    example = QueryExample(
        query_id="q",
        query="question",
        nodes=(_node("c1", [1.0] + [0.0] * 127),),
        metadata={},
    )

    audit = audit_existing_example_embeddings([example])

    assert audit["existing_chunk_embedding_found"] is True
    assert audit["existing_chunk_embedding_dim"] == 128
    assert audit["existing_query_embedding_found"] is False
    assert audit["existing_model_id"] is None
    assert audit["existing_cache_usable_for_semantic_query_chunk"] is False
