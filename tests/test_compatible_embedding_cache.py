from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.compatible_embedding_cache import (
    CHUNK_TEXT_FORMAT,
    QUERY_TEXT_FORMAT,
    CompatibleEmbeddingCache,
    make_metadata,
)


class FakeEmbedder:
    model_id = "fake/model"
    model_revision = "abc"
    embedding_dim = 2
    normalized = True
    pooling = "fake"

    def embed_texts(self, texts):
        out = []
        for text in texts:
            out.append([1.0, 0.0] if "answer" in text or "question" in text else [0.0, 1.0])
        return np.asarray(out, dtype=float)


def _node(node_id: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        metadata={"title": f"title {node_id}"},
    )


def test_compatible_cache_records_shared_query_chunk_metadata(tmp_path) -> None:
    metadata = make_metadata(
        model_id="fake/model",
        model_revision="abc",
        embedding_dim=2,
        dataset="toy",
        pooling="fake",
        model_path="/tmp/fake",
    )
    cache = CompatibleEmbeddingCache.create(tmp_path, metadata)
    example = QueryExample(query_id="q", query="question", nodes=(_node("c1", "answer text"),))

    cache.ensure_query(example, FakeEmbedder())
    cache.ensure_chunks(example.nodes, ["c1"], FakeEmbedder())
    coverage = cache.coverage(query_ids=["q"], chunk_ids=["c1"])

    assert coverage["query_embedding_coverage"] == 1.0
    assert coverage["chunk_embedding_coverage_for_diagnostics"] == 1.0
    assert coverage["model_id"] == "fake/model"
    assert coverage["model_revision"] == "abc"
    assert coverage["embedding_dim"] == 2
    assert coverage["normalized"] is True
    assert coverage["chunk_text_format"] == CHUNK_TEXT_FORMAT
    assert coverage["query_text_format"] == QUERY_TEXT_FORMAT


def test_compatible_cache_missing_embeddings_are_logged_not_replaced(tmp_path) -> None:
    metadata = make_metadata(
        model_id="fake/model",
        model_revision="abc",
        embedding_dim=2,
        dataset="toy",
        pooling="fake",
    )
    cache = CompatibleEmbeddingCache.create(tmp_path, metadata)
    coverage = cache.coverage(query_ids=["q"], chunk_ids=["c_missing"])

    assert coverage["query_embedding_coverage"] == 0.0
    assert coverage["chunk_embedding_coverage_for_diagnostics"] == 0.0
    assert coverage["embedding_missing_count"] == 2


def test_compatible_cache_rejects_dimension_mismatch(tmp_path) -> None:
    metadata = make_metadata(
        model_id="fake/model",
        model_revision="abc",
        embedding_dim=2,
        dataset="toy",
        pooling="fake",
    )
    cache = CompatibleEmbeddingCache.create(tmp_path, metadata)

    try:
        cache.set_query("q", [1.0, 0.0, 0.0])
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch")


def test_compatible_cache_batch_lookup_logs_missing_without_replacement(tmp_path) -> None:
    metadata = make_metadata(
        model_id="fake/model",
        model_revision="abc",
        embedding_dim=2,
        dataset="toy",
        pooling="fake",
    )
    cache = CompatibleEmbeddingCache.create(tmp_path, metadata)
    cache.set_chunk("c1", [1.0, 0.0])

    found, missing = cache.get_chunks(["c1", "c_missing", "c1"])

    assert list(found) == ["c1"]
    assert missing == ["c_missing"]
    assert np.allclose(found["c1"], np.asarray([1.0, 0.0]))
