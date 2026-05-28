from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.angular_distance import normalize_embedding


@dataclass(frozen=True)
class EmbeddingStoreDiagnostics:
    embedding_source: str
    embedding_dim: int
    chunk_embedding_coverage: float
    query_embedding_available: bool
    semantic_mode_enabled: bool
    missing_chunk_embedding_count: int
    chunk_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_source": self.embedding_source,
            "embedding_dim": self.embedding_dim,
            "chunk_embedding_coverage": self.chunk_embedding_coverage,
            "query_embedding_available": self.query_embedding_available,
            "semantic_mode_enabled": self.semantic_mode_enabled,
            "missing_chunk_embedding_count": self.missing_chunk_embedding_count,
            "chunk_count": self.chunk_count,
        }


class EmbeddingStore:
    """Read-only embedding view over existing example data.

    The store intentionally never fabricates vectors. Missing embeddings are
    surfaced through diagnostics so semantic runs can stop safely.
    """

    def __init__(
        self,
        *,
        node_embeddings: dict[str, np.ndarray],
        query_embedding: np.ndarray | None,
        missing_chunk_ids: Sequence[str],
        chunk_count: int,
        source: str,
    ) -> None:
        self._node_embeddings = dict(node_embeddings)
        self._query_embedding = normalize_embedding(query_embedding)
        self._missing_chunk_ids = tuple(str(value) for value in missing_chunk_ids)
        self._chunk_count = int(chunk_count)
        self._source = str(source)

    @classmethod
    def from_example(cls, example: QueryExample) -> "EmbeddingStore":
        embeddings: dict[str, np.ndarray] = {}
        missing: list[str] = []
        dims: set[int] = set()
        chunk_count = 0
        for node in example.nodes:
            node_id = str(node.node_id)
            vector = normalize_embedding(getattr(node, "embedding", None))
            if vector is None:
                if str(getattr(node, "node_type", "chunk")) == "chunk":
                    missing.append(node_id)
                    chunk_count += 1
                continue
            embeddings[node_id] = vector
            dims.add(int(vector.shape[0]))
            if str(getattr(node, "node_type", "chunk")) == "chunk":
                chunk_count += 1

        query_embedding = _query_embedding_from_metadata(example.metadata)
        source = "existing_node_embeddings" if embeddings else "none"
        store = cls(
            node_embeddings=embeddings,
            query_embedding=query_embedding,
            missing_chunk_ids=missing,
            chunk_count=chunk_count,
            source=source,
        )
        store._embedding_dim_hint = min(dims) if dims else 0
        return store

    @property
    def query_embedding(self) -> np.ndarray | None:
        return None if self._query_embedding is None else self._query_embedding.copy()

    @property
    def missing_chunk_ids(self) -> tuple[str, ...]:
        return self._missing_chunk_ids

    def node_embedding(self, node_id: str) -> np.ndarray | None:
        vector = self._node_embeddings.get(str(node_id))
        return None if vector is None else vector.copy()

    def has_node_embedding(self, node_id: str) -> bool:
        return str(node_id) in self._node_embeddings

    def diagnostics(self) -> EmbeddingStoreDiagnostics:
        dim = int(getattr(self, "_embedding_dim_hint", 0))
        if dim <= 0 and self._node_embeddings:
            dim = int(next(iter(self._node_embeddings.values())).shape[0])
        covered = max(0, self._chunk_count - len(self._missing_chunk_ids))
        coverage = float(covered / self._chunk_count) if self._chunk_count else 0.0
        query_available = self._query_embedding is not None
        return EmbeddingStoreDiagnostics(
            embedding_source=self._source,
            embedding_dim=dim,
            chunk_embedding_coverage=coverage,
            query_embedding_available=query_available,
            semantic_mode_enabled=bool(coverage > 0.0 and query_available),
            missing_chunk_embedding_count=len(self._missing_chunk_ids),
            chunk_count=self._chunk_count,
        )


def _query_embedding_from_metadata(metadata: dict[str, Any]) -> np.ndarray | None:
    for key in ("query_embedding", "embedding"):
        value = metadata.get(key)
        vector = normalize_embedding(value)
        if vector is not None:
            return vector
    return None


__all__ = ["EmbeddingStore", "EmbeddingStoreDiagnostics"]
