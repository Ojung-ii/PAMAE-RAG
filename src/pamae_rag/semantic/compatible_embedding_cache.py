from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.angular_distance import normalize_embedding
from pamae_rag.semantic.embedding_store import EmbeddingStore

CHUNK_TEXT_FORMAT = "Title: <title>\nText: <chunk_text>"
QUERY_TEXT_FORMAT = "raw_question"


class TextEmbedder(Protocol):
    embedding_dim: int
    model_id: str
    model_revision: str
    normalized: bool
    pooling: str

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CompatibleEmbeddingMetadata:
    model_id: str
    model_revision: str
    embedding_dim: int
    normalized: bool
    pooling: str
    chunk_text_format: str
    query_text_format: str
    created_at: str
    dataset: str
    model_path: str | None = None
    uses_official_model_instruction_wrapper: bool = False
    instruction_template: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalized,
            "pooling": self.pooling,
            "chunk_text_format": self.chunk_text_format,
            "query_text_format": self.query_text_format,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "model_path": self.model_path,
            "uses_official_model_instruction_wrapper": self.uses_official_model_instruction_wrapper,
            "instruction_template": self.instruction_template,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "CompatibleEmbeddingMetadata":
        return cls(
            model_id=str(obj["model_id"]),
            model_revision=str(obj["model_revision"]),
            embedding_dim=int(obj["embedding_dim"]),
            normalized=bool(obj["normalized"]),
            pooling=str(obj["pooling"]),
            chunk_text_format=str(obj["chunk_text_format"]),
            query_text_format=str(obj["query_text_format"]),
            created_at=str(obj["created_at"]),
            dataset=str(obj["dataset"]),
            model_path=str(obj["model_path"]) if obj.get("model_path") is not None else None,
            uses_official_model_instruction_wrapper=bool(obj.get("uses_official_model_instruction_wrapper", False)),
            instruction_template=(
                str(obj["instruction_template"]) if obj.get("instruction_template") is not None else None
            ),
        )


def model_slug(model_id: str) -> str:
    return str(model_id).replace("/", "__").replace("-", "_")


def default_cache_root(*, model_id: str, dataset: str) -> Path:
    return Path("outputs") / "semantic_embedding_cache" / model_slug(model_id) / dataset


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def chunk_embedding_text(node: EvidenceNode) -> str:
    title = str(node.metadata.get("title") or "")
    return f"Title: {title}\nText: {node.text}"


def query_embedding_text(example: QueryExample) -> str:
    return str(example.query)


def _safe_name(value: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return f"{digest}.npy"


def _load_index(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(key): str(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def _save_index(path: Path, index: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


class CompatibleEmbeddingCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.json"
        self.query_dir = self.root / "queries"
        self.chunk_dir = self.root / "chunks"
        self.query_index_path = self.root / "query_index.json"
        self.chunk_index_path = self.root / "chunk_index.json"
        self._metadata: CompatibleEmbeddingMetadata | None = None
        if self.metadata_path.exists():
            self._metadata = CompatibleEmbeddingMetadata.from_json(
                json.loads(self.metadata_path.read_text(encoding="utf-8"))
            )

    @classmethod
    def create(cls, root: Path, metadata: CompatibleEmbeddingMetadata) -> "CompatibleEmbeddingCache":
        cache = cls(root)
        cache.root.mkdir(parents=True, exist_ok=True)
        if cache.metadata_path.exists():
            existing = cache.metadata
            existing_json = existing.to_json()
            requested_json = metadata.to_json()
            existing_json.pop("created_at", None)
            requested_json.pop("created_at", None)
            if existing_json != requested_json:
                raise ValueError(f"cache metadata mismatch at {cache.metadata_path}")
        else:
            cache.metadata_path.write_text(json.dumps(metadata.to_json(), indent=2, sort_keys=True), encoding="utf-8")
            cache._metadata = metadata
        cache.query_dir.mkdir(parents=True, exist_ok=True)
        cache.chunk_dir.mkdir(parents=True, exist_ok=True)
        return cache

    @property
    def metadata(self) -> CompatibleEmbeddingMetadata:
        if self._metadata is None:
            raise FileNotFoundError(f"Missing compatible embedding metadata: {self.metadata_path}")
        return self._metadata

    def _validate_vector(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        normalized = normalize_embedding(vector)
        if normalized is None:
            raise ValueError("embedding vector is missing or invalid")
        if normalized.shape[0] != self.metadata.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: {normalized.shape[0]} != {self.metadata.embedding_dim}"
            )
        return normalized.astype(np.float32)

    def _get(self, *, key: str, kind: str) -> np.ndarray | None:
        index_path = self.query_index_path if kind == "query" else self.chunk_index_path
        base_dir = self.query_dir if kind == "query" else self.chunk_dir
        index = _load_index(index_path)
        rel = index.get(str(key))
        if rel is None:
            return None
        path = base_dir / rel
        if not path.exists():
            return None
        return self._validate_vector(np.load(path))

    def _set(self, *, key: str, vector: Sequence[float] | np.ndarray, kind: str) -> None:
        index_path = self.query_index_path if kind == "query" else self.chunk_index_path
        base_dir = self.query_dir if kind == "query" else self.chunk_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        index = _load_index(index_path)
        name = index.get(str(key), _safe_name(str(key)))
        np.save(base_dir / name, self._validate_vector(vector))
        index[str(key)] = name
        _save_index(index_path, index)

    def get_query(self, query_id: str) -> np.ndarray | None:
        return self._get(key=query_id, kind="query")

    def set_query(self, query_id: str, vector: Sequence[float] | np.ndarray) -> None:
        self._set(key=query_id, vector=vector, kind="query")

    def get_chunk(self, node_id: str) -> np.ndarray | None:
        return self._get(key=node_id, kind="chunk")

    def set_chunk(self, node_id: str, vector: Sequence[float] | np.ndarray) -> None:
        self._set(key=node_id, vector=vector, kind="chunk")

    def ensure_query(self, example: QueryExample, embedder: TextEmbedder) -> None:
        if self.get_query(example.query_id) is not None:
            return
        vector = embedder.embed_texts([query_embedding_text(example)])[0]
        self.set_query(example.query_id, vector)

    def ensure_chunks(self, nodes: Sequence[EvidenceNode], node_ids: Iterable[str], embedder: TextEmbedder) -> None:
        by_id = {str(node.node_id): node for node in nodes}
        missing_ids = [str(node_id) for node_id in dict.fromkeys(node_ids) if self.get_chunk(str(node_id)) is None]
        missing_nodes = [by_id[node_id] for node_id in missing_ids if node_id in by_id]
        if not missing_nodes:
            return
        vectors = embedder.embed_texts([chunk_embedding_text(node) for node in missing_nodes])
        for node, vector in zip(missing_nodes, vectors, strict=True):
            self.set_chunk(str(node.node_id), vector)

    def embedding_store_for_example(self, example: QueryExample, node_ids: Iterable[str]) -> EmbeddingStore:
        node_embeddings: dict[str, np.ndarray] = {}
        missing: list[str] = []
        requested = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        for node_id in requested:
            vector = self.get_chunk(node_id)
            if vector is None:
                missing.append(node_id)
            else:
                node_embeddings[node_id] = vector
        return EmbeddingStore(
            node_embeddings=node_embeddings,
            query_embedding=self.get_query(example.query_id),
            missing_chunk_ids=missing,
            chunk_count=len(requested),
            source=f"compatible_cache:{self.metadata.model_id}",
        )

    def coverage(self, *, query_ids: Iterable[str], chunk_ids: Iterable[str]) -> dict[str, Any]:
        query_ids = list(dict.fromkeys(str(value) for value in query_ids))
        chunk_ids = list(dict.fromkeys(str(value) for value in chunk_ids))
        query_found = sum(1 for query_id in query_ids if self.get_query(query_id) is not None)
        chunk_found = sum(1 for chunk_id in chunk_ids if self.get_chunk(chunk_id) is not None)
        return {
            "query_embedding_coverage": float(query_found / len(query_ids)) if query_ids else 0.0,
            "chunk_embedding_coverage_for_diagnostics": float(chunk_found / len(chunk_ids)) if chunk_ids else 0.0,
            "embedding_missing_count": int((len(query_ids) - query_found) + (len(chunk_ids) - chunk_found)),
            "embedding_dim_match": True,
            "all_vectors_l2_normalized": True,
            "embedding_dim": self.metadata.embedding_dim,
            "model_id": self.metadata.model_id,
            "model_revision": self.metadata.model_revision,
            "normalized": self.metadata.normalized,
            "pooling": self.metadata.pooling,
            "chunk_text_format": self.metadata.chunk_text_format,
            "query_text_format": self.metadata.query_text_format,
        }


def cache_from_env() -> CompatibleEmbeddingCache | None:
    value = os.environ.get("PAMAE_SEMANTIC_CACHE_DIR")
    if not value:
        return None
    path = Path(value)
    if not (path / "metadata.json").exists():
        return None
    return CompatibleEmbeddingCache(path)


def make_metadata(
    *,
    model_id: str,
    model_revision: str,
    embedding_dim: int,
    dataset: str,
    pooling: str = "nv_embed_encode",
    model_path: str | None = None,
) -> CompatibleEmbeddingMetadata:
    return CompatibleEmbeddingMetadata(
        model_id=model_id,
        model_revision=model_revision,
        embedding_dim=int(embedding_dim),
        normalized=True,
        pooling=pooling,
        chunk_text_format=CHUNK_TEXT_FORMAT,
        query_text_format=QUERY_TEXT_FORMAT,
        created_at=now_utc_iso(),
        dataset=dataset,
        model_path=model_path,
        uses_official_model_instruction_wrapper=False,
        instruction_template=None,
    )


__all__ = [
    "CHUNK_TEXT_FORMAT",
    "QUERY_TEXT_FORMAT",
    "CompatibleEmbeddingCache",
    "CompatibleEmbeddingMetadata",
    "TextEmbedder",
    "cache_from_env",
    "chunk_embedding_text",
    "default_cache_root",
    "make_metadata",
    "model_slug",
    "query_embedding_text",
]
