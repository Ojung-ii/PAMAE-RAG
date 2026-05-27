from __future__ import annotations

import re
from typing import Any

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.objective.query_grounding import (
    entity_title_grounding_score,
    extract_query_title_spans,
    normalize_text,
    title_overlap_score,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _normalise_text(text: str) -> str:
    return normalize_text(text)


def _node_title(node: EvidenceNode) -> str:
    return str(node.metadata.get("title") or "")


def _lexical_overlap(query: str, text: str) -> float:
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokens(text)
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _normalised_title_match(query: str, title: str) -> float:
    q_norm = _normalise_text(query)
    t_norm = _normalise_text(title)
    if not q_norm or not t_norm:
        return 0.0
    if q_norm == t_norm:
        return 1.0
    if t_norm in q_norm:
        return 0.85
    if q_norm in t_norm:
        return 0.65
    return 0.0


def _metadata_value(metadata: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(metadata, dict):
        return None
    if key in metadata:
        return metadata[key]
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _title_aware_scores(nodes: tuple[EvidenceNode, ...] | list[EvidenceNode], query: str) -> np.ndarray:
    scores = []
    for node in nodes:
        title = _node_title(node)
        title_overlap = _lexical_overlap(query, title)
        title_match = _normalised_title_match(query, title)
        body_relevance = _lexical_overlap(query, f"{title} {node.text}")
        score = 0.30 * title_overlap + 0.25 * title_match + 0.45 * body_relevance
        scores.append(max(0.0, score))
    return np.asarray(scores, dtype=np.float64)


def _weights(values: dict[str, float] | None, defaults: dict[str, float]) -> dict[str, float]:
    out = dict(defaults)
    if values:
        for key, value in values.items():
            if key in out:
                out[key] = float(value)
    return out


def _query_embedding(query_metadata: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(query_metadata, dict):
        return None
    value = query_metadata.get("query_embedding")
    if value is None:
        value = query_metadata.get("embedding")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm


def _semantic_scores(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query_metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, bool]:
    q_emb = _query_embedding(query_metadata)
    if q_emb is None:
        return np.zeros(len(nodes), dtype=np.float64), False
    scores: list[float] = []
    for node in nodes:
        emb = np.asarray(node.embedding, dtype=np.float64)
        if emb.shape != q_emb.shape:
            scores.append(0.0)
            continue
        norm = float(np.linalg.norm(emb))
        if norm <= 0:
            scores.append(0.0)
            continue
        cosine = float(np.dot(q_emb, emb / norm))
        scores.append(max(0.0, (cosine + 1.0) / 2.0))
    return np.asarray(scores, dtype=np.float64), True


def _entity_title_scores(nodes: tuple[EvidenceNode, ...] | list[EvidenceNode], query: str) -> np.ndarray:
    return np.asarray(
        [entity_title_grounding_score(query, _node_title(node)) for node in nodes],
        dtype=np.float64,
    )


def _combined_scores(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    query_metadata: dict[str, Any] | None,
    weights: dict[str, float] | None,
    defaults: dict[str, float],
    *,
    include_semantic: bool,
) -> tuple[np.ndarray, bool]:
    w = _weights(weights, defaults)
    lexical = np.asarray([_lexical_overlap(query, f"{_node_title(node)} {node.text}") for node in nodes])
    title = np.asarray([title_overlap_score(query, _node_title(node)) for node in nodes])
    entity_title = _entity_title_scores(nodes, query)
    semantic, semantic_available = _semantic_scores(nodes, query_metadata) if include_semantic else (
        np.zeros(len(nodes), dtype=np.float64),
        False,
    )
    scores = (
        w.get("lexical", 0.0) * lexical
        + w.get("title", 0.0) * title
        + w.get("entity_title", 0.0) * entity_title
        + w.get("semantic", 0.0) * semantic
    )
    return np.maximum(scores, 0.0), semantic_available


def _diagnostic_subject_title_scores(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    query_metadata: dict[str, Any] | None,
) -> np.ndarray:
    subject = _metadata_value(query_metadata, "s_wiki_title")
    if subject is None:
        subject = _metadata_value(query_metadata, "subj")
    base = _title_aware_scores(nodes, query)
    if subject is None:
        return base
    subject_norm = _normalise_text(str(subject))
    subject_scores = []
    for node in nodes:
        title_norm = _normalise_text(_node_title(node))
        subject_scores.append(1.0 if subject_norm and title_norm == subject_norm else 0.0)
    subject_arr = np.asarray(subject_scores, dtype=np.float64)
    return 0.30 * base + 0.70 * subject_arr


def relevance_scores(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    *,
    mode: str = "current",
    query: str = "",
    query_metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    if not nodes:
        raise ValueError("nodes must not be empty")
    if mode == "current":
        return np.asarray([max(0.0, n.relevance) for n in nodes], dtype=np.float64)
    if mode == "title_aware":
        return _title_aware_scores(nodes, query)
    if mode == "entity_title_aware":
        scores, _ = _combined_scores(
            nodes,
            query,
            query_metadata,
            weights,
            {"lexical": 0.35, "title": 0.25, "entity_title": 0.40, "semantic": 0.0},
            include_semantic=False,
        )
        return scores
    if mode == "hybrid_title_semantic":
        semantic_available = _semantic_scores(nodes, query_metadata)[1]
        if not semantic_available:
            return _title_aware_scores(nodes, query)
        scores, _ = _combined_scores(
            nodes,
            query,
            query_metadata,
            weights,
            {"lexical": 0.25, "title": 0.20, "entity_title": 0.25, "semantic": 0.30},
            include_semantic=True,
        )
        return scores
    if mode == "diagnostic_subject_title":
        return _diagnostic_subject_title_scores(nodes, query, query_metadata)
    raise ValueError(f"Unknown relevance mode: {mode}")


def normalize_relevance_scores(scores: np.ndarray, smoothing: float = 1e-8) -> np.ndarray:
    scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0) + smoothing
    total = float(scores.sum())
    if total <= 0:
        return np.full(scores.shape[0], 1.0 / scores.shape[0], dtype=np.float64)
    return scores / total


def relevance_mass(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    smoothing: float = 1e-8,
    *,
    mode: str = "current",
    query: str = "",
    query_metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    scores = relevance_scores(nodes, mode=mode, query=query, query_metadata=query_metadata, weights=weights)
    return normalize_relevance_scores(scores, smoothing=smoothing)


def relevance_diagnostics(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    *,
    mode: str,
    query: str,
    query_metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    top_k: int = 5,
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    if scores is None:
        scores = relevance_scores(nodes, mode=mode, query=query, query_metadata=query_metadata, weights=weights)
    else:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(nodes),):
            raise ValueError("precomputed relevance scores must match node count")
    semantic_available = bool(_semantic_scores(nodes, query_metadata)[1]) if mode == "hybrid_title_semantic" else False
    ranked = sorted(range(len(nodes)), key=lambda i: (-float(scores[int(i)]), int(i)))[:top_k]
    return {
        "relevance_mode": mode,
        "relevance_weights": dict(weights or {}),
        "semantic_component_available": semantic_available,
        "query_title_spans": extract_query_title_spans(query),
        "top_relevance_node_ids": [nodes[int(i)].node_id for i in ranked],
    }
