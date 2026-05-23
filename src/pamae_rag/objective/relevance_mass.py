from __future__ import annotations

import re
from typing import Any

import numpy as np

from pamae_rag.data.schema import EvidenceNode


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _normalise_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


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
) -> np.ndarray:
    if not nodes:
        raise ValueError("nodes must not be empty")
    if mode == "current":
        return np.asarray([max(0.0, n.relevance) for n in nodes], dtype=np.float64)
    if mode == "title_aware":
        return _title_aware_scores(nodes, query)
    if mode == "diagnostic_subject_title":
        return _diagnostic_subject_title_scores(nodes, query, query_metadata)
    raise ValueError(f"Unknown relevance mode: {mode}")


def relevance_mass(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    smoothing: float = 1e-8,
    *,
    mode: str = "current",
    query: str = "",
    query_metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    scores = relevance_scores(nodes, mode=mode, query=query, query_metadata=query_metadata)
    scores = np.maximum(scores, 0.0) + smoothing
    total = float(scores.sum())
    if total <= 0:
        return np.full(len(nodes), 1.0 / len(nodes), dtype=np.float64)
    return scores / total
