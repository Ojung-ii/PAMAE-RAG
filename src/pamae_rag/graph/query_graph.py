from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable

import numpy as np

from pamae_rag.data.schema import EvidenceNode


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")
_POSSESSIVE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)'s\b")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b")

_STOP_SPANS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


@dataclass(frozen=True)
class QueryGraphEdge:
    source: int
    target: int
    length: float
    edge_type: str


@dataclass(frozen=True)
class QueryGraph:
    num_nodes: int
    edges: tuple[QueryGraphEdge, ...]
    edge_counts_by_type: dict[str, int]
    backbone_missing_embedding_count: int = 0

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def canonical_title(title: str) -> str:
    norm = normalize_text(re.sub(r"\([^)]*\)", " ", str(title)))
    prefixes = ("list of ", "lists of ")
    for prefix in prefixes:
        if norm.startswith(prefix):
            norm = norm[len(prefix) :]
    return norm.strip()


def _useful_span(span: str) -> bool:
    norm = normalize_text(span)
    if not norm or norm in _STOP_SPANS:
        return False
    return any(tok not in _STOP_SPANS for tok in norm.split())


def extract_query_spans(query: str) -> list[str]:
    spans: list[str] = []
    for pattern in (_QUOTED_RE, _POSSESSIVE_RE, _CAPITALIZED_RE):
        for match in pattern.finditer(str(query)):
            text = match.group(1) if match.lastindex else match.group(0)
            pieces = [part.strip() for part in re.split(r"\b(?:and|or|vs\.?|versus)\b", text) if part.strip()]
            spans.extend(piece for piece in pieces if _useful_span(piece))
    seen: set[str] = set()
    out: list[str] = []
    for span in spans:
        norm = normalize_text(span)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _node_title(node: EvidenceNode) -> str:
    return str(node.metadata.get("title") or node.node_id)


def _node_surface(node: EvidenceNode) -> str:
    return normalize_text(f"{_node_title(node)} {node.text}")


def _phrase_hits(padded_surfaces: list[str], phrase: str) -> list[int]:
    if not phrase:
        return []
    padded_phrase = f" {phrase} "
    return [idx for idx, surface in enumerate(padded_surfaces) if padded_phrase in surface]


def _candidate_edges(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    edge_lengths: dict[str, float],
) -> list[QueryGraphEdge]:
    titles = [canonical_title(_node_title(node)) for node in nodes]
    surfaces = [_node_surface(node) for node in nodes]
    padded_surfaces = [f" {surface} " for surface in surfaces]
    edges: dict[tuple[int, int], QueryGraphEdge] = {}

    def add(i: int, j: int, edge_type: str) -> None:
        if i == j:
            return
        a, b = sorted((int(i), int(j)))
        length = float(edge_lengths[edge_type])
        key = (a, b)
        prev = edges.get(key)
        if prev is None or (length, edge_type) < (prev.length, prev.edge_type):
            edges[key] = QueryGraphEdge(a, b, length, edge_type)

    by_title: dict[str, list[int]] = defaultdict(list)
    for idx, title in enumerate(titles):
        if title:
            by_title[title].append(idx)
    for group in by_title.values():
        if len(group) < 2:
            continue
        for pos, i in enumerate(group):
            for j in group[pos + 1 :]:
                add(i, j, "same_canonical_title")

    for title, owners in by_title.items():
        for hit in _phrase_hits(padded_surfaces, title):
            if titles[hit] == title:
                continue
            for owner in owners:
                add(owner, hit, "title_mention")

    for span in extract_query_spans(query):
        hits = _phrase_hits(padded_surfaces, span)
        for pos, i in enumerate(hits):
            for j in hits[pos + 1 :]:
                add(i, j, "shared_query_span")

    return sorted(edges.values(), key=lambda e: (e.source, e.target, e.length, e.edge_type))


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Expected mapping-like config, got {type(value).__name__}")


def _valid_embedding(node: EvidenceNode) -> bool:
    try:
        embedding = np.asarray(node.embedding, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return embedding.ndim == 1 and embedding.size > 0 and bool(np.all(np.isfinite(embedding)))


def _knn_sets(semantic_distance_matrix: np.ndarray, valid: list[int], k: int) -> dict[int, set[int]]:
    valid_arr = np.asarray(valid, dtype=np.int64)
    valid_distances = np.asarray(semantic_distance_matrix[np.ix_(valid_arr, valid_arr)], dtype=np.float64)
    out: dict[int, set[int]] = {}
    for local_idx, idx in enumerate(valid_arr):
        distances = valid_distances[local_idx]
        eligible = np.isfinite(distances) & (distances >= 0.0)
        eligible[local_idx] = False
        candidate_ids = valid_arr[eligible]
        candidate_distances = distances[eligible]
        order = np.lexsort((candidate_ids, candidate_distances))
        out[int(idx)] = {int(j) for j in candidate_ids[order[:k]]}
    return out


def _backbone_edges(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    semantic_distance_matrix: np.ndarray | None,
    backbone_config: Any | None,
) -> tuple[list[QueryGraphEdge], int]:
    cfg = _as_plain_dict(backbone_config)
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "none"))
    if not enabled or mode == "none":
        return [], 0
    if mode not in {"knn", "mutual_knn"}:
        raise ValueError(f"Unsupported graph backbone mode: {mode}")
    if str(cfg.get("length_mode", "semantic_distance")) != "semantic_distance":
        raise ValueError("Only semantic_distance graph backbone length_mode is supported")
    if semantic_distance_matrix is None:
        return [], len(nodes)

    matrix = np.asarray(semantic_distance_matrix, dtype=np.float64)
    if matrix.shape != (len(nodes), len(nodes)):
        raise ValueError("semantic_distance_matrix must match graph node count")
    valid = [idx for idx, node in enumerate(nodes) if _valid_embedding(node)]
    missing = len(nodes) - len(valid)
    if len(valid) < 2:
        return [], missing

    k = max(1, int(cfg.get("k", 4)))
    edge_type = "semantic_knn" if mode == "knn" else "mutual_semantic_knn"
    knn = _knn_sets(matrix, valid, k)
    edges: dict[tuple[int, int], QueryGraphEdge] = {}
    for i in sorted(knn):
        for j in sorted(knn[i]):
            if mode == "mutual_knn" and i not in knn.get(j, set()):
                continue
            a, b = sorted((i, j))
            if a == b:
                continue
            length = max(0.0, float(matrix[a, b]))
            key = (a, b)
            prev = edges.get(key)
            if prev is None or length < prev.length:
                edges[key] = QueryGraphEdge(a, b, length, edge_type)

    cap = int(cfg.get("max_edges_per_node", 32))
    capped = _cap_edges(edges.values(), cap)
    return list(capped), missing


def _cap_edges(edges: Iterable[QueryGraphEdge], max_edges_per_node: int) -> tuple[QueryGraphEdge, ...]:
    if max_edges_per_node <= 0:
        return tuple()
    degree: Counter[int] = Counter()
    selected: list[QueryGraphEdge] = []
    ordered = sorted(edges, key=lambda e: (e.length, e.edge_type, e.source, e.target))
    for edge in ordered:
        if degree[edge.source] >= max_edges_per_node or degree[edge.target] >= max_edges_per_node:
            continue
        selected.append(edge)
        degree[edge.source] += 1
        degree[edge.target] += 1
    return tuple(sorted(selected, key=lambda e: (e.source, e.target, e.length, e.edge_type)))


def build_minimal_query_graph(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    *,
    edge_lengths: dict[str, float],
    max_edges_per_node: int,
    semantic_distance_matrix: np.ndarray | None = None,
    backbone_config: Any | None = None,
) -> QueryGraph:
    symbolic_types = {"same_canonical_title", "title_mention", "shared_query_span"}
    backbone_types = {"semantic_knn", "mutual_semantic_knn"}
    required = symbolic_types
    missing = required - set(edge_lengths)
    if missing:
        raise ValueError(f"Missing graph edge lengths: {sorted(missing)}")
    for key in required:
        if float(edge_lengths[key]) < 0:
            raise ValueError(f"Graph edge length must be nonnegative: {key}")
    symbolic_edges = _candidate_edges(nodes, query, edge_lengths)
    backbone_edges, missing_embeddings = _backbone_edges(nodes, semantic_distance_matrix, backbone_config)
    by_pair: dict[tuple[int, int], QueryGraphEdge] = {}
    for edge in [*symbolic_edges, *backbone_edges]:
        key = (edge.source, edge.target)
        prev = by_pair.get(key)
        if prev is None or (edge.length, edge.edge_type) < (prev.length, prev.edge_type):
            by_pair[key] = edge
    edges = _cap_edges(by_pair.values(), max_edges_per_node)
    counts = Counter(edge.edge_type for edge in edges)
    return QueryGraph(
        num_nodes=len(nodes),
        edges=edges,
        edge_counts_by_type={key: int(counts.get(key, 0)) for key in sorted(symbolic_types | backbone_types)},
        backbone_missing_embedding_count=missing_embeddings,
    )
