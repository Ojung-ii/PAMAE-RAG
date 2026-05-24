from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

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


def _contains_phrase(surface: str, phrase: str) -> bool:
    if not surface or not phrase:
        return False
    return f" {phrase} " in f" {surface} "


def _candidate_edges(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    edge_lengths: dict[str, float],
) -> list[QueryGraphEdge]:
    titles = [canonical_title(_node_title(node)) for node in nodes]
    surfaces = [_node_surface(node) for node in nodes]
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

    for i, title_i in enumerate(titles):
        if not title_i:
            continue
        for j, title_j in enumerate(titles):
            if i >= j:
                continue
            if title_i == title_j:
                continue
            if _contains_phrase(surfaces[j], title_i) or _contains_phrase(surfaces[i], title_j):
                add(i, j, "title_mention")

    for span in extract_query_spans(query):
        hits = [idx for idx, surface in enumerate(surfaces) if _contains_phrase(surface, span)]
        for pos, i in enumerate(hits):
            for j in hits[pos + 1 :]:
                add(i, j, "shared_query_span")

    return sorted(edges.values(), key=lambda e: (e.source, e.target, e.length, e.edge_type))


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
) -> QueryGraph:
    required = {"same_canonical_title", "title_mention", "shared_query_span"}
    missing = required - set(edge_lengths)
    if missing:
        raise ValueError(f"Missing graph edge lengths: {sorted(missing)}")
    for key in required:
        if float(edge_lengths[key]) < 0:
            raise ValueError(f"Graph edge length must be nonnegative: {key}")
    edges = _cap_edges(_candidate_edges(nodes, query, edge_lengths), max_edges_per_node)
    counts = Counter(edge.edge_type for edge in edges)
    return QueryGraph(
        num_nodes=len(nodes),
        edges=edges,
        edge_counts_by_type={key: int(counts.get(key, 0)) for key in sorted(required)},
    )
