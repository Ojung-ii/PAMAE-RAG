from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.sentence_splitter import (
    SentenceEvidenceNode,
    build_sentence_nodes,
    unique_ordered,
)


@dataclass(frozen=True)
class LocalFactNode:
    fact_id: str
    chunk_id: str
    sentence_id: str
    sentence_index_in_chunk: int
    text: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocalSurfaceEdge:
    source: str
    target: str
    edge_type: str
    length: float = 1.0


@dataclass(frozen=True)
class LocalSurfaceGraph:
    selected_chunks: tuple[EvidenceNode, ...]
    sentences: tuple[SentenceEvidenceNode, ...]
    facts: tuple[LocalFactNode, ...]
    entity_ids: tuple[str, ...]
    edges: tuple[LocalSurfaceEdge, ...]
    adjacency: dict[str, tuple[tuple[str, float, str], ...]]
    diagnostics: dict[str, Any]

    @property
    def sentence_ids(self) -> tuple[str, ...]:
        return tuple(sentence.sentence_id for sentence in self.sentences)

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return tuple(fact.fact_id for fact in self.facts)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return (*self.sentence_ids, *self.entity_ids, *self.fact_ids)

    @property
    def sentence_by_id(self) -> dict[str, SentenceEvidenceNode]:
        return {sentence.sentence_id: sentence for sentence in self.sentences}

    @property
    def fact_by_id(self) -> dict[str, LocalFactNode]:
        return {fact.fact_id: fact for fact in self.facts}


def _selected_nodes(nodes: Sequence[EvidenceNode], selected_chunk_ids: Iterable[str]) -> tuple[EvidenceNode, ...]:
    by_id = {str(node.node_id): node for node in nodes}
    out: list[EvidenceNode] = []
    seen: set[str] = set()
    for node_id in selected_chunk_ids:
        key = str(node_id)
        if key in seen or key not in by_id:
            continue
        seen.add(key)
        out.append(by_id[key])
    return tuple(out)


def _add_edge(edges: set[LocalSurfaceEdge], source: str, target: str, edge_type: str) -> None:
    if source == target:
        return
    a, b = sorted((str(source), str(target)))
    edges.add(LocalSurfaceEdge(a, b, edge_type, 1.0))


def _adjacency(edges: Iterable[LocalSurfaceEdge]) -> dict[str, tuple[tuple[str, float, str], ...]]:
    raw: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for edge in edges:
        raw[edge.source].append((edge.target, float(edge.length), edge.edge_type))
        raw[edge.target].append((edge.source, float(edge.length), edge.edge_type))
    return {
        node_id: tuple(sorted(neighbors, key=lambda item: (item[1], item[2], item[0])))
        for node_id, neighbors in raw.items()
    }


def build_local_surface_graph(
    nodes: Sequence[EvidenceNode],
    selected_chunk_ids: Iterable[str],
) -> LocalSurfaceGraph:
    selected_chunks = _selected_nodes(nodes, selected_chunk_ids)
    sentences = build_sentence_nodes(selected_chunks)
    edges: set[LocalSurfaceEdge] = set()
    entity_ids = unique_ordered(entity_id for sentence in sentences for entity_id in sentence.entities)
    facts: list[LocalFactNode] = []

    sentence_by_chunk: dict[str, list[SentenceEvidenceNode]] = defaultdict(list)
    for sentence in sentences:
        sentence_by_chunk[sentence.chunk_id].append(sentence)
        for entity_id in sentence.entities:
            _add_edge(edges, entity_id, sentence.sentence_id, "entity_sentence")
        fact_id = f"fact:{sentence.chunk_id}:{sentence.sentence_index_in_chunk}"
        facts.append(
            LocalFactNode(
                fact_id=fact_id,
                chunk_id=sentence.chunk_id,
                sentence_id=sentence.sentence_id,
                sentence_index_in_chunk=sentence.sentence_index_in_chunk,
                text=sentence.sentence_text,
                entity_ids=sentence.entities,
            )
        )
        _add_edge(edges, sentence.sentence_id, fact_id, "sentence_fact")
        for entity_id in sentence.entities:
            _add_edge(edges, fact_id, entity_id, "fact_entity")

    for chunk_sentences in sentence_by_chunk.values():
        ordered = sorted(chunk_sentences, key=lambda sentence: sentence.sentence_index_in_chunk)
        for left, right in zip(ordered, ordered[1:]):
            _add_edge(edges, left.sentence_id, right.sentence_id, "sent_adjacent")

    edge_tuple = tuple(sorted(edges, key=lambda edge: (edge.edge_type, edge.source, edge.target)))
    adjacency = _adjacency(edge_tuple)
    edge_lengths_positive = all(float(edge.length) > 0 for edge in edge_tuple)
    diagnostics = {
        "selected_chunk_count": len(selected_chunks),
        "local_sentence_count": len(sentences),
        "local_entity_count": len(entity_ids),
        "local_fact_count": len(facts),
        "local_edge_count": len(edge_tuple),
        "edge_lengths_positive": edge_lengths_positive,
    }
    return LocalSurfaceGraph(
        selected_chunks=selected_chunks,
        sentences=sentences,
        facts=tuple(facts),
        entity_ids=entity_ids,
        edges=edge_tuple,
        adjacency=adjacency,
        diagnostics=diagnostics,
    )


def shortest_path(
    graph: LocalSurfaceGraph,
    source: str,
    target: str,
) -> tuple[str, ...]:
    source = str(source)
    target = str(target)
    if source == target:
        return (source,)
    queue: deque[str] = deque([source])
    parent: dict[str, str | None] = {source: None}
    while queue:
        node_id = queue.popleft()
        for neighbor, _length, _edge_type in graph.adjacency.get(node_id, ()):
            if neighbor in parent:
                continue
            parent[neighbor] = node_id
            if neighbor == target:
                path = [target]
                cur = target
                while parent[cur] is not None:
                    cur = parent[cur] or source
                    path.append(cur)
                return tuple(reversed(path))
            queue.append(neighbor)
    return tuple()


def reachable_nodes(graph: LocalSurfaceGraph, sources: Iterable[str]) -> frozenset[str]:
    seen: set[str] = set()
    queue: deque[str] = deque()
    for source in sources:
        node_id = str(source)
        if node_id in seen:
            continue
        seen.add(node_id)
        queue.append(node_id)
    while queue:
        node_id = queue.popleft()
        for neighbor, _length, _edge_type in graph.adjacency.get(node_id, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return frozenset(seen)


__all__ = [
    "LocalFactNode",
    "LocalSurfaceEdge",
    "LocalSurfaceGraph",
    "build_local_surface_graph",
    "reachable_nodes",
    "shortest_path",
]
