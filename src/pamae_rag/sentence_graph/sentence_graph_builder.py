from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.graph_variants import (
    ENTITY_SENTENCE,
    ENTITY_SENTENCE_CHUNK_HIER,
    validate_sentence_graph_variant,
)
from pamae_rag.sentence_graph.sentence_splitter import (
    SentenceEvidenceNode,
    build_sentence_nodes,
    unique_ordered,
)


@dataclass(frozen=True)
class SentenceGraphEdge:
    source_id: str
    target_id: str
    edge_type: str
    length: float = 1.0
    include_in_metric: bool = True

    def endpoints(self) -> tuple[str, str]:
        return tuple(sorted((self.source_id, self.target_id)))


@dataclass(frozen=True)
class ChunkParentNode:
    chunk_node_id: str
    doc_id: str
    chunk_id: str
    title: str
    chunk_text: str
    token_count: int
    metadata: dict[str, Any]

    def to_context_node(self) -> dict[str, Any]:
        return {
            "node_id": self.chunk_id,
            "text": self.chunk_text,
            "token_count": self.token_count,
            "metadata": {
                "title": self.title,
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "context_unit": "parent_chunk",
                **self.metadata,
            },
        }


@dataclass(frozen=True)
class SentenceGraphIndex:
    graph_variant: str
    sentence_nodes: tuple[SentenceEvidenceNode, ...]
    entity_ids: tuple[str, ...]
    entity_canonicals: dict[str, str]
    chunk_nodes: tuple[ChunkParentNode, ...]
    edges: tuple[SentenceGraphEdge, ...]
    diagnostics: dict[str, Any]

    @property
    def sentence_ids(self) -> tuple[str, ...]:
        return tuple(sentence.sentence_id for sentence in self.sentence_nodes)

    @property
    def sentence_by_id(self) -> dict[str, SentenceEvidenceNode]:
        return {sentence.sentence_id: sentence for sentence in self.sentence_nodes}

    @property
    def sentence_position(self) -> dict[str, int]:
        return {sentence.sentence_id: idx for idx, sentence in enumerate(self.sentence_nodes)}

    @property
    def chunk_by_id(self) -> dict[str, ChunkParentNode]:
        return {chunk.chunk_id: chunk for chunk in self.chunk_nodes}

    @property
    def sentences_by_chunk(self) -> dict[str, tuple[SentenceEvidenceNode, ...]]:
        grouped: dict[str, list[SentenceEvidenceNode]] = defaultdict(list)
        for sentence in self.sentence_nodes:
            grouped[sentence.chunk_id].append(sentence)
        return {
            chunk_id: tuple(
                sorted(items, key=lambda s: (s.sentence_index_in_chunk, s.sentence_id))
            )
            for chunk_id, items in grouped.items()
        }

    @property
    def entity_by_canonical(self) -> dict[str, str]:
        return {canonical: entity_id for entity_id, canonical in self.entity_canonicals.items()}

    def adjacency(
        self,
        *,
        include_chunk_parent_edges: bool = False,
        metric_only: bool = True,
    ) -> dict[str, list[tuple[str, float, str]]]:
        adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        for edge in self.edges:
            if metric_only and not edge.include_in_metric:
                if not include_chunk_parent_edges:
                    continue
            if edge.edge_type == "sent_chunk_parent" and not include_chunk_parent_edges:
                continue
            adjacency[edge.source_id].append((edge.target_id, float(edge.length), edge.edge_type))
            adjacency[edge.target_id].append((edge.source_id, float(edge.length), edge.edge_type))
        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: (item[1], item[2], item[0]))
        return dict(adjacency)

    @property
    def edge_counts_by_type(self) -> dict[str, int]:
        counts = Counter(edge.edge_type for edge in self.edges)
        return {key: int(counts[key]) for key in sorted(counts)}


def _edge(source_id: str, target_id: str, edge_type: str, *, include_in_metric: bool = True) -> SentenceGraphEdge:
    a, b = sorted((source_id, target_id))
    return SentenceGraphEdge(a, b, edge_type, 1.0, include_in_metric)


def _chunk_node_id(doc_id: str, chunk_id: str) -> str:
    return f"chunk:{doc_id}:{chunk_id}"


def _build_chunk_nodes(sentences: Sequence[SentenceEvidenceNode]) -> tuple[ChunkParentNode, ...]:
    by_chunk: dict[str, SentenceEvidenceNode] = {}
    for sentence in sentences:
        by_chunk.setdefault(sentence.chunk_id, sentence)
    chunks: list[ChunkParentNode] = []
    for chunk_id, sentence in sorted(by_chunk.items()):
        chunks.append(
            ChunkParentNode(
                chunk_node_id=_chunk_node_id(sentence.doc_id, sentence.chunk_id),
                doc_id=sentence.doc_id,
                chunk_id=sentence.chunk_id,
                title=sentence.title,
                chunk_text=sentence.chunk_text,
                token_count=max(1, len(sentence.chunk_text.split())),
                metadata=dict(sentence.source_metadata),
            )
        )
    return tuple(chunks)


def _isolated_sentence_rate(
    sentences: Sequence[SentenceEvidenceNode],
    edges: Iterable[SentenceGraphEdge],
) -> float:
    degree = Counter()
    metric_edges = [edge for edge in edges if edge.include_in_metric]
    for edge in metric_edges:
        degree[edge.source_id] += 1
        degree[edge.target_id] += 1
    if not sentences:
        return 0.0
    isolated = sum(1 for sentence in sentences if degree[sentence.sentence_id] == 0)
    return float(isolated / len(sentences))


def build_sentence_graph_index(
    nodes: Sequence[EvidenceNode],
    *,
    graph_variant: str,
    use_chunk_parent_edges_in_metric: bool = False,
) -> SentenceGraphIndex:
    variant = validate_sentence_graph_variant(graph_variant)
    sentence_nodes = build_sentence_nodes(nodes)
    entity_canonicals: dict[str, str] = {}
    edges: set[SentenceGraphEdge] = set()

    for sentence in sentence_nodes:
        for entity_id in sentence.entities:
            canonical = entity_id.removeprefix("entity:").replace("_", " ")
            entity_canonicals[entity_id] = canonical
            edges.add(_edge(entity_id, sentence.sentence_id, "entity_sentence_mention"))

    sentences_by_chunk: dict[str, list[SentenceEvidenceNode]] = defaultdict(list)
    for sentence in sentence_nodes:
        sentences_by_chunk[sentence.chunk_id].append(sentence)
    for group in sentences_by_chunk.values():
        ordered = sorted(group, key=lambda s: (s.sentence_index_in_chunk, s.sentence_id))
        for left, right in zip(ordered, ordered[1:]):
            edges.add(_edge(left.sentence_id, right.sentence_id, "sent_adjacent"))

    chunk_nodes: tuple[ChunkParentNode, ...] = tuple()
    if variant == ENTITY_SENTENCE_CHUNK_HIER:
        chunk_nodes = _build_chunk_nodes(sentence_nodes)
        chunk_by_id = {chunk.chunk_id: chunk.chunk_node_id for chunk in chunk_nodes}
        for sentence in sentence_nodes:
            edges.add(
                _edge(
                    sentence.sentence_id,
                    chunk_by_id[sentence.chunk_id],
                    "sent_chunk_parent",
                    include_in_metric=bool(use_chunk_parent_edges_in_metric),
                )
            )

    edge_tuple = tuple(sorted(edges, key=lambda edge: (edge.edge_type, edge.source_id, edge.target_id)))
    entity_ids = tuple(sorted(entity_canonicals))
    avg_sentences_per_chunk = (
        len(sentence_nodes) / max(len(sentences_by_chunk), 1) if sentence_nodes else 0.0
    )
    avg_entities_per_sentence = (
        sum(len(sentence.entities) for sentence in sentence_nodes) / max(len(sentence_nodes), 1)
    )
    diagnostics = {
        "graph_variant": variant,
        "num_sentence_nodes": len(sentence_nodes),
        "num_entity_nodes": len(entity_ids),
        "num_chunk_parent_nodes": len(chunk_nodes),
        "edge_counts_by_type": {
            key: int(value) for key, value in Counter(edge.edge_type for edge in edge_tuple).items()
        },
        "avg_sentences_per_chunk": float(avg_sentences_per_chunk),
        "avg_entities_per_sentence": float(avg_entities_per_sentence),
        "isolated_sentence_rate": _isolated_sentence_rate(sentence_nodes, edge_tuple),
        "use_chunk_parent_edges_in_metric": bool(use_chunk_parent_edges_in_metric),
        "chunk_parent_edges_are_metric_edges": any(
            edge.edge_type == "sent_chunk_parent" and edge.include_in_metric for edge in edge_tuple
        ),
    }
    return SentenceGraphIndex(
        graph_variant=variant,
        sentence_nodes=sentence_nodes,
        entity_ids=entity_ids,
        entity_canonicals=entity_canonicals,
        chunk_nodes=chunk_nodes,
        edges=edge_tuple,
        diagnostics=diagnostics,
    )


def sentence_ids_for_chunks(index: SentenceGraphIndex, chunk_ids: Iterable[str]) -> tuple[str, ...]:
    chunks = set(str(chunk_id) for chunk_id in chunk_ids)
    return unique_ordered(
        sentence.sentence_id for sentence in index.sentence_nodes if sentence.chunk_id in chunks
    )
