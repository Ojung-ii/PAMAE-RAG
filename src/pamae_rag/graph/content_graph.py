from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.graph.query_graph import QueryGraph, QueryGraphEdge


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+(?:of|the|and|de|van|von|[A-Z][A-Za-z0-9]+))*\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")

_STOP_ENTITY_SURFACES = {
    "a",
    "an",
    "and",
    "he",
    "it",
    "she",
    "the",
    "they",
    "this",
    "who",
}


@dataclass(frozen=True)
class ContentEntity:
    entity_id: str
    canonical: str
    surfaces: tuple[str, ...]
    mention_count: int


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    surface: str
    node_id: str
    sentence_index: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ContentTriple:
    subject_entity_id: str
    relation: str
    object_entity_id: str
    fact_id: str


@dataclass(frozen=True)
class ContentFact:
    fact_id: str
    node_id: str
    sentence_index: int
    text: str
    entity_ids: tuple[str, ...]
    triple_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContentGraphEdge:
    source_id: str
    target_id: str
    edge_type: str


@dataclass(frozen=True)
class ContentGraphIndex:
    entities: tuple[ContentEntity, ...]
    mentions: tuple[EntityMention, ...]
    facts: tuple[ContentFact, ...]
    triples: tuple[ContentTriple, ...]
    edges: tuple[ContentGraphEdge, ...]
    diagnostics: dict[str, Any]

    @property
    def edge_counts_by_type(self) -> dict[str, int]:
        counts = Counter(edge.edge_type for edge in self.edges)
        return {key: int(counts[key]) for key in sorted(counts)}


def normalize_content_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _entity_id(canonical: str) -> str:
    return _stable_id("entity", canonical)


def _fact_id(node_id: str, sentence_index: int, text: str) -> str:
    return _stable_id("fact", f"{node_id}\n{sentence_index}\n{normalize_content_text(text)}")


def _triple_id(triple: ContentTriple) -> str:
    return _stable_id(
        "triple",
        f"{triple.subject_entity_id}\n{triple.relation}\n{triple.object_entity_id}\n{triple.fact_id}",
    )


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_RE.finditer(str(text)):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = start + len(stripped)
        spans.append((start, end, stripped))
    return spans


def _candidate_entity_spans(sentence: str, offset: int) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in (_QUOTED_RE, _CAPITALIZED_RE):
        for match in pattern.finditer(sentence):
            text = match.group(1) if pattern is _QUOTED_RE else match.group(0)
            start, end = match.span(1) if pattern is _QUOTED_RE else match.span(0)
            spans.append((offset + start, offset + end, " ".join(text.split())))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, surface in spans:
        canonical = normalize_content_text(surface)
        if not canonical or canonical in _STOP_ENTITY_SURFACES:
            continue
        if any(start < prev_end and end > prev_start for prev_start, prev_end in occupied):
            continue
        selected.append((start, end, surface))
        occupied.append((start, end))
    return selected


def _relation_text(sentence: str, left_end: int, right_start: int, sentence_offset: int) -> str:
    local_left = max(0, left_end - sentence_offset)
    local_right = max(local_left, right_start - sentence_offset)
    relation = normalize_content_text(sentence[local_left:local_right])
    return relation


def _dedupe_ordered(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def build_content_graph_index(nodes: tuple[EvidenceNode, ...] | list[EvidenceNode]) -> ContentGraphIndex:
    entity_surfaces: dict[str, set[str]] = {}
    entity_canonicals: dict[str, str] = {}
    mention_counts: Counter[str] = Counter()
    mentions: list[EntityMention] = []
    facts: list[ContentFact] = []
    triples: list[ContentTriple] = []
    edges: set[ContentGraphEdge] = set()

    for node in nodes:
        chunk_id = f"chunk:{node.node_id}"
        for sentence_index, (start, _end, sentence) in enumerate(_sentence_spans(node.text)):
            sentence_mentions: list[EntityMention] = []
            for char_start, char_end, surface in _candidate_entity_spans(sentence, start):
                canonical = normalize_content_text(surface)
                entity_id = _entity_id(canonical)
                entity_canonicals[entity_id] = canonical
                entity_surfaces.setdefault(entity_id, set()).add(surface)
                mention_counts[entity_id] += 1
                mention = EntityMention(
                    entity_id=entity_id,
                    surface=surface,
                    node_id=node.node_id,
                    sentence_index=sentence_index,
                    char_start=char_start,
                    char_end=char_end,
                )
                mentions.append(mention)
                sentence_mentions.append(mention)
                edges.add(ContentGraphEdge(chunk_id, entity_id, "chunk_entity"))

            fact_id = _fact_id(node.node_id, sentence_index, sentence)
            entity_ids = _dedupe_ordered(mention.entity_id for mention in sentence_mentions)
            local_triples: list[ContentTriple] = []
            ordered_mentions = sorted(sentence_mentions, key=lambda mention: (mention.char_start, mention.char_end))
            for left, right in zip(ordered_mentions, ordered_mentions[1:]):
                relation = _relation_text(sentence, left.char_end, right.char_start, start)
                if not relation:
                    continue
                triple = ContentTriple(
                    subject_entity_id=left.entity_id,
                    relation=relation,
                    object_entity_id=right.entity_id,
                    fact_id=fact_id,
                )
                local_triples.append(triple)
                triples.append(triple)

            facts.append(
                ContentFact(
                    fact_id=fact_id,
                    node_id=node.node_id,
                    sentence_index=sentence_index,
                    text=sentence,
                    entity_ids=entity_ids,
                    triple_ids=tuple(_triple_id(triple) for triple in local_triples),
                )
            )
            edges.add(ContentGraphEdge(chunk_id, fact_id, "chunk_fact"))
            for entity_id in entity_ids:
                edges.add(ContentGraphEdge(fact_id, entity_id, "fact_entity"))
            for pos, left in enumerate(entity_ids):
                for right in entity_ids[pos + 1 :]:
                    a, b = sorted((left, right))
                    edges.add(ContentGraphEdge(a, b, "entity_cofact"))

    entities = tuple(
        ContentEntity(
            entity_id=entity_id,
            canonical=entity_canonicals[entity_id],
            surfaces=tuple(sorted(surfaces)),
            mention_count=int(mention_counts[entity_id]),
        )
        for entity_id, surfaces in sorted(entity_surfaces.items())
    )
    edge_tuple = tuple(sorted(edges, key=lambda edge: (edge.edge_type, edge.source_id, edge.target_id)))
    edge_counts = Counter(edge.edge_type for edge in edge_tuple)
    diagnostics = {
        "content_graph_num_chunks": len(nodes),
        "content_graph_num_entities": len(entities),
        "content_graph_num_mentions": len(mentions),
        "content_graph_num_facts": len(facts),
        "content_graph_num_triples": len(triples),
        "content_graph_edge_counts_by_type": {
            key: int(edge_counts[key]) for key in sorted(edge_counts)
        },
        "content_graph_title_metadata_used": False,
    }
    return ContentGraphIndex(
        entities=entities,
        mentions=tuple(mentions),
        facts=tuple(facts),
        triples=tuple(triples),
        edges=edge_tuple,
        diagnostics=diagnostics,
    )


def _cap_query_edges(edges: Iterable[QueryGraphEdge], max_edges_per_node: int) -> tuple[QueryGraphEdge, ...]:
    if max_edges_per_node <= 0:
        return tuple()
    degree: Counter[int] = Counter()
    selected: list[QueryGraphEdge] = []
    ordered = sorted(edges, key=lambda edge: (edge.length, edge.edge_type, edge.source, edge.target))
    for edge in ordered:
        if degree[edge.source] >= max_edges_per_node or degree[edge.target] >= max_edges_per_node:
            continue
        selected.append(edge)
        degree[edge.source] += 1
        degree[edge.target] += 1
    return tuple(sorted(selected, key=lambda edge: (edge.source, edge.target, edge.length, edge.edge_type)))


def project_content_graph_to_query_graph(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    *,
    edge_lengths: dict[str, float],
    max_edges_per_node: int,
) -> tuple[QueryGraph, ContentGraphIndex, tuple[int, ...]]:
    required = {"shared_entity", "entity_fact_bridge"}
    missing = required - set(edge_lengths)
    if missing:
        raise ValueError(f"Missing content graph edge lengths: {sorted(missing)}")
    for key in required:
        if float(edge_lengths[key]) < 0:
            raise ValueError(f"Content graph edge length must be nonnegative: {key}")

    index = build_content_graph_index(nodes)
    node_position = {node.node_id: pos for pos, node in enumerate(nodes)}
    entity_to_chunks: dict[str, set[int]] = {}
    for mention in index.mentions:
        pos = node_position.get(mention.node_id)
        if pos is None:
            continue
        entity_to_chunks.setdefault(mention.entity_id, set()).add(pos)

    edges: dict[tuple[int, int], QueryGraphEdge] = {}

    def add(i: int, j: int, edge_type: str) -> None:
        if i == j:
            return
        a, b = sorted((int(i), int(j)))
        key = (a, b)
        length = float(edge_lengths[edge_type])
        prev = edges.get(key)
        if prev is None or length < prev.length:
            edges[key] = QueryGraphEdge(a, b, length, edge_type)

    for chunks in entity_to_chunks.values():
        ordered = sorted(chunks)
        for pos, left in enumerate(ordered):
            for right in ordered[pos + 1 :]:
                add(left, right, "shared_entity")

    bridge_entity_pairs = {
        tuple(sorted((triple.subject_entity_id, triple.object_entity_id)))
        for triple in index.triples
    }
    for subject_entity_id, object_entity_id in sorted(bridge_entity_pairs):
        subject_chunks = sorted(entity_to_chunks.get(subject_entity_id, set()))
        object_chunks = sorted(entity_to_chunks.get(object_entity_id, set()))
        for left in subject_chunks:
            for right in object_chunks:
                add(left, right, "entity_fact_bridge")

    capped_edges = _cap_query_edges(edges.values(), max_edges_per_node)
    counts = Counter(edge.edge_type for edge in capped_edges)
    projected_node_indices = tuple(
        sorted({edge.source for edge in capped_edges} | {edge.target for edge in capped_edges})
    )
    index.diagnostics["content_graph_unique_entity_fact_bridge_pairs"] = len(bridge_entity_pairs)
    graph = QueryGraph(
        num_nodes=len(nodes),
        edges=capped_edges,
        edge_counts_by_type={key: int(counts.get(key, 0)) for key in sorted(required)},
    )
    return graph, index, projected_node_indices
