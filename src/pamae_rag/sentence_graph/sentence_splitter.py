from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.graph.content_graph import normalize_content_text

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")
_CAPITALIZED_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]+(?:\s+(?:of|the|and|de|van|von|[A-Z][A-Za-z0-9]+))*\b"
)

_STOP_ENTITY_SURFACES = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "from",
    "he",
    "in",
    "is",
    "it",
    "of",
    "on",
    "she",
    "the",
    "they",
    "this",
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
class SentenceSpan:
    sentence_index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SentenceEntity:
    entity_id: str
    canonical: str
    surface: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SentenceEvidenceNode:
    sentence_id: str
    doc_id: str
    chunk_id: str
    title: str
    sentence_text: str
    chunk_text: str
    sentence_index_in_chunk: int
    sentence_index_in_doc: int
    entities: tuple[str, ...]
    entity_surfaces: tuple[str, ...]
    token_count: int
    source_relevance: float
    source_metadata: dict[str, Any]

    def to_context_node(self, *, text: str | None = None, node_id: str | None = None) -> dict[str, Any]:
        rendered = self.sentence_text if text is None else text
        return {
            "node_id": self.sentence_id if node_id is None else node_id,
            "text": rendered,
            "token_count": max(1, len(rendered.split())),
            "metadata": {
                "title": self.title,
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "sentence_id": self.sentence_id,
                "sentence_index_in_chunk": self.sentence_index_in_chunk,
                "sentence_index_in_doc": self.sentence_index_in_doc,
                "context_unit": "sentence",
            },
        }

    def to_log_json(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "sentence_text": self.sentence_text,
            "chunk_text": self.chunk_text,
            "sentence_index_in_chunk": self.sentence_index_in_chunk,
            "sentence_index_in_doc": self.sentence_index_in_doc,
            "entities": list(self.entities),
        }


def sentence_spans(text: str) -> tuple[SentenceSpan, ...]:
    spans: list[SentenceSpan] = []
    for match in _SENTENCE_RE.finditer(str(text)):
        raw = match.group(0)
        stripped = " ".join(raw.split())
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        start = match.start() + leading
        spans.append(
            SentenceSpan(
                sentence_index=len(spans),
                start=start,
                end=start + len(stripped),
                text=stripped,
            )
        )
    return tuple(spans)


def _entity_id(canonical: str) -> str:
    return f"entity:{canonical.replace(' ', '_')}"


def _candidate_entity_spans(sentence: str, offset: int = 0) -> list[tuple[int, int, str]]:
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


def extract_sentence_entities(sentence: str, offset: int = 0) -> tuple[SentenceEntity, ...]:
    entities: list[SentenceEntity] = []
    seen: set[str] = set()
    for start, end, surface in _candidate_entity_spans(sentence, offset):
        canonical = normalize_content_text(surface)
        if canonical in seen:
            continue
        seen.add(canonical)
        entities.append(
            SentenceEntity(
                entity_id=_entity_id(canonical),
                canonical=canonical,
                surface=surface,
                char_start=start,
                char_end=end,
            )
        )
    return tuple(entities)


def extract_query_entities(query: str) -> tuple[SentenceEntity, ...]:
    return extract_sentence_entities(str(query), 0)


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _doc_id(node: EvidenceNode) -> str:
    value = _metadata_value(node.metadata, "doc_id")
    if value is None:
        value = _metadata_value(node.metadata, "corpus_index")
    return str(value if value is not None else node.node_id)


def _title(node: EvidenceNode) -> str:
    return str(_metadata_value(node.metadata, "title") or node.node_id)


def build_sentence_nodes(nodes: Sequence[EvidenceNode]) -> tuple[SentenceEvidenceNode, ...]:
    doc_offsets: dict[str, int] = {}
    sentence_nodes: list[SentenceEvidenceNode] = []
    for node in nodes:
        doc_id = _doc_id(node)
        title = _title(node)
        chunk_id = str(node.node_id)
        spans = sentence_spans(node.text)
        for span in spans:
            sentence_index_in_doc = doc_offsets.get(doc_id, 0)
            entities = extract_sentence_entities(span.text, span.start)
            entity_ids = tuple(entity.entity_id for entity in entities)
            sentence_nodes.append(
                SentenceEvidenceNode(
                    sentence_id=f"sent:{doc_id}:{chunk_id}:{span.sentence_index}",
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    title=title,
                    sentence_text=span.text,
                    chunk_text=str(node.text),
                    sentence_index_in_chunk=span.sentence_index,
                    sentence_index_in_doc=sentence_index_in_doc,
                    entities=entity_ids,
                    entity_surfaces=tuple(entity.surface for entity in entities),
                    token_count=max(1, len(span.text.split())),
                    source_relevance=float(node.relevance),
                    source_metadata=dict(node.metadata),
                )
            )
            doc_offsets[doc_id] = sentence_index_in_doc + 1
    return tuple(sentence_nodes)


def unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))
