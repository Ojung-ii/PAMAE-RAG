from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.eval.support_facts import resolve_support_facts, support_facts_from_metadata
from pamae_rag.qa.metrics import gold_answers, normalize_answer
from pamae_rag.sentence_graph.sentence_splitter import SentenceEvidenceNode, build_sentence_nodes


@dataclass(frozen=True)
class SurfaceSentenceSets:
    all_sentences: tuple[SentenceEvidenceNode, ...]
    answer_sentence_ids: frozenset[str]
    gold_sentence_ids: frozenset[str]


def _contains_answer(text: str, answers: Iterable[str]) -> bool:
    text_norm = normalize_answer(text)
    if not text_norm:
        return False
    padded = f" {text_norm} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded:
            return True
    return False


def _chunk_lookup(nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
    return {str(node.node_id): node for node in nodes}


def _selected_nodes(nodes: Sequence[EvidenceNode], node_ids: Iterable[str]) -> tuple[EvidenceNode, ...]:
    by_id = _chunk_lookup(nodes)
    selected: list[EvidenceNode] = []
    seen: set[str] = set()
    for node_id in node_ids:
        key = str(node_id)
        if key in seen or key not in by_id:
            continue
        selected.append(by_id[key])
        seen.add(key)
    return tuple(selected)


def build_surface_sentence_sets(example: QueryExample) -> SurfaceSentenceSets:
    sentences = build_sentence_nodes(example.nodes)
    answers = gold_answers(example)
    answer_ids = frozenset(
        sentence.sentence_id
        for sentence in sentences
        if answers and _contains_answer(sentence.sentence_text, answers)
    )
    gold_ids: set[str] = set()
    support_facts = support_facts_from_metadata(example.metadata)
    if support_facts:
        for fact in resolve_support_facts(example.nodes, support_facts):
            for sentence in sentences:
                if (
                    sentence.chunk_id == fact.node_id
                    and sentence.sentence_index_in_chunk == fact.sentence_id
                ):
                    gold_ids.add(sentence.sentence_id)
                    break
    return SurfaceSentenceSets(
        all_sentences=sentences,
        answer_sentence_ids=answer_ids,
        gold_sentence_ids=frozenset(gold_ids),
    )


def sentence_ids_in_chunks(
    sentence_sets: SurfaceSentenceSets,
    chunk_ids: Iterable[str],
) -> frozenset[str]:
    chunks = {str(chunk_id) for chunk_id in chunk_ids}
    return frozenset(
        sentence.sentence_id
        for sentence in sentence_sets.all_sentences
        if sentence.chunk_id in chunks
    )


def _rendered_sentence_ids(
    sentence_sets: SurfaceSentenceSets,
    rendered_chunk_ids: Iterable[str],
    explicit_sentence_ids: Iterable[str] = (),
) -> frozenset[str]:
    ids = set(str(sentence_id) for sentence_id in explicit_sentence_ids)
    ids.update(sentence_ids_in_chunks(sentence_sets, rendered_chunk_ids))
    return frozenset(ids)


def selected_chunk_surface_trace(
    *,
    example: QueryExample,
    selected_chunk_ids: Iterable[str],
    rendered_chunk_ids: Iterable[str],
    candidate_chunk_ids: Iterable[str] | None = None,
    explicit_rendered_sentence_ids: Iterable[str] = (),
    qa_f1: float = 0.0,
    current_renderer_answer_in_context: bool | None = None,
) -> dict[str, Any]:
    sentence_sets = build_surface_sentence_sets(example)
    selected_chunks = tuple(dict.fromkeys(str(node_id) for node_id in selected_chunk_ids))
    rendered_chunks = tuple(dict.fromkeys(str(node_id) for node_id in rendered_chunk_ids))
    candidate_chunks = (
        tuple(dict.fromkeys(str(node_id) for node_id in candidate_chunk_ids))
        if candidate_chunk_ids is not None
        else tuple(node.node_id for node in example.nodes)
    )
    selected_sentence_ids = sentence_ids_in_chunks(sentence_sets, selected_chunks)
    candidate_sentence_ids = sentence_ids_in_chunks(sentence_sets, candidate_chunks)
    rendered_sentence_ids = _rendered_sentence_ids(
        sentence_sets,
        rendered_chunks,
        explicit_rendered_sentence_ids,
    )
    answer_in_candidate = bool(sentence_sets.answer_sentence_ids & candidate_sentence_ids)
    answer_in_selected = bool(sentence_sets.answer_sentence_ids & selected_sentence_ids)
    gold_in_selected = bool(sentence_sets.gold_sentence_ids & selected_sentence_ids)
    answer_rendered = bool(sentence_sets.answer_sentence_ids & rendered_sentence_ids)
    gold_rendered = bool(sentence_sets.gold_sentence_ids & rendered_sentence_ids)
    if current_renderer_answer_in_context is None:
        current_renderer_answer_in_context = answer_rendered
    return {
        "query_id": example.query_id,
        "dataset": str(example.metadata.get("dataset") or ""),
        "selected_chunk_ids": list(selected_chunks),
        "rendered_chunk_ids": list(rendered_chunks),
        "answer_containing_chunk_found": bool(sentence_sets.answer_sentence_ids),
        "answer_chunk_in_candidate": answer_in_candidate,
        "answer_chunk_in_selected_chunks": answer_in_selected,
        "answer_sentence_in_selected_chunks": answer_in_selected,
        "gold_support_chunk_in_selected_chunks": gold_in_selected,
        "gold_support_sentence_in_selected_chunks": gold_in_selected,
        "answer_sentence_count_in_selected_chunks": len(sentence_sets.answer_sentence_ids & selected_sentence_ids),
        "gold_sentence_count_in_selected_chunks": len(sentence_sets.gold_sentence_ids & selected_sentence_ids),
        "current_renderer_answer_in_context": bool(current_renderer_answer_in_context),
        "current_renderer_gold_sentence_rendered": gold_rendered,
        "answer_sentence_ids": sorted(sentence_sets.answer_sentence_ids),
        "gold_sentence_ids": sorted(sentence_sets.gold_sentence_ids),
        "selected_sentence_ids": sorted(selected_sentence_ids),
        "rendered_sentence_ids": sorted(rendered_sentence_ids),
        "qa_f1": float(qa_f1),
    }


def aggregate_selected_chunk_surface_traces(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "answer_chunk_selected_rate": 0.0,
            "answer_sentence_available_in_selected_chunks_rate": 0.0,
            "gold_sentence_available_in_selected_chunks_rate": 0.0,
            "current_renderer_answer_recovery_given_available": 0.0,
            "current_renderer_gold_recovery_given_available": 0.0,
        }

    def rate(key: str, subset: Sequence[dict[str, Any]] = rows) -> float:
        if not subset:
            return 0.0
        return float(sum(1 for row in subset if bool(row.get(key))) / len(subset))

    answer_available = [row for row in rows if row.get("answer_sentence_in_selected_chunks")]
    gold_available = [row for row in rows if row.get("gold_support_sentence_in_selected_chunks")]
    return {
        "answer_chunk_selected_rate": rate("answer_chunk_in_selected_chunks"),
        "answer_sentence_available_in_selected_chunks_rate": rate("answer_sentence_in_selected_chunks"),
        "gold_sentence_available_in_selected_chunks_rate": rate("gold_support_sentence_in_selected_chunks"),
        "current_renderer_answer_recovery_given_available": rate(
            "current_renderer_answer_in_context",
            answer_available,
        ),
        "current_renderer_gold_recovery_given_available": rate(
            "current_renderer_gold_sentence_rendered",
            gold_available,
        ),
    }


__all__ = [
    "SurfaceSentenceSets",
    "aggregate_selected_chunk_surface_traces",
    "build_surface_sentence_sets",
    "selected_chunk_surface_trace",
    "sentence_ids_in_chunks",
]
