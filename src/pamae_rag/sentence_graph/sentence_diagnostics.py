from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from pamae_rag.data.schema import QueryExample
from pamae_rag.eval.support_facts import normalize_support_title, support_facts_from_metadata
from pamae_rag.eval.support_recall import f1_score, precision, recall
from pamae_rag.qa.metrics import gold_answers, normalize_answer
from pamae_rag.sentence_graph.sentence_graph_builder import SentenceGraphIndex
from pamae_rag.sentence_graph.sentence_renderer import SentenceRenderResult
from pamae_rag.sentence_graph.sentence_retriever import SentenceRetrievalResult


@dataclass(frozen=True)
class SentenceEvidenceSets:
    gold_sentence_ids: tuple[str, ...]
    answer_sentence_ids: tuple[str, ...]
    support_fact_count: int
    support_fact_mapped_count: int


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def support_sentence_ids(index: SentenceGraphIndex, metadata: dict[str, Any] | None) -> tuple[str, ...]:
    facts = support_facts_from_metadata(metadata)
    if not facts:
        return tuple()
    by_title: dict[str, list] = {}
    for sentence in index.sentence_nodes:
        title = normalize_support_title(sentence.title)
        by_title.setdefault(title, []).append(sentence)

    out: list[str] = []
    for fact in facts:
        title = normalize_support_title(fact.get("title"))
        try:
            sentence_id = int(fact.get("sentence_id"))
        except (TypeError, ValueError):
            continue
        candidates = by_title.get(title, [])
        exact = [
            sentence
            for sentence in candidates
            if sentence.sentence_index_in_doc == sentence_id
        ]
        if not exact:
            exact = [
                sentence
                for sentence in candidates
                if sentence.sentence_index_in_chunk == sentence_id
            ]
        if exact:
            out.append(sorted(exact, key=lambda sentence: sentence.sentence_id)[0].sentence_id)
    return _ordered_unique(out)


def answer_sentence_ids(index: SentenceGraphIndex, answers: Iterable[str]) -> tuple[str, ...]:
    answer_norms = [normalize_answer(answer) for answer in answers]
    answer_norms = [answer for answer in answer_norms if answer]
    if not answer_norms:
        return tuple()
    out: list[str] = []
    for sentence in index.sentence_nodes:
        padded = f" {normalize_answer(sentence.sentence_text)} "
        if any(f" {answer} " in padded for answer in answer_norms):
            out.append(sentence.sentence_id)
    return tuple(out)


def sentence_evidence_sets(index: SentenceGraphIndex, example: QueryExample) -> SentenceEvidenceSets:
    facts = support_facts_from_metadata(example.metadata)
    gold_ids = support_sentence_ids(index, example.metadata)
    answer_ids = answer_sentence_ids(index, gold_answers(example))
    return SentenceEvidenceSets(
        gold_sentence_ids=gold_ids,
        answer_sentence_ids=answer_ids,
        support_fact_count=len(facts),
        support_fact_mapped_count=len(gold_ids),
    )


def sentence_mapping_diagnostics(index: SentenceGraphIndex, example: QueryExample) -> dict[str, Any]:
    evidence = sentence_evidence_sets(index, example)
    return {
        "gold_support_sentence_mapping_rate": (
            evidence.support_fact_mapped_count / evidence.support_fact_count
            if evidence.support_fact_count
            else 0.0
        ),
        "answer_containing_sentence_found_rate": 1.0 if evidence.answer_sentence_ids else 0.0,
        "avg_sentences_per_chunk": index.diagnostics["avg_sentences_per_chunk"],
        "avg_entities_per_sentence": index.diagnostics["avg_entities_per_sentence"],
        "isolated_sentence_rate": index.diagnostics["isolated_sentence_rate"],
        "support_fact_count": evidence.support_fact_count,
        "support_fact_mapped_count": evidence.support_fact_mapped_count,
        "answer_sentence_count": len(evidence.answer_sentence_ids),
    }


def _mean_bool(values: Iterable[bool]) -> float:
    vals = list(values)
    return float(sum(1 for value in vals if value) / len(vals)) if vals else 0.0


def _answer_in_text(answers: Iterable[str], text: str) -> bool:
    context = normalize_answer(text)
    if not context:
        return False
    padded = f" {context} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded:
            return True
    return False


def _nearest_distance(
    target_sentence_ids: Iterable[str],
    selected_sentence_ids: tuple[str, ...],
    retrieval: SentenceRetrievalResult,
) -> float | None:
    active_pos = {sentence_id: idx for idx, sentence_id in enumerate(retrieval.active_sentence_ids)}
    selected_pos = [active_pos[sentence_id] for sentence_id in selected_sentence_ids if sentence_id in active_pos]
    values: list[float] = []
    for sentence_id in target_sentence_ids:
        row = active_pos.get(sentence_id)
        if row is None or not selected_pos:
            continue
        values.append(float(np.min(retrieval.distance_matrix[row, selected_pos])))
    return float(sum(values) / len(values)) if values else None


def build_sentence_diagnostic_trace(
    *,
    example: QueryExample,
    index: SentenceGraphIndex,
    retrieval: SentenceRetrievalResult,
    render: SentenceRenderResult,
    graph_variant: str,
    renderer_mode: str,
    qa_f1: float | None = None,
) -> dict[str, Any]:
    evidence = sentence_evidence_sets(index, example)
    active = set(retrieval.active_sentence_ids)
    selected = set(retrieval.selected_sentence_ids)
    rendered = set(render.rendered_sentence_ids)
    path = set(render.path_sentence_ids)
    gold = set(evidence.gold_sentence_ids)
    answer = set(evidence.answer_sentence_ids)
    rendered_recall = recall(tuple(rendered), frozenset(gold))
    rendered_precision = precision(tuple(rendered), frozenset(gold))
    context_f1 = f1_score(rendered_precision, rendered_recall)
    answer_in_context = _answer_in_text(gold_answers(example), render.context_text)
    answer_projected = bool(answer & active)
    answer_selected = bool(answer & selected)
    answer_rendered = bool(answer & rendered)
    answer_in_selected_basin = bool(answer_projected)

    trace = {
        "query_id": example.query_id,
        "dataset": example.metadata.get("dataset"),
        "graph_variant": graph_variant,
        "renderer_mode": renderer_mode,
        "gold_sentence_ids": sorted(gold),
        "answer_sentence_ids": sorted(answer),
        "selected_sentence_ids": list(retrieval.selected_sentence_ids),
        "rendered_sentence_ids": list(render.rendered_sentence_ids),
        "gold_sentence_projected": bool(gold & active),
        "answer_sentence_found": bool(answer),
        "answer_sentence_projected": answer_projected,
        "answer_sentence_selected": answer_selected,
        "answer_sentence_in_selected_basin": answer_in_selected_basin,
        "answer_sentence_on_support_tree": bool(answer & path),
        "answer_sentence_rendered": answer_rendered,
        "gold_sentence_rendered": bool(gold & rendered),
        "answer_in_context": answer_in_context,
        "rendered_recall": rendered_recall,
        "context_precision": rendered_precision,
        "context_f1": context_f1,
        "mean_d_medoid_gold_sentence": _nearest_distance(
            evidence.gold_sentence_ids,
            retrieval.selected_sentence_ids,
            retrieval,
        ),
        "mean_d_medoid_answer_sentence": _nearest_distance(
            evidence.answer_sentence_ids,
            retrieval.selected_sentence_ids,
            retrieval,
        ),
        "phi_before_refine": retrieval.phi_before_refine,
        "phi_after_refine": retrieval.phi_after_refine,
        "objective_decreased": retrieval.objective_decreased,
        "context_tokens": render.context_tokens,
        "qa_f1": qa_f1,
        "anchor_count": len(retrieval.query_anchor_entity_ids),
        "anchor_fallback_used": retrieval.anchor_fallback_used,
        "query_anchor_entities": list(retrieval.query_anchor_entities),
        "active_sentence_universe_size": len(retrieval.active_sentence_ids),
        "objective_increase_count": 1 if retrieval.objective_increase else 0,
        "triangle_inequality_violation_count": int(
            retrieval.diagnostics.get("triangle_inequality_violation_count", 0)
        ),
        "diagnostic_only": render.diagnostic_only,
    }
    trace.update(sentence_mapping_diagnostics(index, example))
    return trace


def _mean(values: Iterable[float | int | None]) -> float:
    nums = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return float(sum(nums) / len(nums)) if nums else 0.0


def aggregate_sentence_traces(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(traces)
    return {
        "num_queries": len(rows),
        "gold_sentence_projection_rate": _mean_bool(row.get("gold_sentence_projected", False) for row in rows),
        "answer_sentence_projection_rate": _mean_bool(row.get("answer_sentence_projected", False) for row in rows),
        "answer_sentence_selected_rate": _mean_bool(row.get("answer_sentence_selected", False) for row in rows),
        "answer_sentence_rendered_rate": _mean_bool(row.get("answer_sentence_rendered", False) for row in rows),
        "answer_in_context_rate": _mean_bool(row.get("answer_in_context", False) for row in rows),
        "rendered_recall": _mean(row.get("rendered_recall") for row in rows),
        "context_f1": _mean(row.get("context_f1") for row in rows),
        "avg_context_tokens": _mean(row.get("context_tokens") for row in rows),
        "qa_f1": _mean(row.get("qa_f1") for row in rows),
        "objective_increase_count": int(sum(int(row.get("objective_increase_count", 0)) for row in rows)),
        "triangle_inequality_violation_count": int(
            sum(int(row.get("triangle_inequality_violation_count", 0)) for row in rows)
        ),
        "gold_support_sentence_mapping_rate": _mean(
            row.get("gold_support_sentence_mapping_rate") for row in rows
        ),
        "answer_containing_sentence_found_rate": _mean(
            row.get("answer_containing_sentence_found_rate") for row in rows
        ),
        "avg_sentences_per_chunk": _mean(row.get("avg_sentences_per_chunk") for row in rows),
        "avg_entities_per_sentence": _mean(row.get("avg_entities_per_sentence") for row in rows),
        "isolated_sentence_rate": _mean(row.get("isolated_sentence_rate") for row in rows),
    }
