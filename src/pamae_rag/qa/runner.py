from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.eval.support_facts import (
    resolve_support_facts,
    support_fact_stage_metrics,
    support_facts_from_metadata,
)
from pamae_rag.eval.stage_diagnostics import aggregate_stage_diagnostics, make_stage_metrics
from pamae_rag.eval.support_recall import f1_score, precision, recall
from pamae_rag.qa.generator import DeterministicExtractiveSentenceGenerator, PROMPT_TEXT
from pamae_rag.qa.metrics import METRIC_ID, gold_answers, normalize_answer, score_json, score_prediction

_CORPUS_NODE_RE = re.compile(r"^(?P<dataset>.+):doc:(?P<index>[0-9]+)$")


@dataclass(frozen=True)
class QAMetrics:
    num_queries: int
    oracle: bool
    generator_id: str
    prompt_id: str
    metric_id: str
    mean_exact_match: float
    mean_f1: float
    mean_context_recall: float
    mean_context_precision: float
    mean_context_f1: float
    mean_answer_coverage: float
    mean_selected_answer_coverage: float
    avg_context_tokens: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    missing_prediction_count: int
    missing_answer_count: int
    stage_diagnostics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _read_predictions(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    predictions: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[str(row["query_id"])] = row
    return predictions


def _read_corpus(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Corpus must be a JSON list: {path}")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _corpus_index(node_id: str) -> int | None:
    match = _CORPUS_NODE_RE.match(node_id)
    return int(match.group("index")) if match else None


def _node_from_corpus(node_id: str, corpus: list[dict[str, Any]]) -> EvidenceNode | None:
    index = _corpus_index(node_id)
    if index is None or index < 0 or index >= len(corpus):
        return None
    item = corpus[index]
    text = str(item.get("text") or "")
    title = str(item.get("title") or "")
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(1, dtype=np.float64),
        token_count=max(1, len(text.split())),
        metadata={"title": title, "corpus_index": index},
    )


def _node_order(
    nodes: tuple[EvidenceNode, ...],
    node_ids: list[str],
    corpus: list[dict[str, Any]] | None = None,
) -> tuple[list[EvidenceNode], list[str], int]:
    by_id = {node.node_id: node for node in nodes}
    ordered: list[EvidenceNode] = []
    missing: list[str] = []
    corpus_count = 0
    for node_id in node_ids:
        if node_id in by_id:
            ordered.append(by_id[node_id])
            continue
        corpus_node = _node_from_corpus(node_id, corpus or [])
        if corpus_node is None:
            missing.append(node_id)
            continue
        ordered.append(corpus_node)
        corpus_count += 1
    return ordered, missing, corpus_count


def _context_text(nodes: list[EvidenceNode]) -> str:
    return "\n\n".join(node.text for node in nodes)


def _context_tokens(nodes: list[EvidenceNode]) -> int:
    return int(sum(max(1, int(node.token_count)) for node in nodes))


def _prediction_context_ids(example: QueryExample, prediction: dict[str, Any]) -> list[str]:
    values = prediction.get("context_node_ids", [])
    if not isinstance(values, list):
        return []
    valid = {node.node_id for node in example.nodes}
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        node_id = str(value)
        if node_id in valid and node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def _gold_context_ids(example: QueryExample) -> list[str]:
    gold = set(example.gold_node_ids)
    present = [node.node_id for node in example.nodes if node.node_id in gold]
    remaining = sorted(gold - set(present), key=lambda node_id: (_corpus_index(node_id) is None, _corpus_index(node_id) or 0, node_id))
    return [*present, *remaining]


def _support_sentence_node(node: EvidenceNode, sentence_id: int, sentence: str) -> EvidenceNode:
    metadata = dict(node.metadata)
    metadata["oracle_context_unit"] = "support_sentence"
    metadata["support_sentence_id"] = int(sentence_id)
    return EvidenceNode(
        node_id=node.node_id,
        text=sentence,
        embedding=node.embedding,
        relevance=node.relevance,
        token_count=max(1, len(sentence.split())),
        node_type=node.node_type,
        is_anchor_candidate=node.is_anchor_candidate,
        metadata=metadata,
    )


def _oracle_context(
    example: QueryExample,
    corpus: list[dict[str, Any]],
) -> tuple[list[str], list[EvidenceNode], list[str], int, dict[str, Any]]:
    context_ids = _gold_context_ids(example)
    gold_nodes, missing_context_ids, corpus_context_count = _node_order(
        example.nodes,
        context_ids,
        corpus,
    )
    support_facts = support_facts_from_metadata(example.metadata)
    diagnostics: dict[str, Any] = {
        "oracle_context_unit": "gold_node",
        "support_fact_count": len(support_facts),
        "support_fact_resolved_count": 0,
    }
    if not support_facts:
        return context_ids, gold_nodes, missing_context_ids, corpus_context_count, diagnostics

    support_nodes: list[EvidenceNode] = []
    support_ids: list[str] = []
    gold_nodes_by_id = {node.node_id: node for node in gold_nodes}
    for fact in resolve_support_facts(gold_nodes, support_facts):
        support_nodes.append(
            _support_sentence_node(
                gold_nodes_by_id[fact.node_id],
                fact.sentence_id,
                fact.sentence,
            )
        )
        support_ids.append(fact.node_id)

    diagnostics["support_fact_resolved_count"] = len(support_nodes)
    if len(support_nodes) != len(support_facts):
        diagnostics["oracle_context_unit"] = "gold_node_fallback"
        return context_ids, gold_nodes, missing_context_ids, corpus_context_count, diagnostics

    diagnostics["oracle_context_unit"] = "support_sentence"
    return support_ids, support_nodes, missing_context_ids, corpus_context_count, diagnostics


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _answer_coverage(context: str, answers: tuple[str, ...]) -> float | None:
    if not answers:
        return None
    context_norm = normalize_answer(context)
    if not context_norm:
        return 0.0
    padded_context = f" {context_norm} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded_context:
            return 1.0
    return 0.0


def run_qa(
    input_path: str | Path,
    prediction_path: str | Path | None,
    output_path: str | Path,
    metrics_output_path: str | Path,
    *,
    limit: int | None = None,
    oracle_context: bool = False,
    corpus_path: str | Path | None = None,
) -> QAMetrics:
    examples = read_jsonl(input_path, limit=limit)
    predictions = _read_predictions(prediction_path)
    corpus = _read_corpus(corpus_path)
    generator = DeterministicExtractiveSentenceGenerator()
    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, dict[str, Any]]] = []
    exact_matches: list[float] = []
    f1s: list[float] = []
    context_recalls: list[float] = []
    context_precisions: list[float] = []
    context_f1s: list[float] = []
    context_tokens: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    answer_coverages: list[float] = []
    selected_answer_coverages: list[float] = []
    missing_prediction_count = 0
    missing_answer_count = 0

    for example in examples:
        prediction = predictions.get(example.query_id)
        if prediction is None:
            if not oracle_context:
                missing_prediction_count += 1
            prediction = {"query_id": example.query_id, "context_node_ids": []}
        if oracle_context:
            context_ids, context_nodes, missing_context_ids, corpus_context_count, oracle_diagnostics = (
                _oracle_context(example, corpus)
            )
        else:
            context_ids = _prediction_context_ids(example, prediction)
            context_nodes, missing_context_ids, corpus_context_count = _node_order(
                example.nodes,
                context_ids,
                corpus,
            )
            oracle_diagnostics = {}
        context = _context_text(context_nodes)
        start = time.perf_counter()
        generated = generator.generate(example.query, context)
        generation_ms = round((time.perf_counter() - start) * 1000.0, 3)
        answers = gold_answers(example)
        if not answers:
            missing_answer_count += 1
        answer_score = score_prediction(generated.answer, answers)
        score_payload = score_json(answer_score)
        answer_coverage = _answer_coverage(context, answers)
        if answer_coverage is not None:
            answer_coverages.append(answer_coverage)
        selected_answer_coverage = _answer_coverage(generated.answer, answers)
        if selected_answer_coverage is not None:
            selected_answer_coverages.append(selected_answer_coverage)
        if answer_score is not None:
            exact_matches.append(answer_score.exact_match)
            f1s.append(answer_score.f1)
        c_recall = recall(tuple(context_ids), example.gold_node_ids)
        c_precision = precision(tuple(context_ids), example.gold_node_ids)
        c_f1 = f1_score(c_precision, c_recall)
        if c_recall is not None:
            context_recalls.append(c_recall)
        if c_precision is not None:
            context_precisions.append(c_precision)
        if c_f1 is not None:
            context_f1s.append(c_f1)
        token_count = float(_context_tokens(context_nodes))
        context_tokens.append(token_count)
        retrieval_ms = None if oracle_context else _float_value(prediction.get("latency_ms"))
        if retrieval_ms is not None:
            retrieval_latencies.append(retrieval_ms)
        generation_latencies.append(generation_ms)
        prediction_diagnostics = prediction.get("diagnostics") if isinstance(prediction, dict) else {}
        stage_diagnostics = {}
        if isinstance(prediction_diagnostics, dict) and isinstance(
            prediction_diagnostics.get("stage_diagnostics"),
            dict,
        ):
            stage_diagnostics.update(prediction_diagnostics["stage_diagnostics"])
        support_fact_nodes = context_nodes if oracle_context else list(example.nodes)
        support_fact_extra = support_fact_stage_metrics(
            nodes=support_fact_nodes,
            selected_node_ids=context_ids,
            metadata=example.metadata,
        )
        final_extra = {
            "qa_exact_match": score_payload["exact_match"],
            "qa_f1": score_payload["f1"],
            "answer_coverage": answer_coverage,
            "selected_answer_coverage": selected_answer_coverage,
            "context_source": "gold_support" if oracle_context else "retrieval_prediction",
            **support_fact_extra,
            **oracle_diagnostics,
        }
        stage_diagnostics["final_qa"] = make_stage_metrics(
            stage="final_qa",
            selected_node_ids=context_ids,
            gold_node_ids=example.gold_node_ids,
            context_node_ids=context_ids,
            rendered_node_ids=context_ids,
            token_count=token_count,
            latency_ms=generation_ms,
            extra=final_extra,
        )
        stage_rows.append(stage_diagnostics)
        rows.append(
            {
                "query_id": example.query_id,
                "oracle": bool(oracle_context),
                "prediction": generated.answer,
                "gold_answers": list(answers),
                **score_payload,
                "context_node_ids": context_ids,
                "context_recall": c_recall,
                "context_precision": c_precision,
                "context_f1": c_f1,
                "context_tokens": token_count,
                "answer_coverage": answer_coverage,
                "selected_answer_coverage": selected_answer_coverage,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "stage_diagnostics": stage_diagnostics,
                "diagnostics": {
                    "generator_id": generated.generator_id,
                    "prompt_id": generated.prompt_id,
                    "prompt_text": PROMPT_TEXT,
                    "metric_id": METRIC_ID,
                    "selected_sentence_index": generated.selected_sentence_index,
                    "context_source": "gold_support" if oracle_context else "retrieval_prediction",
                    "missing_context_node_ids": missing_context_ids,
                    "corpus_context_node_count": corpus_context_count,
                    **oracle_diagnostics,
                    "source_prediction_query_id": prediction.get("query_id"),
                },
            }
        )

    metrics = QAMetrics(
        num_queries=len(examples),
        oracle=bool(oracle_context),
        generator_id=generator.generator_id,
        prompt_id=generator.prompt_id,
        metric_id=METRIC_ID,
        mean_exact_match=_mean(exact_matches),
        mean_f1=_mean(f1s),
        mean_context_recall=_mean(context_recalls),
        mean_context_precision=_mean(context_precisions),
        mean_context_f1=_mean(context_f1s),
        mean_answer_coverage=_mean(answer_coverages),
        mean_selected_answer_coverage=_mean(selected_answer_coverages),
        avg_context_tokens=_mean(context_tokens),
        avg_retrieval_ms=_mean(retrieval_latencies),
        avg_generation_ms=_mean(generation_latencies),
        missing_prediction_count=missing_prediction_count,
        missing_answer_count=missing_answer_count,
        stage_diagnostics=aggregate_stage_diagnostics(stage_rows),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_output = Path(metrics_output_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
