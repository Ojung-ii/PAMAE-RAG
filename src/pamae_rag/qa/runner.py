from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.eval.support_recall import f1_score, precision, recall
from pamae_rag.qa.generator import DeterministicExtractiveSentenceGenerator, PROMPT_TEXT
from pamae_rag.qa.metrics import METRIC_ID, gold_answers, score_json, score_prediction


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
    avg_context_tokens: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    missing_prediction_count: int
    missing_answer_count: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _read_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[str(row["query_id"])] = row
    return predictions


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _node_order(nodes: tuple[EvidenceNode, ...], node_ids: list[str]) -> list[EvidenceNode]:
    by_id = {node.node_id: node for node in nodes}
    return [by_id[node_id] for node_id in node_ids if node_id in by_id]


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


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_qa(
    input_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    metrics_output_path: str | Path,
    *,
    limit: int | None = None,
) -> QAMetrics:
    examples = read_jsonl(input_path, limit=limit)
    predictions = _read_predictions(prediction_path)
    generator = DeterministicExtractiveSentenceGenerator()
    rows: list[dict[str, Any]] = []
    exact_matches: list[float] = []
    f1s: list[float] = []
    context_recalls: list[float] = []
    context_precisions: list[float] = []
    context_f1s: list[float] = []
    context_tokens: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    missing_prediction_count = 0
    missing_answer_count = 0

    for example in examples:
        prediction = predictions.get(example.query_id)
        if prediction is None:
            missing_prediction_count += 1
            prediction = {"query_id": example.query_id, "context_node_ids": []}
        context_ids = _prediction_context_ids(example, prediction)
        context_nodes = _node_order(example.nodes, context_ids)
        context = _context_text(context_nodes)
        start = time.perf_counter()
        generated = generator.generate(example.query, context)
        generation_ms = round((time.perf_counter() - start) * 1000.0, 3)
        answers = gold_answers(example)
        if not answers:
            missing_answer_count += 1
        answer_score = score_prediction(generated.answer, answers)
        score_payload = score_json(answer_score)
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
        retrieval_ms = _float_value(prediction.get("latency_ms"))
        if retrieval_ms is not None:
            retrieval_latencies.append(retrieval_ms)
        generation_latencies.append(generation_ms)
        rows.append(
            {
                "query_id": example.query_id,
                "oracle": False,
                "prediction": generated.answer,
                "gold_answers": list(answers),
                **score_payload,
                "context_node_ids": context_ids,
                "context_recall": c_recall,
                "context_precision": c_precision,
                "context_f1": c_f1,
                "context_tokens": token_count,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "diagnostics": {
                    "generator_id": generated.generator_id,
                    "prompt_id": generated.prompt_id,
                    "prompt_text": PROMPT_TEXT,
                    "metric_id": METRIC_ID,
                    "selected_sentence_index": generated.selected_sentence_index,
                    "source_prediction_query_id": prediction.get("query_id"),
                },
            }
        )

    metrics = QAMetrics(
        num_queries=len(examples),
        oracle=False,
        generator_id=generator.generator_id,
        prompt_id=generator.prompt_id,
        metric_id=METRIC_ID,
        mean_exact_match=_mean(exact_matches),
        mean_f1=_mean(f1s),
        mean_context_recall=_mean(context_recalls),
        mean_context_precision=_mean(context_precisions),
        mean_context_f1=_mean(context_f1s),
        avg_context_tokens=_mean(context_tokens),
        avg_retrieval_ms=_mean(retrieval_latencies),
        avg_generation_ms=_mean(generation_latencies),
        missing_prediction_count=missing_prediction_count,
        missing_answer_count=missing_answer_count,
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
