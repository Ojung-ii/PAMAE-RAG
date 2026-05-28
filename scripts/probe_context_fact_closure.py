#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.graph.content_graph import build_content_graph_index
from pamae_rag.qa.generator import DeterministicExtractiveSentenceGenerator
from pamae_rag.qa.metrics import gold_answers, score_prediction
from pamae_rag.qa.runner import _answer_coverage


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[str(row["query_id"])] = row
    return predictions


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _query_entities(query: str) -> set[str]:
    node = EvidenceNode("query", query, np.zeros(1, dtype=np.float64), metadata={})
    return {entity.canonical for entity in build_content_graph_index([node]).entities}


def _fallback_context(nodes: list[EvidenceNode]) -> str:
    return "\n".join(node.text for node in nodes)


def _fact_closure_context(example: QueryExample, prediction: dict[str, Any], mode: str) -> str:
    by_id = {node.node_id: node for node in example.nodes}
    selected = [
        by_id[str(node_id)]
        for node_id in prediction.get("context_node_ids", [])
        if str(node_id) in by_id
    ]
    if not selected:
        return ""

    query_entities = _query_entities(example.query)
    if not query_entities:
        return _fallback_context(selected)

    index = build_content_graph_index(selected)
    entity_canonicals = {entity.entity_id: entity.canonical for entity in index.entities}
    seed_entity_ids = {
        entity_id
        for entity_id, canonical in entity_canonicals.items()
        if canonical in query_entities
    }
    if not seed_entity_ids:
        return _fallback_context(selected)

    fact_entities = {fact.fact_id: set(fact.entity_ids) for fact in index.facts}
    include_entity_ids = set(seed_entity_ids)
    if mode in {"onehop", "twohop"}:
        for entity_ids in fact_entities.values():
            if entity_ids & seed_entity_ids:
                include_entity_ids |= entity_ids
    if mode == "twohop":
        for entity_ids in fact_entities.values():
            if entity_ids & include_entity_ids:
                include_entity_ids |= entity_ids

    texts = [
        fact.text
        for fact in index.facts
        if set(fact.entity_ids) & include_entity_ids
    ]
    return "\n".join(texts) if texts else _fallback_context(selected)


def probe(input_path: Path, prediction_path: Path, *, limit: int | None, mode: str) -> dict[str, Any]:
    examples = read_jsonl(input_path, limit=limit)
    predictions = _read_predictions(prediction_path)
    generator = DeterministicExtractiveSentenceGenerator()
    exact_matches: list[float] = []
    f1s: list[float] = []
    answer_coverages: list[float] = []
    selected_answer_coverages: list[float] = []
    context_tokens: list[float] = []

    for example in examples:
        prediction = predictions.get(example.query_id, {"context_node_ids": []})
        context = _fact_closure_context(example, prediction, mode)
        generated = generator.generate(example.query, context)
        answers = gold_answers(example)
        score = score_prediction(generated.answer, answers)
        if score is not None:
            exact_matches.append(score.exact_match)
            f1s.append(score.f1)
        answer_coverage = _answer_coverage(context, answers)
        if answer_coverage is not None:
            answer_coverages.append(answer_coverage)
        selected_answer_coverage = _answer_coverage(generated.answer, answers)
        if selected_answer_coverage is not None:
            selected_answer_coverages.append(selected_answer_coverage)
        context_tokens.append(float(len(context.split())) if context else 0.0)

    return {
        "num_queries": len(examples),
        "mode": mode,
        "mean_exact_match": _mean(exact_matches),
        "mean_f1": _mean(f1s),
        "mean_answer_coverage": _mean(answer_coverages),
        "mean_selected_answer_coverage": _mean(selected_answer_coverages),
        "avg_context_tokens": _mean(context_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe content fact closure over existing retrieved contexts without changing retrieval."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input examples JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="Retrieval predictions JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=("seed", "onehop", "twohop"), default="onehop")
    args = parser.parse_args()

    metrics = probe(args.input, args.predictions, limit=args.limit, mode=args.mode)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
