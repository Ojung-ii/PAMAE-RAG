#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from pamae_rag.data.io import read_jsonl
from pamae_rag.objective.relevance_mass import relevance_scores


def _optional_auc(labels: list[int], scores: list[float]) -> tuple[float | None, str | None]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover - depends on optional env
        return None, f"sklearn unavailable: {exc}"
    if len(set(labels)) < 2:
        return None, "requires both positive and negative labels"
    return float(roc_auc_score(labels, scores)), None


def _optional_spearman(labels: list[int], scores: list[float]) -> tuple[float | None, str | None]:
    try:
        from scipy.stats import spearmanr
    except Exception as exc:  # pragma: no cover - depends on optional env
        return None, f"scipy unavailable: {exc}"
    if len(set(labels)) < 2:
        return None, "requires both positive and negative labels"
    value = spearmanr(scores, labels).correlation
    if value is None or np.isnan(value):
        return None, "spearman correlation is undefined"
    return float(value), None


def analyze_examples(path: str | Path, relevance_mode: str = "current") -> dict[str, Any]:
    examples = read_jsonl(path)
    gold_total = 0
    gold_ranks: list[int] = []
    gold_relevances: list[float] = []
    non_gold_relevances: list[float] = []
    labels: list[int] = []
    scores_for_labels: list[float] = []
    examples_with_no_gold = 0
    examples_with_gold_outside_top50 = 0
    sample_gold_outside_top50_query_ids: list[str] = []

    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}

    for example in examples:
        gold = set(example.gold_node_ids)
        if not gold:
            examples_with_no_gold += 1
        gold_total += len(gold)

        scores = relevance_scores(
            example.nodes,
            mode=relevance_mode,
            query=example.query,
            query_metadata=example.metadata,
        )
        ranked = sorted(range(len(example.nodes)), key=lambda i: (-float(scores[int(i)]), int(i)))
        rank_by_id = {example.nodes[int(idx)].node_id: rank for rank, idx in enumerate(ranked, start=1)}
        node_by_id = {node.node_id: idx for idx, node in enumerate(example.nodes)}

        outside_top50 = False
        for gold_id in gold:
            rank = rank_by_id.get(gold_id)
            if rank is None:
                outside_top50 = True
                continue
            gold_ranks.append(rank)
            gold_relevances.append(float(scores[int(node_by_id[gold_id])]))
            if rank > 50:
                outside_top50 = True
            for k in top_hits:
                if rank <= k:
                    top_hits[k] += 1

        if outside_top50:
            examples_with_gold_outside_top50 += 1
            if len(sample_gold_outside_top50_query_ids) < 10:
                sample_gold_outside_top50_query_ids.append(example.query_id)

        for idx, node in enumerate(example.nodes):
            label = 1 if node.node_id in gold else 0
            labels.append(label)
            score = float(scores[int(idx)])
            scores_for_labels.append(score)
            if label:
                continue
            non_gold_relevances.append(score)

    auc, auc_reason = _optional_auc(labels, scores_for_labels)
    spearman, spearman_reason = _optional_spearman(labels, scores_for_labels)
    diagnostics: dict[str, Any] = {}
    if auc is None:
        diagnostics["relevance_label_auc_reason"] = auc_reason
    if spearman is None:
        diagnostics["relevance_label_spearman_reason"] = spearman_reason

    denom = max(gold_total, 1)
    return {
        "num_queries": len(examples),
        "gold_total": gold_total,
        "gold_rank_mean": mean(gold_ranks) if gold_ranks else None,
        "gold_rank_median": median(gold_ranks) if gold_ranks else None,
        "gold_top1_rate": top_hits[1] / denom,
        "gold_top3_rate": top_hits[3] / denom,
        "gold_top5_rate": top_hits[5] / denom,
        "gold_top10_rate": top_hits[10] / denom,
        "gold_top20_rate": top_hits[20] / denom,
        "gold_top50_rate": top_hits[50] / denom,
        "examples_with_no_gold": examples_with_no_gold,
        "examples_with_gold_outside_top50": examples_with_gold_outside_top50,
        "sample_gold_outside_top50_query_ids": sample_gold_outside_top50_query_ids,
        "mean_gold_relevance": mean(gold_relevances) if gold_relevances else None,
        "mean_non_gold_relevance": mean(non_gold_relevances) if non_gold_relevances else None,
        "relevance_label_auc": auc,
        "relevance_label_spearman": spearman,
        "diagnostics": diagnostics,
    }


def _markdown(metrics: dict[str, Any], input_path: str, relevance_mode: str) -> str:
    rows = [
        ("input", input_path),
        ("relevance_mode", relevance_mode),
        *[(key, value) for key, value in metrics.items() if key not in {"diagnostics", "sample_gold_outside_top50_query_ids"}],
    ]
    lines = ["# Relevance Alignment Diagnostic", "", "| metric | value |", "| --- | ---: |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    if metrics.get("sample_gold_outside_top50_query_ids"):
        lines.extend(["", "## Sample Queries Outside Top 50", ""])
        for qid in metrics["sample_gold_outside_top50_query_ids"]:
            lines.append(f"- `{qid}`")
    if metrics.get("diagnostics"):
        lines.extend(["", "## Diagnostics", ""])
        for key, value in metrics["diagnostics"].items():
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--relevance-mode", default="current")
    args = parser.parse_args(argv)

    metrics = analyze_examples(args.input, args.relevance_mode)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(metrics, args.input, args.relevance_mode), encoding="utf-8")


if __name__ == "__main__":
    main()
