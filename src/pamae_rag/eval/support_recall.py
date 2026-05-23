from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pamae_rag.data.schema import QueryExample


@dataclass(frozen=True)
class RetrievalMetrics:
    num_queries: int
    mean_context_recall: float
    mean_context_hit: float
    mean_anchor_recall: float
    mean_anchor_hit: float
    missing_prediction_count: int
    missing_anchor_key_count: int
    anchor_non_empty_ratio: float
    avg_context_size: float
    avg_latency_ms: float
    objective_before_refinement_mean: float
    objective_after_refinement_mean: float
    refinement_accept_rate: float
    objective_support_spearman: float | None
    diagnostics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "mean_context_recall": self.mean_context_recall,
            "mean_context_hit": self.mean_context_hit,
            "mean_anchor_recall": self.mean_anchor_recall,
            "mean_anchor_hit": self.mean_anchor_hit,
            "missing_prediction_count": self.missing_prediction_count,
            "missing_anchor_key_count": self.missing_anchor_key_count,
            "anchor_non_empty_ratio": self.anchor_non_empty_ratio,
            "avg_context_size": self.avg_context_size,
            "avg_latency_ms": self.avg_latency_ms,
            "objective_before_refinement_mean": self.objective_before_refinement_mean,
            "objective_after_refinement_mean": self.objective_after_refinement_mean,
            "refinement_accept_rate": self.refinement_accept_rate,
            "objective_support_spearman": self.objective_support_spearman,
            "diagnostics": self.diagnostics,
        }


def recall(selected: list[str] | tuple[str, ...], gold: set[str] | frozenset[str]) -> float | None:
    if not gold:
        return None
    return len(set(selected) & set(gold)) / len(gold)


def hit(selected: list[str] | tuple[str, ...], gold: set[str] | frozenset[str]) -> float | None:
    if not gold:
        return None
    return 1.0 if set(selected) & set(gold) else 0.0


def aggregate(values: list[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return float(sum(nums) / len(nums)) if nums else 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> tuple[float | None, str | None]:
    if len(xs) < 2:
        return None, "need at least two objective/support pairs"
    if len(set(xs)) < 2:
        return None, "objective values are constant"
    if len(set(ys)) < 2:
        return None, "support values are constant"
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = _mean(rx)
    my = _mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if den_x == 0.0 or den_y == 0.0:
        return None, "rank variance is zero"
    return float(num / (den_x * den_y)), None


def evaluate_predictions(examples: list[QueryExample], predictions: dict[str, dict[str, Any]]) -> RetrievalMetrics:
    c_recalls: list[float | None] = []
    c_hits: list[float | None] = []
    a_recalls: list[float | None] = []
    a_hits: list[float | None] = []
    missing_prediction_count = 0
    missing_anchor_key_count = 0
    anchor_non_empty_count = 0
    context_sizes: list[float] = []
    latencies: list[float] = []
    objectives_before: list[float] = []
    objectives_after: list[float] = []
    refinement_accepts = 0
    objective_support_x: list[float] = []
    objective_support_y: list[float] = []
    for ex in examples:
        pred = predictions.get(ex.query_id)
        if pred is None:
            missing_prediction_count += 1
            pred = {}
        context = tuple(str(x) for x in pred.get("context_node_ids", []))
        if "anchor_node_ids" in pred:
            anchors = tuple(str(x) for x in pred.get("anchor_node_ids", []))
        elif "anchor_ids" in pred:
            anchors = tuple(str(x) for x in pred.get("anchor_ids", []))
        else:
            missing_anchor_key_count += 1
            anchors = ()
        context_recall = recall(context, ex.gold_node_ids)
        c_recalls.append(context_recall)
        c_hits.append(hit(context, ex.gold_node_ids))
        a_recalls.append(recall(anchors, ex.gold_node_ids))
        a_hits.append(hit(anchors, ex.gold_node_ids))
        if anchors:
            anchor_non_empty_count += 1
        context_sizes.append(float(len(context)))
        latency = _float_or_none(pred.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)
        before = _float_or_none(pred.get("objective_before_refinement"))
        after = _float_or_none(pred.get("objective_after_refinement"))
        if before is not None:
            objectives_before.append(before)
        if after is not None:
            objectives_after.append(after)
        diagnostics = pred.get("diagnostics")
        if isinstance(diagnostics, dict) and bool(diagnostics.get("refinement_accepted", False)):
            refinement_accepts += 1
        if after is not None and context_recall is not None:
            objective_support_x.append(after)
            objective_support_y.append(context_recall)
    spearman, spearman_reason = _spearman(objective_support_x, objective_support_y)
    diagnostics: dict[str, Any] = {}
    if spearman_reason is not None:
        diagnostics["objective_support_spearman_reason"] = spearman_reason
    num_queries = len(examples)
    return RetrievalMetrics(
        num_queries=num_queries,
        mean_context_recall=aggregate(c_recalls),
        mean_context_hit=aggregate(c_hits),
        mean_anchor_recall=aggregate(a_recalls),
        mean_anchor_hit=aggregate(a_hits),
        missing_prediction_count=missing_prediction_count,
        missing_anchor_key_count=missing_anchor_key_count,
        anchor_non_empty_ratio=float(anchor_non_empty_count / num_queries) if num_queries else 0.0,
        avg_context_size=_mean(context_sizes),
        avg_latency_ms=_mean(latencies),
        objective_before_refinement_mean=_mean(objectives_before),
        objective_after_refinement_mean=_mean(objectives_after),
        refinement_accept_rate=float(refinement_accepts / num_queries) if num_queries else 0.0,
        objective_support_spearman=spearman,
        diagnostics=diagnostics,
    )


def write_metrics(path: str | Path, metrics: RetrievalMetrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
