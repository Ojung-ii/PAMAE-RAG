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
    retrieval_variant: str | None
    renderer: str | None
    relevance_mode: str | None
    k: int | None
    k_max: int | None
    max_context_nodes: int | None
    max_context_tokens: int | None
    mean_context_recall: float
    mean_context_hit: float
    mean_context_precision: float | None
    mean_context_f1: float | None
    mean_context_recall_per_node: float | None
    mean_context_recall_per_1k_tokens: float | None
    mean_anchor_recall: float
    mean_anchor_hit: float
    mean_anchor_precision: float | None
    mean_anchor_f1: float | None
    mean_anchor_recall_per_anchor: float | None
    missing_prediction_count: int
    missing_anchor_key_count: int
    anchor_non_empty_ratio: float
    avg_context_size: float
    avg_context_tokens: float
    context_node_budget_satisfied_rate: float | None
    context_token_budget_satisfied_rate: float | None
    avg_latency_ms: float
    objective_before_refinement_mean: float
    objective_after_refinement_mean: float
    refinement_accept_rate: float
    objective_support_spearman: float | None
    diagnostics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "retrieval_variant": self.retrieval_variant,
            "renderer": self.renderer,
            "relevance_mode": self.relevance_mode,
            "k": self.k,
            "k_max": self.k_max,
            "max_context_nodes": self.max_context_nodes,
            "max_context_tokens": self.max_context_tokens,
            "mean_context_recall": self.mean_context_recall,
            "mean_context_hit": self.mean_context_hit,
            "mean_context_precision": self.mean_context_precision,
            "mean_context_f1": self.mean_context_f1,
            "mean_context_recall_per_node": self.mean_context_recall_per_node,
            "mean_context_recall_per_1k_tokens": self.mean_context_recall_per_1k_tokens,
            "mean_anchor_recall": self.mean_anchor_recall,
            "mean_anchor_hit": self.mean_anchor_hit,
            "mean_anchor_precision": self.mean_anchor_precision,
            "mean_anchor_f1": self.mean_anchor_f1,
            "mean_anchor_recall_per_anchor": self.mean_anchor_recall_per_anchor,
            "missing_prediction_count": self.missing_prediction_count,
            "missing_anchor_key_count": self.missing_anchor_key_count,
            "anchor_non_empty_ratio": self.anchor_non_empty_ratio,
            "avg_context_size": self.avg_context_size,
            "avg_context_tokens": self.avg_context_tokens,
            "context_node_budget_satisfied_rate": self.context_node_budget_satisfied_rate,
            "context_token_budget_satisfied_rate": self.context_token_budget_satisfied_rate,
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


def precision(selected: list[str] | tuple[str, ...], gold: set[str] | frozenset[str]) -> float | None:
    if not selected:
        return None
    return len(set(selected) & set(gold)) / len(set(selected))


def f1_score(precision_value: float | None, recall_value: float | None) -> float | None:
    if precision_value is None or recall_value is None:
        return None
    if precision_value + recall_value == 0.0:
        return 0.0
    return 2.0 * precision_value * recall_value / (precision_value + recall_value)


def aggregate(values: list[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return float(sum(nums) / len(nums)) if nums else 0.0


def aggregate_optional(values: list[float | None], diagnostics: dict[str, Any], name: str) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        diagnostics[f"{name}_reason"] = "no computable query values"
        return None
    return float(sum(nums) / len(nums))


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


def _context_token_count(ex: QueryExample, context: tuple[str, ...], pred: dict[str, Any]) -> float:
    diagnostics = pred.get("diagnostics")
    if isinstance(diagnostics, dict):
        value = _float_or_none(diagnostics.get("final_context_tokens"))
        if value is not None:
            return value
    token_by_id = {node.node_id: max(1, int(node.token_count)) for node in ex.nodes}
    return float(sum(token_by_id.get(node_id, 0) for node_id in context))


def _budget_flag(pred: dict[str, Any], key: str, context: tuple[str, ...]) -> float | None:
    diagnostics = pred.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    value = diagnostics.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if key == "node_budget_satisfied":
        max_nodes = diagnostics.get("max_context_nodes")
        if isinstance(max_nodes, int) and max_nodes > 0:
            return 1.0 if len(context) <= max_nodes else 0.0
    if key == "token_budget_satisfied":
        max_tokens = diagnostics.get("max_context_tokens")
        final_tokens = diagnostics.get("final_context_tokens")
        if isinstance(max_tokens, (int, float)) and isinstance(final_tokens, (int, float)):
            return 1.0 if float(final_tokens) <= float(max_tokens) else 0.0
    return None


def _first_value(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def evaluate_predictions(examples: list[QueryExample], predictions: dict[str, dict[str, Any]]) -> RetrievalMetrics:
    c_recalls: list[float | None] = []
    c_hits: list[float | None] = []
    c_precisions: list[float | None] = []
    c_f1s: list[float | None] = []
    c_recall_per_node: list[float | None] = []
    c_recall_per_1k_tokens: list[float | None] = []
    a_recalls: list[float | None] = []
    a_hits: list[float | None] = []
    a_precisions: list[float | None] = []
    a_f1s: list[float | None] = []
    a_recall_per_anchor: list[float | None] = []
    missing_prediction_count = 0
    missing_anchor_key_count = 0
    anchor_non_empty_count = 0
    context_sizes: list[float] = []
    context_tokens: list[float] = []
    node_budget_flags: list[float | None] = []
    token_budget_flags: list[float | None] = []
    latencies: list[float] = []
    objectives_before: list[float] = []
    objectives_after: list[float] = []
    refinement_accepts = 0
    objective_support_x: list[float] = []
    objective_support_y: list[float] = []
    retrieval_variants: list[Any] = []
    renderers: list[Any] = []
    relevance_modes: list[Any] = []
    ks: list[Any] = []
    k_maxes: list[Any] = []
    max_context_nodes_values: list[Any] = []
    max_context_tokens_values: list[Any] = []
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
        context_precision = precision(context, ex.gold_node_ids)
        context_f1 = f1_score(context_precision, context_recall)
        anchor_recall = recall(anchors, ex.gold_node_ids)
        anchor_precision = precision(anchors, ex.gold_node_ids)
        anchor_f1 = f1_score(anchor_precision, anchor_recall)
        token_count = _context_token_count(ex, context, pred)
        c_recalls.append(context_recall)
        c_hits.append(hit(context, ex.gold_node_ids))
        c_precisions.append(context_precision)
        c_f1s.append(context_f1)
        c_recall_per_node.append(
            None if context_recall is None else context_recall / max(float(len(context)), 1.0)
        )
        c_recall_per_1k_tokens.append(
            None if context_recall is None else context_recall / max(token_count / 1000.0, 1e-12)
        )
        a_recalls.append(anchor_recall)
        a_hits.append(hit(anchors, ex.gold_node_ids))
        a_precisions.append(anchor_precision)
        a_f1s.append(anchor_f1)
        a_recall_per_anchor.append(
            None if anchor_recall is None else anchor_recall / max(float(len(anchors)), 1.0)
        )
        if anchors:
            anchor_non_empty_count += 1
        context_sizes.append(float(len(context)))
        context_tokens.append(token_count)
        node_budget_flags.append(_budget_flag(pred, "node_budget_satisfied", context))
        token_budget_flags.append(_budget_flag(pred, "token_budget_satisfied", context))
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
        if isinstance(diagnostics, dict):
            retrieval_variants.append(diagnostics.get("retrieval_variant"))
            renderers.append(diagnostics.get("renderer"))
            relevance_modes.append(diagnostics.get("relevance_mode"))
            ks.append(diagnostics.get("k"))
            k_maxes.append(diagnostics.get("k_max"))
            max_context_nodes_values.append(diagnostics.get("max_context_nodes"))
            max_context_tokens_values.append(diagnostics.get("max_context_tokens"))
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
        retrieval_variant=_first_value(retrieval_variants),
        renderer=_first_value(renderers),
        relevance_mode=_first_value(relevance_modes),
        k=_first_value(ks),
        k_max=_first_value(k_maxes),
        max_context_nodes=_first_value(max_context_nodes_values),
        max_context_tokens=_first_value(max_context_tokens_values),
        mean_context_recall=aggregate(c_recalls),
        mean_context_hit=aggregate(c_hits),
        mean_context_precision=aggregate_optional(c_precisions, diagnostics, "mean_context_precision"),
        mean_context_f1=aggregate_optional(c_f1s, diagnostics, "mean_context_f1"),
        mean_context_recall_per_node=aggregate_optional(
            c_recall_per_node, diagnostics, "mean_context_recall_per_node"
        ),
        mean_context_recall_per_1k_tokens=aggregate_optional(
            c_recall_per_1k_tokens, diagnostics, "mean_context_recall_per_1k_tokens"
        ),
        mean_anchor_recall=aggregate(a_recalls),
        mean_anchor_hit=aggregate(a_hits),
        mean_anchor_precision=aggregate_optional(a_precisions, diagnostics, "mean_anchor_precision"),
        mean_anchor_f1=aggregate_optional(a_f1s, diagnostics, "mean_anchor_f1"),
        mean_anchor_recall_per_anchor=aggregate_optional(
            a_recall_per_anchor, diagnostics, "mean_anchor_recall_per_anchor"
        ),
        missing_prediction_count=missing_prediction_count,
        missing_anchor_key_count=missing_anchor_key_count,
        anchor_non_empty_ratio=float(anchor_non_empty_count / num_queries) if num_queries else 0.0,
        avg_context_size=_mean(context_sizes),
        avg_context_tokens=_mean(context_tokens),
        context_node_budget_satisfied_rate=aggregate_optional(
            node_budget_flags, diagnostics, "context_node_budget_satisfied_rate"
        ),
        context_token_budget_satisfied_rate=aggregate_optional(
            token_budget_flags, diagnostics, "context_token_budget_satisfied_rate"
        ),
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
