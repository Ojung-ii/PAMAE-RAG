from __future__ import annotations

import json
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
    def to_json(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "mean_context_recall": self.mean_context_recall,
            "mean_context_hit": self.mean_context_hit,
            "mean_anchor_recall": self.mean_anchor_recall,
            "mean_anchor_hit": self.mean_anchor_hit,
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


def evaluate_predictions(examples: list[QueryExample], predictions: dict[str, dict[str, Any]]) -> RetrievalMetrics:
    c_recalls: list[float | None] = []
    c_hits: list[float | None] = []
    a_recalls: list[float | None] = []
    a_hits: list[float | None] = []
    for ex in examples:
        pred = predictions.get(ex.query_id, {})
        context = tuple(str(x) for x in pred.get("context_node_ids", []))
        anchors = tuple(str(x) for x in pred.get("anchor_ids", []))
        c_recalls.append(recall(context, ex.gold_node_ids))
        c_hits.append(hit(context, ex.gold_node_ids))
        a_recalls.append(recall(anchors, ex.gold_node_ids))
        a_hits.append(hit(anchors, ex.gold_node_ids))
    return RetrievalMetrics(len(examples), aggregate(c_recalls), aggregate(c_hits), aggregate(a_recalls), aggregate(a_hits))


def write_metrics(path: str | Path, metrics: RetrievalMetrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
