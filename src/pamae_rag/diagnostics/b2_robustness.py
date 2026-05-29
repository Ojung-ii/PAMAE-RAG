from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

PAIRED_QA_METRICS = {
    "qa_f1": "f1",
    "answer_in_context": "answer_coverage",
    "rendered_recall": "context_recall",
    "context_f1": "context_f1",
    "context_tokens": "context_tokens",
    "retrieval_ms": "retrieval_ms",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def rows_by_query(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in rows if row.get("query_id") is not None}


def paired_deltas(
    *,
    candidate_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    bootstrap_samples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    candidate = rows_by_query(candidate_rows)
    baseline = rows_by_query(baseline_rows)
    query_ids = [query_id for query_id in baseline if query_id in candidate]
    rng = random.Random(seed)
    out: dict[str, Any] = {
        "num_pairs": len(query_ids),
        "query_ids_match": len(query_ids) == len(candidate) == len(baseline),
        "metrics": {},
    }
    for metric_name, field in PAIRED_QA_METRICS.items():
        values: list[float] = []
        for query_id in query_ids:
            left = _number(candidate[query_id].get(field))
            right = _number(baseline[query_id].get(field))
            if left is None or right is None:
                continue
            values.append(left - right)
        if not values:
            out["metrics"][metric_name] = {
                "mean": 0.0,
                "improved": 0,
                "tied": 0,
                "regressed": 0,
                "ci95": None,
            }
            continue
        mean = sum(values) / len(values)
        tied = sum(1 for value in values if abs(value) <= 1e-12)
        improved = sum(1 for value in values if value > 1e-12)
        regressed = sum(1 for value in values if value < -1e-12)
        ci95 = None
        if bootstrap_samples > 0 and len(values) > 1:
            means = []
            n = len(values)
            for _ in range(int(bootstrap_samples)):
                sample = [values[rng.randrange(n)] for _ in range(n)]
                means.append(sum(sample) / n)
            means.sort()
            lo = means[int(0.025 * (len(means) - 1))]
            hi = means[int(0.975 * (len(means) - 1))]
            ci95 = [lo, hi]
        out["metrics"][metric_name] = {
            "mean": mean,
            "improved": improved,
            "tied": tied,
            "regressed": regressed,
            "ci95": ci95,
        }
    return out


def flattened_delta_means(delta: dict[str, Any]) -> dict[str, float]:
    metrics = delta.get("metrics", {})
    return {
        f"{metric_name}_mean": float(metric.get("mean", 0.0))
        for metric_name, metric in metrics.items()
        if isinstance(metric, dict)
    }


__all__ = ["PAIRED_QA_METRICS", "flattened_delta_means", "paired_deltas", "rows_by_query"]
