from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Sequence


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def median_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not clean:
        return None
    return float(median(clean))


def difference_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def aggregate_semantic_groups(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group", "")), []).append(row)

    current_answer = by_group.get("current_only_answer", [])
    current_non_answer = by_group.get("current_only_non_answer", [])

    mean_query_answer = mean_or_none(row.get("d_ang_query_chunk") for row in current_answer)
    mean_query_non_answer = mean_or_none(row.get("d_ang_query_chunk") for row in current_non_answer)
    mean_tree_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in current_answer)
    mean_tree_non_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in current_non_answer)

    return {
        "semantic_attribution_row_count": len(rows),
        "mean_d_ang_query_current_only_answer": mean_query_answer,
        "mean_d_ang_query_current_only_non_answer": mean_query_non_answer,
        "median_d_ang_query_current_only_answer": median_or_none(row.get("d_ang_query_chunk") for row in current_answer),
        "median_d_ang_query_current_only_non_answer": median_or_none(
            row.get("d_ang_query_chunk") for row in current_non_answer
        ),
        "mean_d_ang_tree_current_only_answer": mean_tree_answer,
        "mean_d_ang_tree_current_only_non_answer": mean_tree_non_answer,
        "median_d_ang_tree_current_only_answer": median_or_none(row.get("d_ang_chunk_tree") for row in current_answer),
        "median_d_ang_tree_current_only_non_answer": median_or_none(
            row.get("d_ang_chunk_tree") for row in current_non_answer
        ),
        "semantic_separation_query": difference_or_none(mean_query_non_answer, mean_query_answer),
        "semantic_separation_tree": difference_or_none(mean_tree_non_answer, mean_tree_answer),
    }


__all__ = ["aggregate_semantic_groups", "difference_or_none", "mean_or_none", "median_or_none"]
