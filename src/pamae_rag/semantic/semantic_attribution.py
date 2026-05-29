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
    shell1_answer = by_group.get("shell1_answer", [])
    shell1_non_answer = by_group.get("shell1_non_answer", [])

    mean_query_answer = mean_or_none(row.get("d_ang_query_chunk") for row in current_answer)
    mean_query_non_answer = mean_or_none(row.get("d_ang_query_chunk") for row in current_non_answer)
    mean_tree_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in current_answer)
    mean_tree_non_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in current_non_answer)
    mean_shell1_query_answer = mean_or_none(row.get("d_ang_query_chunk") for row in shell1_answer)
    mean_shell1_query_non_answer = mean_or_none(row.get("d_ang_query_chunk") for row in shell1_non_answer)
    mean_shell1_tree_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in shell1_answer)
    mean_shell1_tree_non_answer = mean_or_none(row.get("d_ang_chunk_tree") for row in shell1_non_answer)

    payload: dict[str, Any] = {
        "semantic_attribution_row_count": len(rows),
    }
    for group in sorted(group for group in by_group if group):
        group_rows = by_group[group]
        payload[f"{group}_count"] = len(group_rows)
        payload[f"mean_d_ang_query_{group}"] = mean_or_none(row.get("d_ang_query_chunk") for row in group_rows)
        payload[f"median_d_ang_query_{group}"] = median_or_none(row.get("d_ang_query_chunk") for row in group_rows)
        payload[f"mean_d_ang_tree_{group}"] = mean_or_none(row.get("d_ang_chunk_tree") for row in group_rows)
        payload[f"median_d_ang_tree_{group}"] = median_or_none(row.get("d_ang_chunk_tree") for row in group_rows)

    payload.update(
        {
            "mean_d_ang_query_current_only_answer": mean_query_answer,
            "mean_d_ang_query_current_only_non_answer": mean_query_non_answer,
            "median_d_ang_query_current_only_answer": median_or_none(
                row.get("d_ang_query_chunk") for row in current_answer
            ),
            "median_d_ang_query_current_only_non_answer": median_or_none(
                row.get("d_ang_query_chunk") for row in current_non_answer
            ),
            "mean_d_ang_tree_current_only_answer": mean_tree_answer,
            "mean_d_ang_tree_current_only_non_answer": mean_tree_non_answer,
            "median_d_ang_tree_current_only_answer": median_or_none(
                row.get("d_ang_chunk_tree") for row in current_answer
            ),
            "median_d_ang_tree_current_only_non_answer": median_or_none(
                row.get("d_ang_chunk_tree") for row in current_non_answer
            ),
            "semantic_separation_query": difference_or_none(mean_query_non_answer, mean_query_answer),
            "semantic_separation_tree": difference_or_none(mean_tree_non_answer, mean_tree_answer),
            "semantic_separation_query_current_only": difference_or_none(mean_query_non_answer, mean_query_answer),
            "semantic_separation_tree_current_only": difference_or_none(mean_tree_non_answer, mean_tree_answer),
            "semantic_separation_query_shell1": difference_or_none(
                mean_shell1_query_non_answer,
                mean_shell1_query_answer,
            ),
            "semantic_separation_tree_shell1": difference_or_none(
                mean_shell1_tree_non_answer,
                mean_shell1_tree_answer,
            ),
        }
    )
    return payload


__all__ = ["aggregate_semantic_groups", "difference_or_none", "mean_or_none", "median_or_none"]
