from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from pamae_rag.eval.support_recall import f1_score, precision, recall


STAGE_NAMES = (
    "query_anchor_construction",
    "candidate_generation",
    "content_graph_projection",
    "local_refinement",
    "reranking_scoring",
    "context_rendering",
    "final_qa",
)


def _ids(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    return tuple(str(value) for value in values)


def _support_counts(selected: tuple[str, ...], gold: set[str] | frozenset[str]) -> tuple[int, int]:
    selected_set = set(selected)
    gold_set = set(gold)
    return len(selected_set & gold_set), len(gold_set)


def _pair_metrics(selected: tuple[str, ...], gold: set[str] | frozenset[str]) -> dict[str, Any]:
    selected_list = list(selected)
    rec = recall(tuple(selected_list), gold)
    prec = precision(tuple(selected_list), gold)
    return {
        "recall": rec,
        "precision": prec,
        "f1": f1_score(prec, rec),
    }


def make_stage_metrics(
    *,
    stage: str,
    selected_node_ids: Iterable[str] | None,
    gold_node_ids: set[str] | frozenset[str],
    candidate_node_ids: Iterable[str] | None = None,
    context_node_ids: Iterable[str] | None = None,
    rendered_node_ids: Iterable[str] | None = None,
    token_count: int | float | None = None,
    latency_ms: int | float | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = _ids(selected_node_ids)
    candidate = _ids(candidate_node_ids)
    context = _ids(context_node_ids)
    rendered = _ids(rendered_node_ids)
    survived, total = _support_counts(selected, gold_node_ids)
    candidate_metrics = _pair_metrics(candidate, gold_node_ids) if candidate_node_ids is not None else {}
    context_metrics = _pair_metrics(context, gold_node_ids) if context_node_ids is not None else {}
    rendered_metrics = _pair_metrics(rendered, gold_node_ids) if rendered_node_ids is not None else {}
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "survivor_count": len(set(selected)),
        "gold_support_count": total,
        "gold_support_surviving_count": survived,
        "gold_supporting_evidence_survival": recall(selected, gold_node_ids),
        "candidate_recall": candidate_metrics.get("recall"),
        "candidate_precision": candidate_metrics.get("precision"),
        "candidate_f1": candidate_metrics.get("f1"),
        "context_recall": context_metrics.get("recall"),
        "context_precision": context_metrics.get("precision"),
        "context_f1": context_metrics.get("f1"),
        "rendered_recall": rendered_metrics.get("recall"),
        "rendered_precision": rendered_metrics.get("precision"),
        "rendered_f1": rendered_metrics.get("f1"),
        "token_count": None if token_count is None else float(token_count),
        "latency_ms": None if latency_ms is None else float(latency_ms),
    }
    if extra:
        payload.update(extra)
    return payload


def aggregate_stage_diagnostics(rows: Iterable[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for stage, metrics in row.items():
            if isinstance(metrics, dict):
                by_stage[stage].append(metrics)

    out: dict[str, Any] = {}
    for stage in sorted(by_stage):
        stage_rows = by_stage[stage]
        numeric: dict[str, list[float]] = defaultdict(list)
        statuses: Counter[str] = Counter()
        for metrics in stage_rows:
            status = metrics.get("status")
            if status is not None:
                statuses[str(status)] += 1
            for key, value in metrics.items():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    numeric[key].append(float(value))
        out[stage] = {
            "num_queries": len(stage_rows),
            "status_counts": dict(sorted(statuses.items())),
            "mean": {
                key: float(sum(values) / len(values))
                for key, values in sorted(numeric.items())
                if values
            },
        }
    return out
