from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPRESENTATIVE_FAILURE_TYPES = (
    "A_projection_miss",
    "B_basin_partition_miss",
    "C_representative_mismatch",
    "D_renderer_sparsity",
    "E_budget_cutoff",
    "F_generator_fail",
    "G_success",
)


@dataclass(frozen=True)
class RepresentativeTaxonomyResult:
    rows: list[dict[str, Any]]
    representative_failure_counts: dict[str, int]
    mean_d_medoid_gold: float | None
    mean_gold_distance_percentile_within_basin: float | None
    gold_on_support_tree_rate: float
    gold_path_exists_but_not_rendered_rate: float
    answer_chunk_projected_but_not_rendered_rate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "representative_failure_counts": self.representative_failure_counts,
            "mean_d_medoid_gold": self.mean_d_medoid_gold,
            "mean_gold_distance_percentile_within_basin": self.mean_gold_distance_percentile_within_basin,
            "gold_on_support_tree_rate": self.gold_on_support_tree_rate,
            "gold_path_exists_but_not_rendered_rate": self.gold_path_exists_but_not_rendered_rate,
            "answer_chunk_projected_but_not_rendered_rate": self.answer_chunk_projected_but_not_rendered_rate,
            "rows": self.rows,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _by_query_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("query_id")): row for row in rows if row.get("query_id") is not None}


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _is_success(qa_row: dict[str, Any]) -> bool:
    return _float(qa_row.get("exact_match")) > 0.0 or _float(qa_row.get("f1")) > 0.0


def _path_payload(retrieval_row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = retrieval_row.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("path_realizability"), dict):
        return diagnostics["path_realizability"]
    if isinstance(retrieval_row.get("path_realizability"), dict):
        return retrieval_row["path_realizability"]
    return {}


def classify_representative_failure(path_payload: dict[str, Any], qa_row: dict[str, Any]) -> str:
    gold_rows = [row for row in path_payload.get("gold_rows", []) if isinstance(row, dict)]
    answer_trace = path_payload.get("answer_trace") if isinstance(path_payload.get("answer_trace"), dict) else {}
    projected_hit = any(bool(row.get("gold_in_projected")) for row in gold_rows) or bool(
        answer_trace.get("answer_chunk_in_projected")
    )
    gold_in_selected_basin = any(bool(row.get("gold_in_selected_basin")) for row in gold_rows) or bool(
        answer_trace.get("answer_chunk_in_selected_basin")
    )
    gold_on_support_tree = any(bool(row.get("gold_on_existing_support_tree")) for row in gold_rows) or bool(
        answer_trace.get("answer_chunk_on_support_tree")
    )
    medoid_to_gold_path_exists = any(bool(row.get("medoid_to_gold_path_exists")) for row in gold_rows)
    gold_rendered = any(bool(row.get("gold_rendered")) for row in gold_rows) or bool(
        answer_trace.get("answer_chunk_rendered")
    )
    budget_cutoff = any(bool(row.get("render_budget_cutoff_before_gold")) for row in gold_rows)
    answer_in_context = any(bool(row.get("answer_in_context")) for row in gold_rows) or bool(
        answer_trace.get("answer_chunk_rendered")
    )
    qa_success = _is_success(qa_row)

    if not projected_hit:
        return "A_projection_miss"
    if not gold_in_selected_basin:
        return "B_basin_partition_miss"
    if gold_in_selected_basin and not gold_on_support_tree and not medoid_to_gold_path_exists:
        return "C_representative_mismatch"
    if medoid_to_gold_path_exists and not gold_rendered and not budget_cutoff:
        return "D_renderer_sparsity"
    if budget_cutoff:
        return "E_budget_cutoff"
    if answer_in_context and not qa_success:
        return "F_generator_fail"
    if gold_rendered and answer_in_context and qa_success:
        return "G_success"
    return "F_generator_fail"


def representative_taxonomy_rows(
    retrieval_rows: Iterable[dict[str, Any]],
    qa_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for retrieval_row in retrieval_rows:
        query_id = str(retrieval_row.get("query_id"))
        qa_row = qa_rows.get(query_id, {"query_id": query_id})
        payload = _path_payload(retrieval_row)
        gold_rows = [row for row in payload.get("gold_rows", []) if isinstance(row, dict)]
        answer_trace = payload.get("answer_trace") if isinstance(payload.get("answer_trace"), dict) else {}
        basin_rows = [row for row in payload.get("basin_position_rows", []) if isinstance(row, dict)]
        failure = classify_representative_failure(payload, qa_row)
        d_values = [
            float(row["d_medoid_gold"])
            for row in gold_rows
            if isinstance(row.get("d_medoid_gold"), (int, float)) and not isinstance(row.get("d_medoid_gold"), bool)
        ]
        percentiles = [
            float(row["gold_distance_percentile_within_basin"])
            for row in basin_rows
            if isinstance(row.get("gold_distance_percentile_within_basin"), (int, float))
            and not isinstance(row.get("gold_distance_percentile_within_basin"), bool)
        ]
        rows.append(
            {
                "query_id": query_id,
                "failure_type": failure,
                "projected_hit": any(bool(row.get("gold_in_projected")) for row in gold_rows)
                or bool(answer_trace.get("answer_chunk_in_projected")),
                "gold_in_selected_basin": any(bool(row.get("gold_in_selected_basin")) for row in gold_rows),
                "answer_chunk_in_selected_basin": bool(answer_trace.get("answer_chunk_in_selected_basin")),
                "gold_on_support_tree": any(bool(row.get("gold_on_existing_support_tree")) for row in gold_rows),
                "medoid_to_gold_path_exists": any(bool(row.get("medoid_to_gold_path_exists")) for row in gold_rows),
                "gold_rendered": any(bool(row.get("gold_rendered")) for row in gold_rows),
                "answer_chunk_rendered": bool(answer_trace.get("answer_chunk_rendered")),
                "answer_in_context": any(bool(row.get("answer_in_context")) for row in gold_rows)
                or bool(answer_trace.get("answer_chunk_rendered")),
                "qa_f1": _float(qa_row.get("f1")),
                "exact_match": _float(qa_row.get("exact_match")),
                "mean_d_medoid_gold": _mean(d_values),
                "mean_gold_distance_percentile_within_basin": _mean(percentiles),
                "gold_path_exists_but_not_rendered": any(
                    bool(row.get("medoid_to_gold_path_exists")) and not bool(row.get("gold_rendered"))
                    for row in gold_rows
                ),
                "answer_chunk_projected_but_not_rendered": bool(answer_trace.get("answer_chunk_in_projected"))
                and not bool(answer_trace.get("answer_chunk_rendered")),
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("failure_type")) for row in rows)
    return {failure_type: int(counter.get(failure_type, 0)) for failure_type in REPRESENTATIVE_FAILURE_TYPES}


def summarize_representative_taxonomy(rows: list[dict[str, Any]]) -> RepresentativeTaxonomyResult:
    d_values = [
        float(row["mean_d_medoid_gold"])
        for row in rows
        if isinstance(row.get("mean_d_medoid_gold"), (int, float))
        and not isinstance(row.get("mean_d_medoid_gold"), bool)
    ]
    percentile_values = [
        float(row["mean_gold_distance_percentile_within_basin"])
        for row in rows
        if isinstance(row.get("mean_gold_distance_percentile_within_basin"), (int, float))
        and not isinstance(row.get("mean_gold_distance_percentile_within_basin"), bool)
    ]
    denom = max(len(rows), 1)
    return RepresentativeTaxonomyResult(
        rows=rows,
        representative_failure_counts=_counts(rows),
        mean_d_medoid_gold=_mean(d_values),
        mean_gold_distance_percentile_within_basin=_mean(percentile_values),
        gold_on_support_tree_rate=sum(1.0 for row in rows if row.get("gold_on_support_tree")) / denom,
        gold_path_exists_but_not_rendered_rate=sum(
            1.0 for row in rows if row.get("gold_path_exists_but_not_rendered")
        )
        / denom,
        answer_chunk_projected_but_not_rendered_rate=sum(
            1.0 for row in rows if row.get("answer_chunk_projected_but_not_rendered")
        )
        / denom,
    )


def analyze_representative_taxonomy(
    *,
    retrieval_path: str | Path,
    qa_path: str | Path,
) -> RepresentativeTaxonomyResult:
    retrieval_rows = _read_jsonl(retrieval_path)
    qa_rows = _by_query_id(_read_jsonl(qa_path))
    rows = representative_taxonomy_rows(retrieval_rows, qa_rows)
    return summarize_representative_taxonomy(rows)

