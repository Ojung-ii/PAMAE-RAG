from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.schema import QueryExample


FAILURE_TYPES = (
    "projection_miss",
    "selection_miss",
    "rendering_miss",
    "qa_fail",
    "success",
)


@dataclass(frozen=True)
class FailureTaxonomyResult:
    rows: list[dict[str, Any]]
    failure_type_counts: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "failure_type_counts": self.failure_type_counts,
            "rows": self.rows,
        }


def _read_jsonl_rows(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _stage(row: dict[str, Any], name: str) -> dict[str, Any]:
    stage_rows = row.get("stage_diagnostics")
    if isinstance(stage_rows, dict) and isinstance(stage_rows.get(name), dict):
        return stage_rows[name]
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        nested = diagnostics.get("stage_diagnostics")
        if isinstance(nested, dict) and isinstance(nested.get(name), dict):
            return nested[name]
    return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _positive_stage_survival(row: dict[str, Any], stage: str) -> bool:
    value = _stage(row, stage).get("gold_supporting_evidence_survival")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0


def _stage_gold_ids(row: dict[str, Any], stage: str, gold_ids: list[str] | None = None) -> list[str]:
    explicit = _string_list(_stage(row, stage).get("gold_supporting_node_ids"))
    if explicit:
        return explicit
    if gold_ids is not None and _positive_stage_survival(row, stage):
        return list(gold_ids)
    return []


def _selected_basin_gold_ids(row: dict[str, Any]) -> list[str]:
    local = _stage(row, "local_refinement")
    basin_ids = _string_list(local.get("selected_basin_gold_chunk_ids"))
    if basin_ids:
        return basin_ids
    return _stage_gold_ids(row, "local_refinement", _string_list(row.get("gold_chunk_ids")))


def _rendered_gold_ids(row: dict[str, Any]) -> list[str]:
    gold_ids = _string_list(row.get("gold_chunk_ids"))
    rendered = _stage_gold_ids(row, "context_rendering", gold_ids)
    if rendered:
        return rendered
    context_ids = set(_string_list(row.get("context_node_ids")))
    return sorted(context_ids & set(gold_ids))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _answer_in_context(row: dict[str, Any]) -> bool:
    return _float(row, "answer_coverage") > 0.0


def classify_failure(row: dict[str, Any]) -> str:
    gold_ids = _string_list(row.get("gold_chunk_ids"))
    projected_hit = bool(_stage_gold_ids(row, "content_graph_projection", gold_ids))
    selected_basin_hit = bool(_selected_basin_gold_ids(row))
    rendered_hit = bool(_rendered_gold_ids(row))
    answer_in_context = _answer_in_context(row)
    qa_f1 = _float(row, "f1")
    exact_match = _float(row, "exact_match")

    if exact_match > 0.0 or qa_f1 >= 1.0:
        return "success"
    if not projected_hit:
        return "projection_miss"
    if not selected_basin_hit:
        return "selection_miss"
    if rendered_hit or answer_in_context:
        return "qa_fail"
    return "rendering_miss"


def taxonomy_rows(
    examples: Iterable[QueryExample],
    qa_rows: dict[str, dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    retrieval_rows = retrieval_rows or {}
    rows: list[dict[str, Any]] = []
    for example in examples:
        qa_row = dict(qa_rows.get(example.query_id, {"query_id": example.query_id}))
        retrieval_row = retrieval_rows.get(example.query_id)
        if retrieval_row is not None and "stage_diagnostics" not in qa_row:
            diagnostics = retrieval_row.get("diagnostics")
            if isinstance(diagnostics, dict) and isinstance(diagnostics.get("stage_diagnostics"), dict):
                qa_row["stage_diagnostics"] = diagnostics["stage_diagnostics"]

        gold_ids = sorted(str(node_id) for node_id in example.gold_node_ids)
        qa_row["gold_chunk_ids"] = gold_ids
        projected_gold = _stage_gold_ids(qa_row, "content_graph_projection", gold_ids)
        selected_basin_gold = _selected_basin_gold_ids(qa_row)
        rendered_gold = _rendered_gold_ids(qa_row)
        row = {
            "query_id": example.query_id,
            "failure_type": classify_failure(qa_row),
            "projected_hit": bool(projected_gold),
            "selected_basin_hit": bool(selected_basin_gold),
            "rendered_hit": bool(rendered_gold),
            "answer_in_context": _answer_in_context(qa_row),
            "qa_f1": _float(qa_row, "f1"),
            "gold_chunk_ids": gold_ids,
            "projected_gold_chunk_ids": projected_gold,
            "selected_basin_gold_chunk_ids": selected_basin_gold,
            "rendered_gold_chunk_ids": rendered_gold,
        }
        rows.append(row)
    return rows


def summarize_taxonomy(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("failure_type")) for row in rows)
    return {failure_type: int(counts.get(failure_type, 0)) for failure_type in FAILURE_TYPES}


def analyze_failure_taxonomy(
    input_path: str | Path,
    qa_path: str | Path,
    *,
    retrieval_path: str | Path | None = None,
    limit: int | None = None,
) -> FailureTaxonomyResult:
    examples = read_jsonl(input_path, limit=limit)
    qa_rows = _read_jsonl_rows(qa_path)
    retrieval_rows = _read_jsonl_rows(retrieval_path)
    rows = taxonomy_rows(examples, qa_rows, retrieval_rows)
    return FailureTaxonomyResult(
        rows=rows,
        failure_type_counts=summarize_taxonomy(rows),
    )
