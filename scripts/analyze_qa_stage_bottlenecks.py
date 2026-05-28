#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _stage_value(row: dict[str, Any], stage: str, key: str) -> float | None:
    value = row.get("stage_diagnostics", {}).get(stage, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _bucket(row: dict[str, Any]) -> str:
    projection = _stage_value(row, "content_graph_projection", "gold_supporting_evidence_survival")
    pre_local = _stage_value(
        row,
        "local_refinement",
        "pre_refinement_gold_supporting_evidence_survival",
    )
    local = _stage_value(row, "local_refinement", "gold_supporting_evidence_survival")
    rendered = _stage_value(row, "context_rendering", "rendered_recall")
    answer_coverage = row.get("answer_coverage")
    selected_answer_coverage = row.get("selected_answer_coverage")
    f1 = float(row.get("f1") or 0.0)
    if projection is not None and projection <= 0.0:
        return "projection_loss"
    if pre_local is not None and pre_local <= 0.0:
        return "pre_refinement_anchor_loss"
    if pre_local is not None and local is not None and pre_local > 0.0 and local <= 0.0:
        return "refinement_update_loss"
    if local is not None and local <= 0.0:
        return "local_refinement_loss"
    if rendered is not None and rendered <= 0.0:
        return "context_rendering_loss"
    if answer_coverage == 0.0:
        return "answer_absent_from_context"
    if selected_answer_coverage == 0.0:
        return "answer_present_not_selected"
    if f1 <= 0.0:
        return "selected_answer_low_metric_match"
    return "qa_partial_or_success"


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_f1": None,
            "mean_context_recall": None,
            "mean_answer_coverage": None,
            "mean_selected_answer_coverage": None,
            "sample_query_ids": [],
        }
    return {
        "count": len(rows),
        "mean_f1": mean(_float(row, "content_f1") for row in rows),
        "mean_context_recall": mean(_float(row, "content_context_recall") for row in rows),
        "mean_answer_coverage": mean(_float(row, "content_answer_coverage") for row in rows),
        "mean_selected_answer_coverage": mean(
            _float(row, "content_selected_answer_coverage") for row in rows
        ),
        "sample_query_ids": [str(row["query_id"]) for row in rows[:10]],
    }


def analyze(
    baseline_qa: str | Path,
    content_qa: str | Path,
    oracle_qa: str | Path | None = None,
) -> dict[str, Any]:
    baseline = _read_rows(baseline_qa)
    content = _read_rows(content_qa)
    oracle = _read_rows(oracle_qa) if oracle_qa else {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(baseline) & set(content)):
        baseline_row = baseline[query_id]
        content_row = content[query_id]
        oracle_row = oracle.get(query_id, {})
        baseline_bucket = _bucket(baseline_row)
        content_bucket = _bucket(content_row)
        transition = f"{baseline_bucket}->{content_bucket}"
        transitions[transition] += 1
        delta_f1 = _float(content_row, "f1") - _float(baseline_row, "f1")
        row = {
            "query_id": query_id,
            "baseline_bucket": baseline_bucket,
            "content_bucket": content_bucket,
            "transition": transition,
            "delta_f1": delta_f1,
            "baseline_f1": _float(baseline_row, "f1"),
            "content_f1": _float(content_row, "f1"),
            "oracle_f1": _float(oracle_row, "f1") if oracle_row else None,
            "baseline_context_recall": _float(baseline_row, "context_recall"),
            "content_context_recall": _float(content_row, "context_recall"),
            "baseline_answer_coverage": _float(baseline_row, "answer_coverage"),
            "content_answer_coverage": _float(content_row, "answer_coverage"),
            "baseline_selected_answer_coverage": _float(baseline_row, "selected_answer_coverage"),
            "content_selected_answer_coverage": _float(content_row, "selected_answer_coverage"),
            "content_pre_refinement_survival": _stage_value(
                content_row,
                "local_refinement",
                "pre_refinement_gold_supporting_evidence_survival",
            ),
            "content_post_refinement_survival": _stage_value(
                content_row,
                "local_refinement",
                "gold_supporting_evidence_survival",
            ),
        }
        rows.append(row)
        groups[content_bucket].append(row)
    return {
        "num_queries": len(rows),
        "content_bucket_summary": {
            key: _summarize_group(group_rows) for key, group_rows in sorted(groups.items())
        },
        "transition_counts": dict(sorted(transitions.items())),
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# QA Stage Bottleneck Analysis",
        "",
        "## Content Buckets",
        "",
        "| bucket | count | mean_f1 | mean_context_recall | mean_answer_coverage | mean_selected_answer_coverage | sample_query_ids |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bucket, row in result["content_bucket_summary"].items():
        samples = ", ".join(f"`{query_id}`" for query_id in row["sample_query_ids"])
        lines.append(
            "| "
            + " | ".join(
                [
                    bucket,
                    _fmt(row["count"]),
                    _fmt(row["mean_f1"]),
                    _fmt(row["mean_context_recall"]),
                    _fmt(row["mean_answer_coverage"]),
                    _fmt(row["mean_selected_answer_coverage"]),
                    samples,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Bucket Transitions", ""])
    for transition, count in result["transition_counts"].items():
        lines.append(f"- `{transition}`: {count}")
    lines.extend(
        [
            "",
            "## Largest Content Regressions",
            "",
            "| query_id | baseline_bucket | content_bucket | delta_f1 | baseline_f1 | content_f1 | oracle_f1 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    regressions = sorted(result["rows"], key=lambda row: (row["delta_f1"], row["query_id"]))[:10]
    for row in regressions:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['query_id']}`",
                    row["baseline_bucket"],
                    row["content_bucket"],
                    _fmt(row["delta_f1"]),
                    _fmt(row["baseline_f1"]),
                    _fmt(row["content_f1"]),
                    _fmt(row["oracle_f1"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-query QA bottlenecks from stage diagnostics.")
    parser.add_argument("--baseline-qa", required=True)
    parser.add_argument("--content-qa", required=True)
    parser.add_argument("--oracle-qa", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    result = analyze(args.baseline_qa, args.content_qa, args.oracle_qa)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
