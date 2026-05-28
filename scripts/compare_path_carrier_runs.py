#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.diagnostics.path_carrier_taxonomy import aggregate_path_carrier_taxonomy

REFERENCE_RUN = "entity_chunk_reference_current_renderer"
METRIC_RUN = "entity_chunk_reference_metric_path_carrier"
RUN_ORDER = (
    REFERENCE_RUN,
    METRIC_RUN,
    "entity_chunk_reference_metric_path_carrier_no_medoids",
    "entity_chunk_reference_metric_path_carrier_medoids_first",
    "entity_chunk_reference_current_answer_role_oracle",
    "entity_chunk_reference_support_tree_answer_oracle",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _stage_mean(metrics: dict[str, Any], stage: str, key: str) -> float:
    stage_diag = metrics.get("qa_metrics", {}).get("stage_diagnostics", {})
    if isinstance(stage_diag, dict):
        value = stage_diag.get(stage, {}).get("mean", {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "answer_carrier_metrics.json")
    carrier = metrics.get("answer_carrier_attribution")
    if not isinstance(carrier, dict):
        carrier = {}
    support = metrics.get("support_tree_carrier")
    if not isinstance(support, dict):
        support = {}
    rendered_recall = float(metrics.get("rendered_recall", 0.0))
    if rendered_recall == 0.0:
        rendered_recall = _stage_mean(metrics, "context_rendering", "rendered_recall")
    return {
        "run": run_dir.name,
        "renderer_mode": metrics.get("renderer_mode", ""),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": rendered_recall,
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
        "selected_medoid_answer_availability": float(carrier.get("selected_medoid_answer_availability", 0.0)),
        "current_minus_medoid_answer_gap": float(carrier.get("current_minus_medoid_answer_gap", 0.0)),
        "answer_chunk_projected_rate": float(carrier.get("answer_chunk_projected_rate", 0.0)),
        "answer_chunk_selected_basin_rate": float(carrier.get("answer_chunk_selected_basin_rate", 0.0)),
        "answer_on_refined_support_tree_rate": float(support.get("answer_on_refined_support_tree_rate", 0.0)),
        "answer_metric_path_rendered_rate": float(support.get("answer_metric_path_rendered_rate", 0.0)),
        "current_minus_refined_tree_gap": float(support.get("current_minus_refined_tree_gap", 0.0)),
        "current_minus_metric_path_gap": float(support.get("current_minus_metric_path_gap", 0.0)),
        "mean_d_answer_to_support_tree": float(support.get("mean_d_answer_to_support_tree", 0.0)),
        "answer_path_role_distribution": support.get("answer_path_role_distribution", {}),
    }


def _combined_taxonomy(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in (REFERENCE_RUN, METRIC_RUN):
        rows.extend(_read_jsonl(root / run / "support_tree_carrier_trace.jsonl"))
    return aggregate_path_carrier_taxonomy(rows) if rows else {"path_carrier_failure_counts": {}}


def _cross_role_distribution(root: Path) -> dict[str, int]:
    current_rows = _read_jsonl(root / REFERENCE_RUN / "support_tree_carrier_trace.jsonl")
    metric_rows = _read_jsonl(root / METRIC_RUN / "support_tree_carrier_trace.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in [*current_rows, *metric_rows]:
        grouped.setdefault(str(row.get("query_id")), []).append(row)
    counts = {
        "medoid": 0,
        "anchor_medoid_path": 0,
        "medoid_medoid_path": 0,
        "current_only_hidden_recovery": 0,
        "budget_cutoff": 0,
    }
    for query_rows in grouped.values():
        if any(bool(row.get("answer_chunk_is_refined_medoid")) and bool(row.get("answer_chunk_metric_path_rendered")) for row in query_rows):
            counts["medoid"] += 1
        if any(bool(row.get("answer_chunk_on_anchor_medoid_path")) and bool(row.get("answer_chunk_metric_path_rendered")) for row in query_rows):
            counts["anchor_medoid_path"] += 1
        if any(bool(row.get("answer_chunk_on_medoid_medoid_path")) and bool(row.get("answer_chunk_metric_path_rendered")) for row in query_rows):
            counts["medoid_medoid_path"] += 1
        current = any(bool(row.get("answer_chunk_current_rendered")) for row in query_rows)
        metric = any(bool(row.get("answer_chunk_metric_path_rendered")) for row in query_rows)
        if current and not metric:
            counts["current_only_hidden_recovery"] += 1
        if any(bool(row.get("answer_chunk_dropped_by_budget")) for row in query_rows):
            counts["budget_cutoff"] += 1
    return counts


def _decision(rows: list[dict[str, Any]]) -> tuple[str, str]:
    ref = next((row for row in rows if row["run"] == REFERENCE_RUN), None)
    metric = next((row for row in rows if row["run"] == METRIC_RUN), None)
    if ref is None or metric is None:
        return "STOP_BEFORE_100", "current and metric_path_carrier runs are both required"
    if ref["triangle_inequality_violation_count"] or metric["triangle_inequality_violation_count"]:
        return "STOP_BEFORE_100", "triangle invariant failed"
    if ref["oracle_leakage_count"] or metric["oracle_leakage_count"]:
        return "STOP_BEFORE_100", "oracle leakage was detected"
    current = float(ref["answer_in_context"])
    tree = float(ref["answer_on_refined_support_tree_rate"])
    metric_answer = float(metric["answer_in_context"])
    if current <= 0.0:
        return "STOP_BEFORE_100", "current renderer has no answer-in-context signal to explain"
    if tree + 1e-12 < current * 0.70:
        return "STOP_BEFORE_100", "refined support tree does not explain the current answer recovery"
    if metric_answer + 1e-12 >= current * 0.85 and tree + 1e-12 >= current * 0.85:
        return "GO_TO_100", "metric path carrier reproduces most current answer-in-context and the support tree explains the gap"
    if tree + 1e-12 >= current * 0.85:
        return "DIAGNOSTIC_ONLY_100", "support tree explains the gap, but deterministic rendering order/budget loses answers"
    return "STOP_BEFORE_100", "metric path carrier is far below current and support-tree explanation is incomplete"


def compare_dataset(root: Path) -> dict[str, Any]:
    rows = [_row(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    decision, reason = _decision(rows)
    taxonomy = _combined_taxonomy(root)
    return {
        "dataset": root.name,
        "root": str(root),
        "rows": rows,
        "reference": next((row for row in rows if row["run"] == REFERENCE_RUN), {}),
        "metric": next((row for row in rows if row["run"] == METRIC_RUN), {}),
        "oracle_rows": [row for row in rows if row["oracle_renderer"]],
        "answer_role_distribution": _cross_role_distribution(root),
        "taxonomy": taxonomy.get("path_carrier_failure_counts", {}),
        "decision": decision,
        "reason": reason,
    }


def compare(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    if datasets:
        results = [compare_dataset(root / dataset) for dataset in datasets]
    else:
        results = [compare_dataset(root)]
    decisions = {result["decision"] for result in results}
    final = "STOP_BEFORE_100"
    if decisions == {"GO_TO_100"}:
        final = "GO_TO_100"
    elif decisions <= {"GO_TO_100", "DIAGNOSTIC_ONLY_100"} and "DIAGNOSTIC_ONLY_100" in decisions:
        final = "DIAGNOSTIC_ONLY_100"
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": results,
        "decision": final,
    }


def _dataset_markdown(result: dict[str, Any]) -> list[str]:
    ref = result.get("reference", {})
    metric = result.get("metric", {})
    lines = [
        f"## {result.get('dataset', '')}",
        "",
        f"- Decision: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Reason: {result.get('reason', '')}",
        f"- current answer-in-context: {_fmt(float(ref.get('answer_in_context', 0.0)))}",
        f"- selected medoid answer availability: {_fmt(float(ref.get('selected_medoid_answer_availability', 0.0)))}",
        f"- current-minus-medoid answer gap: {_fmt(float(ref.get('current_minus_medoid_answer_gap', 0.0)))}",
        f"- answer on refined support tree: {_fmt(float(ref.get('answer_on_refined_support_tree_rate', 0.0)))}",
        f"- answer rendered by metric path carrier: {_fmt(float(metric.get('answer_in_context', 0.0)))}",
        f"- current-minus-metric-path answer gap: {_fmt(float(ref.get('answer_in_context', 0.0)) - float(metric.get('answer_in_context', 0.0)))}",
        "",
        "### Answer Role Distribution",
        "",
        _fmt(result.get("answer_role_distribution", {})),
        "",
        "### Path Carrier Failure Taxonomy",
        "",
        _fmt(result.get("taxonomy", {})),
        "",
        "### Variant Table",
        "",
        "| run | renderer_mode | oracle | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | triangle | oracle_leakage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.get("rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["renderer_mode"]),
                    str(row["oracle_renderer"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["triangle_inequality_violation_count"]),
                    _fmt(row["oracle_leakage_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Oracle Comparison",
            "",
            "| oracle | qa_f1 | answer_in_context | avg_context_tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in result.get("oracle_rows", []):
        lines.append(
            f"| {row['renderer_mode']} | {_fmt(row['qa_f1'])} | {_fmt(row['answer_in_context'])} | {_fmt(row['avg_context_tokens'])} |"
        )
    lines.append("")
    return lines


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Path Carrier Completion Diagnostic",
        "",
        f"- Branch: `{summary.get('branch', '')}`",
        f"- Commit: `{summary.get('commit', '')}`",
        f"- Final decision: **{summary.get('decision', 'STOP_BEFORE_100')}**",
        "",
        "PAMAE principle check: Phase I/II retrieval is unchanged. The non-oracle renderer uses `SPClosure(A_q + Theta_refined)`, graph shortest-path distance, deterministic ordering, and the existing context budget. It does not use scalar score mixing, dense/BM25/LLM reranking, answer-aware retrieval, or gold-aware retrieval.",
        "",
    ]
    for result in summary.get("datasets", []):
        lines.extend(_dataset_markdown(result))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metric path carrier diagnostic runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = compare(Path(args.root), datasets=args.datasets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
