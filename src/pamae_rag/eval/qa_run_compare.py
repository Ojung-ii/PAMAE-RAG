from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QARunSpec:
    name: str
    graph_mode: str
    qa_metrics_path: Path
    retrieval_metrics_path: Path | None = None
    risk_decision: str = "measurement"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_mean(qa_metrics: dict[str, Any], stage: str, key: str) -> float | None:
    stage_payload = qa_metrics.get("stage_diagnostics", {}).get(stage, {})
    value = stage_payload.get("mean", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _metric(qa_metrics: dict[str, Any], retrieval_metrics: dict[str, Any] | None, key: str) -> float | None:
    if key in qa_metrics and isinstance(qa_metrics[key], (int, float)):
        return float(qa_metrics[key])
    if retrieval_metrics and key in retrieval_metrics and isinstance(retrieval_metrics[key], (int, float)):
        return float(retrieval_metrics[key])
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _row(spec: QARunSpec, oracle_f1: float | None) -> dict[str, Any]:
    qa_metrics = _read_json(spec.qa_metrics_path)
    retrieval_metrics = _read_json(spec.retrieval_metrics_path) if spec.retrieval_metrics_path else None
    f1 = _metric(qa_metrics, retrieval_metrics, "mean_f1")
    oracle = bool(qa_metrics.get("oracle", False))
    oracle_gap = None if f1 is None or oracle_f1 is None else oracle_f1 - f1
    return {
        "run": spec.name,
        "graph_mode": spec.graph_mode,
        "oracle": oracle,
        "candidate_recall": _stage_mean(qa_metrics, "query_anchor_construction", "candidate_recall"),
        "projected_recall": _stage_mean(
            qa_metrics,
            "content_graph_projection",
            "gold_supporting_evidence_survival",
        )
        if _stage_mean(qa_metrics, "content_graph_projection", "projected_node_count") is not None
        else None,
        "post_refine_recall": _stage_mean(qa_metrics, "local_refinement", "gold_supporting_evidence_survival"),
        "rendered_recall": _stage_mean(qa_metrics, "context_rendering", "rendered_recall")
        or _metric(qa_metrics, retrieval_metrics, "mean_context_recall"),
        "context_f1": _metric(qa_metrics, retrieval_metrics, "mean_context_f1"),
        "avg_context_tokens": _metric(qa_metrics, retrieval_metrics, "avg_context_tokens"),
        "retrieval_ms": _metric(qa_metrics, retrieval_metrics, "avg_retrieval_ms"),
        "generation_ms": _metric(qa_metrics, retrieval_metrics, "avg_generation_ms"),
        "EM": _metric(qa_metrics, retrieval_metrics, "mean_exact_match"),
        "F1": f1,
        "oracle_gap": oracle_gap,
        "risk_decision": spec.risk_decision,
        "answer_coverage": _metric(qa_metrics, retrieval_metrics, "mean_answer_coverage"),
        "selected_answer_coverage": _metric(qa_metrics, retrieval_metrics, "mean_selected_answer_coverage"),
        "generator_id": qa_metrics.get("generator_id"),
        "prompt_id": qa_metrics.get("prompt_id"),
        "metric_id": qa_metrics.get("metric_id"),
        "context_recall": _metric(qa_metrics, retrieval_metrics, "mean_context_recall"),
    }


def compare_runs(specs: list[QARunSpec]) -> dict[str, Any]:
    oracle_rows = []
    for spec in specs:
        metrics = _read_json(spec.qa_metrics_path)
        if bool(metrics.get("oracle", False)):
            oracle_rows.append((spec, metrics))
    if len(oracle_rows) != 1:
        raise ValueError(f"Expected exactly one oracle run, found {len(oracle_rows)}")
    oracle_spec, oracle_metrics = oracle_rows[0]
    oracle_f1 = float(oracle_metrics.get("mean_f1", 0.0))
    oracle_context_recall = float(oracle_metrics.get("mean_context_recall", 0.0))
    oracle_answer_coverage = _metric(oracle_metrics, None, "mean_answer_coverage")
    oracle_selected_answer_coverage = _metric(oracle_metrics, None, "mean_selected_answer_coverage")
    rows = [_row(spec, oracle_f1) for spec in specs]
    generator_ids = {row.get("generator_id") for row in rows}
    prompt_ids = {row.get("prompt_id") for row in rows}
    metric_ids = {row.get("metric_id") for row in rows}
    dominance_violations = [
        row["run"]
        for row in rows
        if not row["oracle"] and row["F1"] is not None and row["F1"] > oracle_f1 + 1e-12
    ]
    qa_settings_consistent = (
        len(generator_ids) == 1
        and len(prompt_ids) == 1
        and len(metric_ids) == 1
    )
    oracle_context_complete = oracle_context_recall >= 1.0 - 1e-12
    return {
        "oracle_run": oracle_spec.name,
        "oracle_f1": oracle_f1,
        "oracle_context_recall": oracle_context_recall,
        "oracle_answer_coverage": oracle_answer_coverage,
        "oracle_selected_answer_coverage": oracle_selected_answer_coverage,
        "oracle_context_complete": oracle_context_complete,
        "qa_settings_consistent": qa_settings_consistent,
        "oracle_dominance_valid": not dominance_violations,
        "dominance_violations": dominance_violations,
        "rows": rows,
    }


def write_summary(summary: dict[str, Any], output_json: Path, output_csv: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary["rows"]
    fieldnames = [
        "run",
        "graph_mode",
        "oracle",
        "candidate_recall",
        "projected_recall",
        "post_refine_recall",
        "rendered_recall",
        "context_f1",
        "avg_context_tokens",
        "retrieval_ms",
        "generation_ms",
        "EM",
        "F1",
        "oracle_gap",
        "risk_decision",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        "| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["graph_mode"]),
                    str(row["oracle"]).lower(),
                    _fmt(row["candidate_recall"]),
                    _fmt(row["projected_recall"]),
                    _fmt(row["post_refine_recall"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["generation_ms"]),
                    _fmt(row["EM"]),
                    _fmt(row["F1"]),
                    _fmt(row["oracle_gap"]),
                    str(row["risk_decision"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- oracle_context_complete: `{str(summary['oracle_context_complete']).lower()}`",
            f"- oracle_answer_coverage: `{_fmt(summary.get('oracle_answer_coverage'))}`",
            f"- oracle_selected_answer_coverage: `{_fmt(summary.get('oracle_selected_answer_coverage'))}`",
            f"- qa_settings_consistent: `{str(summary['qa_settings_consistent']).lower()}`",
            f"- oracle_dominance_valid: `{str(summary['oracle_dominance_valid']).lower()}`",
            f"- dominance_violations: `{', '.join(summary['dominance_violations']) or 'none'}`",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
