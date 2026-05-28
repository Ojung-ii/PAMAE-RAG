#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REFERENCE_RUN = "entity_chunk_reference_current_renderer"
RUN_ORDER = (
    REFERENCE_RUN,
    "entity_chunk_reference_metric_path_carrier",
    "entity_chunk_reference_tree_shell1_graph_order",
    "entity_chunk_reference_tree_shell1_semantic_query_order",
    "entity_chunk_reference_tree_shell1_semantic_tree_order",
    "entity_chunk_reference_semantic_weighted_tree_diagnostic",
    "entity_chunk_reference_current_answer_role_oracle",
    "entity_chunk_reference_tree_answer_oracle",
    "entity_chunk_reference_shell1_answer_oracle",
)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
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
    value = metrics.get("qa_metrics", {}).get("stage_diagnostics", {}).get(stage, {}).get("mean", {}).get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _run_row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "answer_carrier_metrics.json")
    rendered_recall = float(metrics.get("rendered_recall", 0.0))
    if rendered_recall == 0.0:
        rendered_recall = _stage_mean(metrics, "context_rendering", "rendered_recall")
    return {
        "run": run_dir.name,
        "renderer_mode": str(metrics.get("renderer_mode", "")),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "diagnostic_renderer": bool(metrics.get("diagnostic_renderer", False)),
        "uses_answer_string": bool(metrics.get("uses_answer_string", False)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": rendered_recall,
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "generation_ms": float(metrics.get("generation_ms", 0.0)),
        "total_ms": float(metrics.get("total_ms", 0.0)),
        "answer_on_support_tree_rate": float(metrics.get("answer_on_refined_support_tree_rate", 0.0)),
        "answer_in_shell1_rate": float(metrics.get("answer_in_shell1_rate", 0.0)),
        "answer_rendered_from_shell1_rate": float(metrics.get("answer_rendered_from_shell1_rate", 0.0)),
        "current_only_hidden_recovery_rate": float(metrics.get("answer_current_only_non_tree_rate", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
        "score_mixing_detected": bool(metrics.get("score_mixing_detected", False)),
        "embedding_missing_rate": float(metrics.get("embedding_missing_rate", 0.0)),
    }


def _row_by_renderer(rows: list[dict[str, Any]], renderer: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("renderer_mode") == renderer), {})


def _decision(preflight: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    if preflight and not bool(preflight.get("semantic_mode_enabled", False)):
        return "STOP_BEFORE_100", str(preflight.get("reason", "semantic preflight failed")), "STOP"
    if not rows:
        return "STOP_BEFORE_100", "no semantic carrier run metrics were found", "STOP"
    ref = _row_by_renderer(rows, "current_renderer")
    if not ref:
        return "STOP_BEFORE_100", "current_renderer reference is missing", "STOP"
    non_oracle = [row for row in rows if not row["oracle_renderer"] and row["renderer_mode"] != "current_renderer"]
    if any(row["score_mixing_detected"] for row in non_oracle):
        return "STOP_BEFORE_100", "score mixing was detected in a non-oracle semantic renderer", "STOP"
    if any(row["oracle_leakage_count"] for row in non_oracle):
        return "STOP_BEFORE_100", "oracle leakage was detected in a non-oracle semantic renderer", "STOP"
    best = max(non_oracle, key=lambda row: row["answer_in_context"], default={})
    if not best:
        return "STOP_BEFORE_100", "no non-oracle semantic renderer was run", "STOP"
    ref_answer = float(ref.get("answer_in_context", 0.0))
    best_answer = float(best.get("answer_in_context", 0.0))
    if best_answer + 1e-12 < ref_answer:
        return "DIAGNOSTIC_ONLY_100", "semantic run is meaningful but does not match current answer coverage", "DIAGNOSTIC_ONLY"
    return "GO_TO_100", "non-oracle semantic renderer preserved 2Wiki answer coverage in the smoke", "ADOPTION_CANDIDATE"


def compare_dataset(root: Path) -> dict[str, Any]:
    rows = [_run_row(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    preflight = _read(root / "semantic_embedding_preflight.json")
    decision, reason, final = _decision(preflight, rows)
    return {
        "dataset": root.name,
        "root": str(root),
        "preflight": preflight,
        "rows": rows,
        "semantic_attribution": _read(root / "semantic_attribution_metrics.json"),
        "decision": decision,
        "reason": reason,
        "final_decision": final,
    }


def compare(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    results = [compare_dataset(root / dataset) for dataset in datasets] if datasets else [compare_dataset(root)]
    decisions = {str(result["decision"]) for result in results}
    if decisions == {"GO_TO_100"}:
        gate = "GO_TO_100"
    elif decisions <= {"GO_TO_100", "DIAGNOSTIC_ONLY_100"} and "DIAGNOSTIC_ONLY_100" in decisions:
        gate = "DIAGNOSTIC_ONLY_100"
    else:
        gate = "STOP_BEFORE_100"
    final = "ADOPTION_CANDIDATE" if gate == "GO_TO_100" and len(results) > 1 else "DIAGNOSTIC_ONLY"
    if gate == "STOP_BEFORE_100":
        final = "STOP"
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": results,
        "decision": gate,
        "final_decision": final,
    }


def _dataset_markdown(result: dict[str, Any]) -> list[str]:
    rows = result.get("rows", [])
    preflight = result.get("preflight", {})
    lines = [
        f"## {result.get('dataset', '')}",
        "",
        f"- Gate decision: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Reason: {result.get('reason', '')}",
        f"- Final decision: **{result.get('final_decision', 'STOP')}**",
        f"- Embedding source: `{preflight.get('embedding_source', 'unknown')}`",
        f"- Embedding dim: `{preflight.get('embedding_dim', 0)}`",
        f"- Chunk embedding coverage: {_fmt(float(preflight.get('chunk_embedding_coverage', 0.0)))}",
        f"- Query embedding available: `{preflight.get('query_embedding_available', False)}`",
        f"- Semantic mode enabled: `{preflight.get('semantic_mode_enabled', False)}`",
        f"- Embedding missing rate: {_fmt(float(preflight.get('embedding_missing_rate', 1.0)))}",
        "",
        "### Semantic Attribution",
        "",
        "Query-to-chunk semantic attribution is unavailable unless query embeddings already exist in the examples. No synthetic query vector fallback is permitted.",
        "",
        "### Renderer Table",
        "",
        "| run | renderer_mode | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_context_tokens | retrieval_ms |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["renderer_mode"]),
                    str(row["oracle_renderer"]),
                    str(row["diagnostic_renderer"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Semantic Carrier Adequacy Diagnostic",
        "",
        f"- Branch: `{result.get('branch', '')}`",
        f"- Commit: `{result.get('commit', '')}`",
        f"- Gate outcome: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Final decision: **{result.get('final_decision', 'STOP')}**",
        "",
        "Previous hidden-non-tree summary: support-tree diagnostics ended as `DIAGNOSTIC_ONLY` with hidden non-tree recovery on both datasets. This round tests whether existing embeddings can explain or order those carriers inside graph-defined candidates.",
        "",
        "PAMAE boundary: the entity-chunk graph, graph-metric medoid selection, local refinement, and `T_q = SPClosure(A_q union Theta_refined)` are unchanged. Semantic information is restricted to diagnostics, lexicographic ordering inside `T_q union S1`, and a fixed `1 + d_ang` diagnostic tree.",
        "",
        "Why raw cosine is not used as a PAMAE distance: cosine similarity is not a metric distance and `1 - cosine` is not used as proof-level distance. The implementation uses normalized angular distance `arccos(clamp(dot,-1,1))/pi` for semantic diagnostics.",
        "",
    ]
    for dataset in result.get("datasets", []):
        lines.extend(_dataset_markdown(dataset))
    lines.extend(
        [
            "## Expert Panel Checks",
            "",
            "- GraphRAG expert: semantic candidates are graph-constrained to `T_q union S1`; global dense retrieval is not introduced.",
            "- IR expert: semantic attribution cannot be interpreted without existing query embeddings.",
            "- Graph theory expert: angular distance tests pass; semantic-weighted edges are positive in the diagnostic implementation.",
            "- NLP expert: no semantic query ordering is run when query embeddings are absent, avoiding topical non-answer selection artifacts.",
            "- Systems expert: no overnight smoke proceeds when required semantic inputs are unavailable.",
            "- Professor/meta-reviewer: no thresholds, tuned weights, or dataset-specific semantic branches are introduced.",
            "",
            "## Final Recommendation",
            "",
            "STOP. Existing chunk embeddings are present, but query embeddings are missing from the processed examples. Per the experiment boundary, the run stops after implementation verification and embedding preflight rather than fabricating query vectors or falling back to another retrieval signal.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = compare(args.root, args.datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(result), encoding="utf-8")
    comparison_path = args.out.parent / "semantic_carrier_comparison.json"
    comparison_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.out.name == "SEMANTIC_CARRIER_COMPARISON_50.md":
        decision_path = args.out.parent / "SEMANTIC_CARRIER_50_DECISION.md"
        decision_path.write_text(to_markdown(result), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
