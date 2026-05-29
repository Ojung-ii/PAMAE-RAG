#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

RUN_ORDER = (
    "entity_chunk_reference_current_renderer",
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


def _run_row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "answer_carrier_metrics.json")
    return {
        "run": run_dir.name,
        "renderer_mode": str(metrics.get("renderer_mode", "")),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "diagnostic_renderer": bool(metrics.get("diagnostic_renderer", False)),
        "uses_answer_string": bool(metrics.get("uses_answer_string", False)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": float(metrics.get("rendered_recall", 0.0)),
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "generation_ms": float(metrics.get("generation_ms", 0.0)),
        "total_ms": float(metrics.get("total_ms", 0.0)),
        "support_tree_chunk_count": float(metrics.get("support_tree_chunk_count", 0.0)),
        "shell1_chunk_count": float(metrics.get("shell1_chunk_count", 0.0)),
        "shell2_chunk_count": float(metrics.get("shell2_chunk_count", 0.0)),
        "rendered_shell1_chunk_count": float(metrics.get("rendered_shell1_chunk_count", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
        "score_mixing_detected": bool(metrics.get("score_mixing_detected", False)),
        "embedding_missing_rate": float(metrics.get("embedding_missing_rate", 0.0)),
    }


def _row_by_renderer(rows: list[dict[str, Any]], renderer: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("renderer_mode") == renderer), {})


def _decision(rows: list[dict[str, Any]], semantic: dict[str, Any], cache: dict[str, Any]) -> tuple[str, str]:
    if not rows:
        return "STOP_BEFORE_100", "no run metrics were found"
    if float(cache.get("query_embedding_coverage", 0.0)) < 1.0:
        return "STOP_BEFORE_100", "query embedding coverage is below 1.0"
    if float(cache.get("chunk_embedding_coverage_for_diagnostics", 0.0)) < 1.0:
        return "STOP_BEFORE_100", "diagnostic chunk embedding coverage is below 1.0"
    if not bool(cache.get("embedding_dim_match", False)):
        return "STOP_BEFORE_100", "embedding dimensions mismatch"
    if not bool(cache.get("all_vectors_l2_normalized", False)):
        return "STOP_BEFORE_100", "vectors are not L2-normalized"
    non_oracle = [row for row in rows if not row["oracle_renderer"]]
    if any(row["score_mixing_detected"] for row in non_oracle):
        return "STOP_BEFORE_100", "score mixing detected in a non-oracle semantic path"
    if any(row["oracle_leakage_count"] for row in non_oracle):
        return "STOP_BEFORE_100", "oracle leakage detected in a non-oracle semantic path"
    attr = semantic.get("semantic_hidden_carrier", {})
    separation_query = attr.get("semantic_separation_query")
    if not isinstance(separation_query, (int, float)) or float(separation_query) <= 0.0:
        return "STOP_BEFORE_100", "semantic attribution does not separate current-only answer chunks from non-answer chunks"
    ref = _row_by_renderer(rows, "current_renderer")
    best_semantic = max(
        (
            row
            for row in rows
            if row["renderer_mode"]
            in {
                "tree_shell1_graph_order",
                "tree_shell1_semantic_query_order",
                "tree_shell1_semantic_tree_order",
            }
        ),
        key=lambda row: row["answer_in_context"],
        default={},
    )
    if not ref or not best_semantic:
        return "STOP_BEFORE_100", "missing current or semantic renderer metrics"
    if best_semantic["answer_in_context"] + 1e-12 < ref["answer_in_context"]:
        return "DIAGNOSTIC_ONLY_100", "semantic attribution is meaningful but renderers do not preserve current answer coverage"
    if best_semantic["rendered_recall"] + 1e-12 < ref["rendered_recall"]:
        return "DIAGNOSTIC_ONLY_100", "semantic renderer preserves answer coverage but regresses rendered recall"
    return "GO_TO_100", "semantic attribution is meaningful and 2Wiki smoke preserves coverage metrics"


def compare_dataset(root: Path) -> dict[str, Any]:
    rows = [_run_row(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    semantic = _read(root / "semantic_attribution_metrics.json")
    cache = _read(root / "compatible_embedding_cache_summary.json")
    decision, reason = _decision(rows, semantic, cache)
    return {
        "dataset": root.name,
        "root": str(root),
        "rows": rows,
        "semantic": semantic,
        "cache": cache,
        "decision": decision,
        "reason": reason,
    }


def compare(root: Path, datasets: list[str] | None) -> dict[str, Any]:
    dataset_results = [compare_dataset(root / dataset) for dataset in datasets] if datasets else [compare_dataset(root)]
    decisions = {str(result["decision"]) for result in dataset_results}
    if decisions == {"GO_TO_100"}:
        gate = "GO_TO_100"
    elif decisions <= {"GO_TO_100", "DIAGNOSTIC_ONLY_100"} and "DIAGNOSTIC_ONLY_100" in decisions:
        gate = "DIAGNOSTIC_ONLY_100"
    else:
        gate = "STOP_BEFORE_100"
    if len(dataset_results) > 1 and gate == "GO_TO_100":
        final = "ADOPTION_CANDIDATE"
    elif gate == "STOP_BEFORE_100":
        final = "STOP"
    else:
        final = "DIAGNOSTIC_ONLY"
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": dataset_results,
        "decision": gate,
        "final_decision": final,
    }


def _dataset_markdown(result: dict[str, Any]) -> list[str]:
    rows = result.get("rows", [])
    semantic = result.get("semantic", {})
    cache = result.get("cache", {})
    attr = semantic.get("semantic_hidden_carrier", {})
    pools = semantic.get("pool_sizes", {})
    lines = [
        f"## {result.get('dataset', '')}",
        "",
        f"- Gate decision: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Reason: {result.get('reason', '')}",
        f"- Model: `{cache.get('model_id')}`",
        f"- Revision: `{cache.get('model_revision')}`",
        f"- Dim: `{cache.get('embedding_dim')}`",
        f"- Query coverage: {_fmt(cache.get('query_embedding_coverage'))}",
        f"- Chunk coverage: {_fmt(cache.get('chunk_embedding_coverage_for_diagnostics'))}",
        f"- Normalized: `{cache.get('normalized')}`",
        f"- Text format: chunk `{cache.get('chunk_text_format')}`, query `{cache.get('query_text_format')}`",
        "",
        "### Semantic Attribution",
        "",
        f"- mean d_ang(q,u), current-only answer: {_fmt(attr.get('mean_d_ang_query_current_only_answer'))}",
        f"- mean d_ang(q,u), current-only non-answer: {_fmt(attr.get('mean_d_ang_query_current_only_non_answer'))}",
        f"- median d_ang(q,u), current-only answer: {_fmt(attr.get('median_d_ang_query_current_only_answer'))}",
        f"- median d_ang(q,u), current-only non-answer: {_fmt(attr.get('median_d_ang_query_current_only_non_answer'))}",
        f"- mean d_ang(u,T_q), current-only answer: {_fmt(attr.get('mean_d_ang_tree_current_only_answer'))}",
        f"- mean d_ang(u,T_q), current-only non-answer: {_fmt(attr.get('mean_d_ang_tree_current_only_non_answer'))}",
        f"- semantic_separation_query: {_fmt(attr.get('semantic_separation_query'))}",
        f"- semantic_separation_tree: {_fmt(attr.get('semantic_separation_tree'))}",
        "",
        "### Pool Sizes",
        "",
        f"- avg strict tree chunks: {_fmt(pools.get('avg_support_tree_chunks'))}",
        f"- avg shell1 chunks: {_fmt(pools.get('avg_shell1_chunks'))}",
        f"- avg shell2 chunks: {_fmt(pools.get('avg_shell2_chunks'))}",
        f"- answer on support tree rate: {_fmt(pools.get('answer_on_support_tree_rate'))}",
        f"- answer in shell1 rate: {_fmt(pools.get('answer_in_shell1_rate'))}",
        f"- answer in shell2 rate: {_fmt(pools.get('answer_in_shell2_rate'))}",
        "",
        "### Variant Table",
        "",
        "| renderer | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | shell1 | rendered_shell1 | missing_rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["renderer_mode"]),
                    str(row["oracle_renderer"]),
                    str(row["diagnostic_renderer"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["shell1_chunk_count"]),
                    _fmt(row["rendered_shell1_chunk_count"]),
                    _fmt(row["embedding_missing_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Semantic Embedding-Space Rerun Report",
        "",
        f"- Branch: `{result.get('branch', '')}`",
        f"- Commit: `{result.get('commit', '')}`",
        f"- Gate outcome: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Final decision: **{result.get('final_decision', 'STOP')}**",
        "",
        "Previous semantic STOP summary: the first semantic carrier run stopped because query embeddings were absent. This rerun audits provenance, rejects legacy 128D chunks, and uses one compatible local encoder when available.",
        "",
        "Theory: raw cosine is not treated as a PAMAE distance. Semantic diagnostics use normalized angular distance, and semantic renderers are graph-constrained to `T_q union S1`; the entity-chunk graph retrieval core remains unchanged.",
        "",
    ]
    for dataset in result.get("datasets", []):
        lines.extend(_dataset_markdown(dataset))
    lines.extend(
        [
            "## Expert Panel Rules",
            "",
            "- GraphRAG expert: reject semantic gains outside graph-defined candidates.",
            "- IR expert: semantic adequacy is justified only if answer/non-answer separation appears inside the graph shell.",
            "- Graph theory expert: reject if angular metric or positive edge constraints fail.",
            "- NLP expert: stop if query similarity selects topical non-answer chunks.",
            "- RAG expert: if coverage improves without QA, inspect formatting before retrieval changes.",
            "- Systems expert: do not adopt if retrieval time increases too much.",
            "- Professor/meta-reviewer: one-dataset or weight/threshold-dependent gains are local-minimum risk.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic embedding rerun variants.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = compare(args.root, args.datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(result), encoding="utf-8")
    comparison_path = args.out.parent / "semantic_embedding_rerun_comparison.json"
    comparison_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.out.name == "SEMANTIC_EMBEDDING_RERUN_50.md":
        (args.out.parent / "SEMANTIC_EMBEDDING_RERUN_50_DECISION.md").write_text(
            to_markdown(result),
            encoding="utf-8",
        )
    print(args.out)


if __name__ == "__main__":
    main()
