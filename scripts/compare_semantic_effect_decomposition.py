#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.diagnostics.semantic_effect_decomposition import (
    VARIANT_RENDERERS,
    decomposition_deltas,
    load_json,
    prompt_protocol_status,
    same_query_sample,
)

RUN_ORDER = tuple(f"entity_chunk_reference_{renderer}" for renderer in VARIANT_RENDERERS)
ADOPTABLE_RENDERERS = {
    "tree_shell1_graph_order",
    "tree_shell1_semantic_query_order",
    "tree_shell1_semantic_tree_order",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _read_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "answer_carrier_metrics.json")
    return {
        "run": run_dir.name,
        "renderer_mode": str(metrics.get("renderer_mode", "")),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "diagnostic_renderer": bool(metrics.get("diagnostic_renderer", False)),
        "qa_prompt_name": metrics.get("qa_prompt_name"),
        "qa_prompt_hash": metrics.get("qa_prompt_hash"),
        "qa_prompt_text_exact_match": bool(metrics.get("qa_prompt_text_exact_match", False)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": float(metrics.get("rendered_recall", 0.0)),
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "generation_ms": float(metrics.get("generation_ms", 0.0)),
        "total_ms": float(metrics.get("total_ms", 0.0)),
        "answer_on_support_tree_rate": float(metrics.get("answer_on_support_tree_rate", 0.0)),
        "answer_in_shell1_rate": float(metrics.get("answer_in_shell1_rate", 0.0)),
        "answer_rendered_from_shell1_rate": float(metrics.get("answer_rendered_from_shell1_rate", 0.0)),
        "current_only_hidden_recovery_rate": float(metrics.get("current_only_hidden_recovery_rate", 0.0)),
        "bridge_or_path_answer_retained_rate": float(metrics.get("bridge_or_path_answer_retained_rate", 0.0)),
        "bridge_or_path_answer_pushed_after_budget_rate": float(
            metrics.get("bridge_or_path_answer_pushed_after_budget_rate", 0.0)
        ),
        "bridge_answer_chunk_count": int(metrics.get("bridge_answer_chunk_count", 0)),
        "bridge_answer_rendered_count": int(metrics.get("bridge_answer_rendered_count", 0)),
        "bridge_answer_cut_by_budget_count": int(metrics.get("bridge_answer_cut_by_budget_count", 0)),
        "bridge_answer_mean_render_rank": float(metrics.get("bridge_answer_mean_render_rank", 0.0)),
        "semantic_order_pushed_bridge_after_budget_rate": float(
            metrics.get("semantic_order_pushed_bridge_after_budget_rate", 0.0)
        ),
        "support_tree_chunk_count": float(metrics.get("support_tree_chunk_count", 0.0)),
        "shell1_chunk_count": float(metrics.get("shell1_chunk_count", 0.0)),
        "shell2_chunk_count": float(metrics.get("shell2_chunk_count", 0.0)),
        "rendered_shell1_chunk_count": float(metrics.get("rendered_shell1_chunk_count", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
        "score_mixing_detected": bool(metrics.get("score_mixing_detected", False)),
        "embedding_missing_rate": float(metrics.get("embedding_missing_rate", 0.0)),
    }


def _row_by_renderer(rows: list[dict[str, Any]], renderer_mode: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("renderer_mode") == renderer_mode), {})


def _cache_ok(cache: dict[str, Any]) -> bool:
    return (
        str(cache.get("model_id")) == "nvidia/NV-Embed-v2"
        and int(cache.get("embedding_dim", 0) or 0) == 4096
        and float(cache.get("query_embedding_coverage", 0.0)) >= 1.0
        and float(cache.get("chunk_embedding_coverage_for_diagnostics", 0.0)) >= 1.0
        and bool(cache.get("all_vectors_l2_normalized", False))
    )


def _adoption_gate_for_dataset(rows: list[dict[str, Any]], renderer_mode: str) -> tuple[bool, list[str]]:
    ref = _row_by_renderer(rows, "current_renderer")
    row = _row_by_renderer(rows, renderer_mode)
    blockers: list[str] = []
    if not ref or not row:
        return False, ["missing reference or candidate row"]
    if row["qa_f1"] + 1e-12 < ref["qa_f1"]:
        blockers.append("qa_f1_regression")
    if row["answer_in_context"] + 1e-12 < ref["answer_in_context"]:
        blockers.append("answer_in_context_regression")
    if row["rendered_recall"] + 1e-12 < ref["rendered_recall"]:
        blockers.append("rendered_recall_regression")
    if row["context_f1"] + 1e-12 < ref["context_f1"]:
        blockers.append("context_f1_regression")
    if row["avg_context_tokens"] > ref["avg_context_tokens"] * 1.10 + 1e-12:
        blockers.append("token_gate")
    if row["retrieval_ms"] > ref["retrieval_ms"] * 1.25 + 1e-12:
        blockers.append("retrieval_time_gate")
    if row["oracle_leakage_count"] != 0:
        blockers.append("oracle_leakage")
    if row["score_mixing_detected"]:
        blockers.append("score_mixing")
    if row["qa_prompt_name"] != "common_qa":
        blockers.append("prompt_not_common_qa")
    return not blockers, blockers


def _dominant_effect(deltas: dict[str, dict[str, float]]) -> str:
    shell = deltas["delta_shell_B1_minus_A1"]["answer_in_context"]
    query = deltas["delta_query_semantic_B2_minus_B1"]["answer_in_context"]
    tree = deltas["delta_tree_semantic_B3_minus_B1"]["answer_in_context"]
    best = max(
        (("shell_expansion", shell), ("query_semantic_ordering", query), ("tree_semantic_ordering", tree)),
        key=lambda item: item[1],
    )
    if best[1] <= 0:
        return "no_positive_answer_coverage_effect"
    return best[0]


def _dataset_decision(dataset: dict[str, Any]) -> tuple[str, str]:
    rows = dataset["rows"]
    cache = dataset["cache"]
    if not rows:
        return "STOP", "no run metrics found"
    if not _cache_ok(cache):
        return "STOP", "NV-Embed-v2 cache compatibility failed"
    if not dataset["same_sample"].get("same_sample", False):
        return "STOP", "variant query samples differ"
    if not dataset["prompt_protocol"].get("qa_prompt_consistent", False):
        return "STOP", "common_qa prompt protocol is inconsistent"
    non_oracle = [row for row in rows if not row["oracle_renderer"]]
    if any(row["score_mixing_detected"] for row in non_oracle):
        return "STOP", "score mixing detected"
    if any(row["oracle_leakage_count"] for row in non_oracle):
        return "STOP", "oracle leakage detected"
    if any(row["triangle_inequality_violation_count"] for row in non_oracle):
        return "STOP", "triangle inequality invariant failed"
    return "DIAGNOSTIC_ONLY", "diagnostics valid; adoption requires both datasets to pass gates"


def compare_dataset(root: Path) -> dict[str, Any]:
    rows = [_read_run(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    semantic = load_json(root / "semantic_effect_decomposition_metrics.json") or load_json(
        root / "semantic_attribution_metrics.json"
    )
    cache = load_json(root / "compatible_embedding_cache_summary.json")
    sample = same_query_sample([root / run / "qa.jsonl" for run in RUN_ORDER])
    prompt = prompt_protocol_status(rows)
    deltas = decomposition_deltas(rows)
    adoption = {
        renderer: {
            "pass": passed,
            "blockers": blockers,
        }
        for renderer in sorted(ADOPTABLE_RENDERERS)
        for passed, blockers in [_adoption_gate_for_dataset(rows, renderer)]
    }
    decision, reason = _dataset_decision(
        {
            "rows": rows,
            "cache": cache,
            "same_sample": sample,
            "prompt_protocol": prompt,
        }
    )
    return {
        "dataset": root.name,
        "root": str(root),
        "rows": rows,
        "semantic": semantic,
        "cache": cache,
        "same_sample": sample,
        "prompt_protocol": prompt,
        "deltas": deltas,
        "dominant_effect": _dominant_effect(deltas),
        "adoption_gates": adoption,
        "decision": decision,
        "reason": reason,
    }


def compare(root: Path, datasets: list[str]) -> dict[str, Any]:
    dataset_results = [compare_dataset(root / dataset) for dataset in datasets]
    candidate_pass = {
        renderer
        for renderer in ADOPTABLE_RENDERERS
        if all(dataset["adoption_gates"].get(renderer, {}).get("pass", False) for dataset in dataset_results)
    }
    decisions = {dataset["decision"] for dataset in dataset_results}
    if "STOP" in decisions:
        final = "STOP"
    elif candidate_pass:
        final = "ADOPTION_CANDIDATE"
    else:
        final = "DIAGNOSTIC_ONLY"

    query_signs = [
        _sign(
            dataset.get("semantic", {})
            .get("semantic_hidden_carrier", {})
            .get("semantic_separation_query_shell1")
        )
        for dataset in dataset_results
    ]
    tree_signs = [
        _sign(
            dataset.get("semantic", {})
            .get("semantic_hidden_carrier", {})
            .get("semantic_separation_tree_shell1")
        )
        for dataset in dataset_results
    ]
    recommendation = _recommendation(final, sorted(candidate_pass), dataset_results)
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": dataset_results,
        "candidate_pass_renderers": sorted(candidate_pass),
        "semantic_query_shell1_signs": query_signs,
        "semantic_tree_shell1_signs": tree_signs,
        "semantic_attribution_direction_consistent": len(set(query_signs) - {"zero", "missing"}) <= 1
        and len(set(tree_signs) - {"zero", "missing"}) <= 1,
        "final_decision": final,
        "next_recommendation": recommendation,
    }


def _sign(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "missing"
    if float(value) > 0:
        return "positive"
    if float(value) < 0:
        return "negative"
    return "zero"


def _recommendation(final: str, candidates: list[str], datasets: list[dict[str, Any]]) -> str:
    if final == "STOP":
        return "Stop semantic rendering changes until cache, prompt, sample, leakage, or metric invariants are repaired."
    if final == "ADOPTION_CANDIDATE" and "tree_shell1_semantic_query_order" in candidates:
        return (
            "Treat tree_shell1_semantic_query_order as an adoption candidate, not an automatic adoption: "
            "it passed both 100-query datasets under common_qa, but should get larger-sample validation and "
            "a bridge-carrier safety check before paper-method promotion."
        )
    if candidates:
        return (
            "Treat the passing semantic renderer as a candidate for larger validation; keep semantic_weighted_tree "
            "diagnostic-only and do not claim semantic gains unless the B2/B3 deltas over B1 stay positive."
        )
    dominant = {dataset.get("dominant_effect") for dataset in datasets}
    if dominant == {"shell_expansion"}:
        return "Do not claim semantic ordering gains; investigate graph shell expansion or principled tree-budget allocation next."
    return "Keep this diagnostic-only and inspect whether graph shell expansion, prompt formatting, or tree-budget allocation is the next bottleneck."


def _semantic_group_table(attr: dict[str, Any]) -> list[str]:
    groups = (
        "current_only_answer",
        "current_only_non_answer",
        "shell1_answer",
        "shell1_non_answer",
        "tree_answer",
        "tree_non_answer",
        "projected_nonrendered_answer",
    )
    lines = [
        "| group | count | mean d_ang(q,u) | median d_ang(q,u) | mean d_ang(u,T_q) | median d_ang(u,T_q) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    _fmt(attr.get(f"{group}_count")),
                    _fmt(attr.get(f"mean_d_ang_query_{group}")),
                    _fmt(attr.get(f"median_d_ang_query_{group}")),
                    _fmt(attr.get(f"mean_d_ang_tree_{group}")),
                    _fmt(attr.get(f"median_d_ang_tree_{group}")),
                ]
            )
            + " |"
        )
    return lines


def _dataset_markdown(result: dict[str, Any]) -> list[str]:
    rows = result["rows"]
    semantic = result.get("semantic", {})
    attr = semantic.get("semantic_hidden_carrier", {})
    pools = semantic.get("pool_sizes", {})
    cache = result.get("cache", {})
    prompt = result.get("prompt_protocol", {})
    lines = [
        f"## {result.get('dataset')}",
        "",
        f"- Decision: **{result.get('decision')}**",
        f"- Reason: {result.get('reason')}",
        f"- Dominant answer-coverage effect vs strict tree: **{result.get('dominant_effect')}**",
        f"- Same sample: `{result.get('same_sample', {}).get('same_sample')}` ({result.get('same_sample', {}).get('reason')})",
        f"- Prompt: `{prompt.get('qa_prompt_name')}` hash `{prompt.get('qa_prompt_hash')}` exact `{prompt.get('qa_prompt_text_exact_match')}`",
        f"- Embedding: `{cache.get('model_id')}` dim `{cache.get('embedding_dim')}` query coverage {_fmt(cache.get('query_embedding_coverage'))}, chunk coverage {_fmt(cache.get('chunk_embedding_coverage_for_diagnostics'))}",
        "",
        "### Variant Table",
        "",
        "| renderer | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["renderer_mode"],
                    _fmt(row["answer_in_context"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["generation_ms"]),
                    _fmt(row["total_ms"]),
                    _fmt(row["answer_on_support_tree_rate"]),
                    _fmt(row["answer_in_shell1_rate"]),
                    _fmt(row["answer_rendered_from_shell1_rate"]),
                    _fmt(row["bridge_or_path_answer_retained_rate"]),
                    _fmt(row["bridge_or_path_answer_pushed_after_budget_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Decomposition Deltas",
            "",
            "| effect | answer_in_context | rendered_recall | context_f1 | qa_f1 | tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, delta in result["deltas"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(delta.get("answer_in_context")),
                    _fmt(delta.get("rendered_recall")),
                    _fmt(delta.get("context_f1")),
                    _fmt(delta.get("qa_f1")),
                    _fmt(delta.get("tokens")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Semantic Attribution",
            "",
            f"- semantic_separation_query_current_only: {_fmt(attr.get('semantic_separation_query_current_only'))}",
            f"- semantic_separation_tree_current_only: {_fmt(attr.get('semantic_separation_tree_current_only'))}",
            f"- semantic_separation_query_shell1: {_fmt(attr.get('semantic_separation_query_shell1'))}",
            f"- semantic_separation_tree_shell1: {_fmt(attr.get('semantic_separation_tree_shell1'))}",
            "",
            *_semantic_group_table(attr),
            "",
            "### Pool And Bridge Diagnostics",
            "",
            f"- avg strict tree chunks: {_fmt(pools.get('avg_support_tree_chunks'))}",
            f"- avg shell1 chunks: {_fmt(pools.get('avg_shell1_chunks'))}",
            f"- avg shell2 chunks: {_fmt(pools.get('avg_shell2_chunks'))}",
            f"- answer on support tree rate: {_fmt(pools.get('answer_on_support_tree_rate'))}",
            f"- answer in shell1 rate: {_fmt(pools.get('answer_in_shell1_rate'))}",
            f"- answer in shell2 rate: {_fmt(pools.get('answer_in_shell2_rate'))}",
            "",
            "### Adoption Gates",
            "",
        ]
    )
    for renderer, gate in result["adoption_gates"].items():
        lines.append(f"- `{renderer}`: pass `{gate['pass']}`, blockers `{', '.join(gate['blockers']) or 'none'}`")
    lines.append("")
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    first_prompt_hash = None
    for dataset in result["datasets"]:
        first_prompt_hash = dataset.get("prompt_protocol", {}).get("qa_prompt_hash")
        if first_prompt_hash:
            break
    lines = [
        "# Semantic Effect Decomposition Report",
        "",
        f"- Branch: `{result.get('branch')}`",
        f"- Commit: `{result.get('commit')}`",
            f"- Fixed common_qa prompt hash: `{first_prompt_hash}`",
            f"- Final decision: **{result.get('final_decision')}**",
            f"- Adoption-candidate renderers: `{', '.join(result.get('candidate_pass_renderers', [])) or 'none'}`",
            f"- Next recommendation: {result.get('next_recommendation')}",
            "",
        "Previous semantic rerun summary: valid local NV-Embed-v2 embeddings replaced legacy 128D vectors, but the earlier rerun stopped because the semantic attribution direction was not stable across 2Wiki and Hotpot. This run reuses the validated embedding cache and decomposes shell expansion from semantic ordering under a fixed prompt.",
        "",
        "Theory boundary: the PAMAE entity-chunk retrieval core, graph-metric medoid selection, local refinement, support-tree construction, context budget, generator, evaluator, dataset order, and embedding cache remain unchanged. Semantic distance is normalized angular distance and is used only inside graph-defined candidates or the diagnostic semantic-weighted tree.",
        "",
    ]
    for dataset in result["datasets"]:
        lines.extend(_dataset_markdown(dataset))
    lines.extend(
        [
            "## Cross-Dataset Interpretation",
            "",
            f"- Query shell1 attribution signs: `{', '.join(result.get('semantic_query_shell1_signs', []))}`",
            f"- Tree shell1 attribution signs: `{', '.join(result.get('semantic_tree_shell1_signs', []))}`",
            f"- Attribution direction consistent: `{result.get('semantic_attribution_direction_consistent')}`",
            f"- Next recommendation: {result.get('next_recommendation')}",
            "",
            "Expert-panel read: if B1 improves over A1 but B2/B3 do not improve over B1, the gain is graph shell expansion rather than semantic ordering. If query-semantic ordering damages bridge/path retention, it is risky for GraphRAG even when Hotpot improves. If B4 improves alone, semantic edge lengths remain diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic effect decomposition variants.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = compare(args.root, args.datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(result), encoding="utf-8")
    (args.out.parent / "semantic_effect_decomposition_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
