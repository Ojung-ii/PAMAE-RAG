#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.diagnostics.b2_robustness import flattened_delta_means, paired_deltas
from pamae_rag.diagnostics.semantic_effect_decomposition import load_json, prompt_protocol_status, same_query_sample

EXPECTED_PROMPT_HASH = "31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d"
B2 = "tree_shell1_semantic_query_order"
CURRENT = "current_renderer"
B1 = "tree_shell1_graph_order"
A1 = "metric_path_carrier"
PRIMARY_DATASETS = {"2wikimultihopqa", "hotpotqa"}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_run(root: Path, renderer: str) -> dict[str, Any]:
    run_dir = root / f"entity_chunk_reference_{renderer}"
    metrics = load_json(run_dir / "answer_carrier_metrics.json")
    qa_metrics = load_json(run_dir / "qa_metrics.json")
    if not metrics:
        return {"renderer_mode": renderer, "missing": True}
    return {
        "renderer_mode": renderer,
        "missing": False,
        "em": float(qa_metrics.get("mean_exact_match", 0.0)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": float(metrics.get("rendered_recall", 0.0)),
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "generation_ms": float(metrics.get("generation_ms", 0.0)),
        "total_ms": float(metrics.get("total_ms", 0.0)),
        "answer_on_support_tree_rate": float(metrics.get("answer_on_support_tree_rate", 0.0)),
        "answer_in_shell1_rate": float(metrics.get("answer_in_shell1_rate", 0.0)),
        "answer_rendered_from_shell1_rate": float(metrics.get("answer_rendered_from_shell1_rate", 0.0)),
        "bridge_or_path_answer_retained_rate": float(metrics.get("bridge_or_path_answer_retained_rate", 0.0)),
        "bridge_or_path_answer_pushed_after_budget_rate": float(
            metrics.get("bridge_or_path_answer_pushed_after_budget_rate", 0.0)
        ),
        "shell1_chunk_count": float(metrics.get("shell1_chunk_count", 0.0)),
        "rendered_shell1_chunk_count": float(metrics.get("rendered_shell1_chunk_count", 0.0)),
        "semantic_order_pushed_bridge_after_budget_rate": float(
            metrics.get("semantic_order_pushed_bridge_after_budget_rate", 0.0)
        ),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
        "score_mixing_detected": bool(metrics.get("score_mixing_detected", False)),
        "qa_prompt_name": metrics.get("qa_prompt_name"),
        "qa_prompt_hash": metrics.get("qa_prompt_hash"),
        "qa_prompt_text_exact_match": bool(metrics.get("qa_prompt_text_exact_match", False)),
    }


def _cache_ok(cache: dict[str, Any]) -> bool:
    return (
        str(cache.get("model_id")) == "nvidia/NV-Embed-v2"
        and int(cache.get("embedding_dim", 0) or 0) == 4096
        and float(cache.get("query_embedding_coverage", 0.0)) >= 1.0
        and float(cache.get("chunk_embedding_coverage_for_diagnostics", 0.0)) >= 1.0
        and bool(cache.get("all_vectors_l2_normalized", False))
    )


def _b2_gate(rows_by_renderer: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    ref = rows_by_renderer.get(CURRENT, {})
    b2 = rows_by_renderer.get(B2, {})
    blockers: list[str] = []
    if ref.get("missing") or b2.get("missing"):
        return False, ["missing current or B2 metrics"]
    if b2["qa_f1"] + 1e-12 < ref["qa_f1"]:
        blockers.append("qa_f1_regression")
    if b2["answer_in_context"] + 1e-12 < ref["answer_in_context"]:
        blockers.append("answer_in_context_regression")
    if b2["rendered_recall"] + 1e-12 < ref["rendered_recall"]:
        blockers.append("rendered_recall_regression")
    if b2["context_f1"] + 1e-12 < ref["context_f1"]:
        blockers.append("context_f1_regression")
    if b2["avg_context_tokens"] > ref["avg_context_tokens"] * 1.10 + 1e-12:
        blockers.append("token_gate")
    if b2["retrieval_ms"] > ref["retrieval_ms"] * 1.25 + 1e-12:
        blockers.append("retrieval_time_gate")
    if b2["oracle_leakage_count"] != 0:
        blockers.append("oracle_leakage")
    if b2["score_mixing_detected"]:
        blockers.append("score_mixing")
    if b2["qa_prompt_name"] != "common_qa" or b2["qa_prompt_hash"] != EXPECTED_PROMPT_HASH:
        blockers.append("prompt_hash")
    if not b2["qa_prompt_text_exact_match"]:
        blockers.append("prompt_text_mismatch")
    if b2["semantic_order_pushed_bridge_after_budget_rate"] > ref["bridge_or_path_answer_pushed_after_budget_rate"] + 0.05:
        blockers.append("bridge_budget_risk")
    return not blockers, blockers


def _strong_confirmation(rows_by_renderer: dict[str, dict[str, Any]], paired: dict[str, Any]) -> tuple[bool, list[str]]:
    ref = rows_by_renderer.get(CURRENT, {})
    b2 = rows_by_renderer.get(B2, {})
    blockers: list[str] = []
    if not ref or not b2 or ref.get("missing") or b2.get("missing"):
        return False, ["missing current or B2"]
    if b2["qa_f1"] <= ref["qa_f1"] + 1e-12:
        blockers.append("qa_f1_not_strictly_better")
    if b2["answer_in_context"] <= ref["answer_in_context"] + 1e-12:
        blockers.append("answer_in_context_not_strictly_better")
    if b2["rendered_recall"] + 1e-12 < ref["rendered_recall"]:
        blockers.append("rendered_recall_regression")
    if b2["avg_context_tokens"] > ref["avg_context_tokens"] + 1e-12:
        blockers.append("strict_token_gate")
    if b2["retrieval_ms"] > ref["retrieval_ms"] * 1.10 + 1e-12:
        blockers.append("strict_time_gate")
    b2_b1 = paired.get("paired_delta_B2_minus_B1", {}).get("metrics", {})
    if b2_b1.get("answer_in_context", {}).get("mean", -1.0) < -1e-12:
        blockers.append("B2_minus_B1_answer_negative")
    if b2_b1.get("rendered_recall", {}).get("mean", -1.0) < -1e-12:
        blockers.append("B2_minus_B1_recall_negative")
    return not blockers, blockers


def _paired(root: Path) -> dict[str, Any]:
    qa = {
        renderer: _read_jsonl(root / f"entity_chunk_reference_{renderer}" / "qa.jsonl")
        for renderer in (CURRENT, A1, B1, B2)
    }
    out: dict[str, Any] = {}
    if qa[B2] and qa[CURRENT]:
        delta = paired_deltas(candidate_rows=qa[B2], baseline_rows=qa[CURRENT])
        out["paired_delta_B2_minus_current"] = delta
        out["paired_delta_B2_minus_current_flat"] = flattened_delta_means(delta)
    if qa[B2] and qa[B1]:
        delta = paired_deltas(candidate_rows=qa[B2], baseline_rows=qa[B1])
        out["paired_delta_B2_minus_B1"] = delta
        out["paired_delta_B2_minus_B1_flat"] = flattened_delta_means(delta)
    if qa[B1] and qa[A1]:
        delta = paired_deltas(candidate_rows=qa[B1], baseline_rows=qa[A1])
        out["paired_delta_B1_minus_A1"] = delta
        out["paired_delta_B1_minus_A1_flat"] = flattened_delta_means(delta)
    return out


def compare_dataset(root: Path, variants: list[str]) -> dict[str, Any]:
    unavailable = root / "SECONDARY_UNAVAILABLE.md"
    if unavailable.exists():
        return {
            "dataset": root.name,
            "unavailable": True,
            "reason": unavailable.read_text(encoding="utf-8").strip(),
            "decision": "DIAGNOSTIC_ONLY",
        }
    rows = [_read_run(root, renderer) for renderer in variants]
    by_renderer = {row["renderer_mode"]: row for row in rows}
    cache = load_json(root / "compatible_embedding_cache_summary.json")
    semantic = load_json(root / "semantic_effect_decomposition_metrics.json") or load_json(
        root / "semantic_attribution_metrics.json"
    )
    sample = same_query_sample([root / f"entity_chunk_reference_{renderer}" / "qa.jsonl" for renderer in variants])
    prompt = prompt_protocol_status([row for row in rows if not row.get("missing")])
    paired = _paired(root)
    gate_pass, gate_blockers = _b2_gate(by_renderer)
    strong, strong_blockers = _strong_confirmation(by_renderer, paired)
    decision = "ADOPTION_CANDIDATE_CONFIRMED" if gate_pass else "STOP"
    if not _cache_ok(cache):
        decision = "STOP"
        gate_blockers = [*gate_blockers, "embedding_cache"]
    if not sample.get("same_sample", False):
        decision = "STOP"
        gate_blockers = [*gate_blockers, "sample_mismatch"]
    if not prompt.get("qa_prompt_consistent", False):
        decision = "STOP"
        gate_blockers = [*gate_blockers, "prompt_protocol"]
    attr = semantic.get("semantic_hidden_carrier", {})
    return {
        "dataset": root.name,
        "unavailable": False,
        "rows": rows,
        "cache": cache,
        "semantic": semantic,
        "same_sample": sample,
        "prompt_protocol": prompt,
        "paired": paired,
        "b2_gate_pass": gate_pass,
        "b2_gate_blockers": sorted(set(gate_blockers)),
        "strong_confirmation": strong,
        "strong_blockers": sorted(set(strong_blockers)),
        "semantic_attribution_query_separation": attr.get("semantic_separation_query_shell1"),
        "semantic_attribution_tree_separation": attr.get("semantic_separation_tree_shell1"),
        "decision": decision,
    }


def compare(root: Path, datasets: list[str], variants: list[str]) -> dict[str, Any]:
    dataset_results = [compare_dataset(root / dataset, variants) for dataset in datasets]
    primary = [dataset for dataset in dataset_results if dataset["dataset"] in PRIMARY_DATASETS]
    primary_pass = bool(primary) and all(dataset.get("decision") == "ADOPTION_CANDIDATE_CONFIRMED" for dataset in primary)
    secondary_missing = any(dataset.get("unavailable") for dataset in dataset_results if dataset["dataset"] not in PRIMARY_DATASETS)
    if not primary_pass:
        final = "STOP"
    elif secondary_missing:
        final = "DIAGNOSTIC_ONLY"
    else:
        final = "ADOPTION_CANDIDATE_CONFIRMED"
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": dataset_results,
        "variants": variants,
        "primary_pass": primary_pass,
        "secondary_missing": secondary_missing,
        "final_decision": final,
        "next_recommendation": _next_recommendation(final, primary_pass, secondary_missing),
    }


def _next_recommendation(final: str, primary_pass: bool, secondary_missing: bool) -> str:
    if final == "ADOPTION_CANDIDATE_CONFIRMED":
        return "Write the mainline integration plan, then validate the config flag in a separate integration branch before changing defaults."
    if primary_pass and secondary_missing:
        return "Keep B2 diagnostic-only until MuSiQue/PopQA configs or another secondary check are available; primary robustness is positive but not enough for mainline adoption by this protocol."
    return "Do not adopt B2; inspect the gate blockers and keep the graph-constrained semantic layer experimental."


def _variant_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| renderer | EM | QA F1 | answer_in_context | rendered_recall | context_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut | shell1_chunks | rendered_shell1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("missing"):
            lines.append(f"| {row['renderer_mode']} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["renderer_mode"],
                    _fmt(row["em"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["generation_ms"]),
                    _fmt(row["total_ms"]),
                    _fmt(row["answer_on_support_tree_rate"]),
                    _fmt(row["answer_in_shell1_rate"]),
                    _fmt(row["answer_rendered_from_shell1_rate"]),
                    _fmt(row["bridge_or_path_answer_retained_rate"]),
                    _fmt(row["bridge_or_path_answer_pushed_after_budget_rate"]),
                    _fmt(row["shell1_chunk_count"]),
                    _fmt(row["rendered_shell1_chunk_count"]),
                ]
            )
            + " |"
        )
    return lines


def _delta_table(paired: dict[str, Any]) -> list[str]:
    lines = [
        "| comparison | metric | mean | ci95 | improved | tied | regressed |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for name in ("paired_delta_B2_minus_current", "paired_delta_B2_minus_B1", "paired_delta_B1_minus_A1"):
        delta = paired.get(name, {})
        for metric_name, metric in delta.get("metrics", {}).items():
            ci = metric.get("ci95")
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        metric_name,
                        _fmt(metric.get("mean")),
                        ci_text,
                        str(metric.get("improved")),
                        str(metric.get("tied")),
                        str(metric.get("regressed")),
                    ]
                )
                + " |"
            )
    return lines


def _dataset_markdown(dataset: dict[str, Any]) -> list[str]:
    lines = [f"## {dataset['dataset']}", ""]
    if dataset.get("unavailable"):
        lines.extend(["- Decision: **DIAGNOSTIC_ONLY**", "- Secondary dataset unavailable.", "", "```text", dataset["reason"], "```", ""])
        return lines
    cache = dataset.get("cache", {})
    prompt = dataset.get("prompt_protocol", {})
    lines.extend(
        [
            f"- Decision: **{dataset.get('decision')}**",
            f"- B2 gate pass: `{dataset.get('b2_gate_pass')}` blockers `{', '.join(dataset.get('b2_gate_blockers', [])) or 'none'}`",
            f"- Strong confirmation: `{dataset.get('strong_confirmation')}` blockers `{', '.join(dataset.get('strong_blockers', [])) or 'none'}`",
            f"- Same sample: `{dataset.get('same_sample', {}).get('same_sample')}` ({dataset.get('same_sample', {}).get('reason')})",
            f"- Prompt: `{prompt.get('qa_prompt_name')}` hash `{prompt.get('qa_prompt_hash')}` exact `{prompt.get('qa_prompt_text_exact_match')}`",
            f"- Embedding: `{cache.get('model_id')}` dim `{cache.get('embedding_dim')}` query coverage {_fmt(cache.get('query_embedding_coverage'))}, chunk coverage {_fmt(cache.get('chunk_embedding_coverage_for_diagnostics'))}",
            f"- B2 semantic attribution query separation: {_fmt(dataset.get('semantic_attribution_query_separation'))}",
            f"- B2 semantic attribution tree separation: {_fmt(dataset.get('semantic_attribution_tree_separation'))}",
            "",
            "### Variant Table",
            "",
            *_variant_table(dataset.get("rows", [])),
            "",
            "### Paired Deltas",
            "",
            *_delta_table(dataset.get("paired", {})),
            "",
        ]
    )
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# B2 Adoption Candidate Validation Report",
        "",
        f"- Branch: `{result.get('branch')}`",
        f"- Commit: `{result.get('commit')}`",
        f"- Fixed common_qa prompt hash: `{EXPECTED_PROMPT_HASH}`",
        "- Embedding model: `nvidia/NV-Embed-v2`, dim `4096`, shared query/chunk space, L2 normalized.",
        f"- Variants: `{', '.join(result.get('variants', []))}`",
        f"- Primary pass: `{result.get('primary_pass')}`",
        f"- Secondary missing: `{result.get('secondary_missing')}`",
        f"- Final decision: **{result.get('final_decision')}**",
        f"- Next recommendation: {result.get('next_recommendation')}",
        "",
        "Verification status: compileall/pytest should be recorded from the run log; this report checks prompt, sample, embedding, score-mixing, oracle-leakage, and B2 gate fields from emitted artifacts.",
        "",
        "Variant definitions: A0 current_renderer is the reference; A1 metric_path_carrier renders strict support-tree chunks; B1 tree_shell1_graph_order renders T_q union S1 by graph order; B2 tree_shell1_semantic_query_order renders the same graph-constrained pool with query angular distance as a lexicographic tie-breaker.",
        "",
    ]
    for dataset in result["datasets"]:
        lines.extend(_dataset_markdown(dataset))
    lines.extend(
        [
            "## Local-Minimum Guard",
            "",
            "- Did we change the PAMAE core retrieval objective? No.",
            "- Did we use global dense retrieval? No.",
            "- Did we use scalar score mixing? No.",
            "- Did semantic ordering improve beyond shell expansion? See paired `B2_minus_B1` deltas above.",
            "- Did B2 preserve answer coverage and rendered recall? See B2 gate checklist above.",
            "- Did B2 remain stable on both 2Wiki and Hotpot? Captured by primary pass.",
            "- Did B2 avoid excessive token/time cost? Captured by token and retrieval gates.",
            "- Is the improvement likely general or dataset-specific? Secondary availability and results determine whether this stays diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B2 semantic carrier robustness runs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = compare(args.root, args.datasets, args.variants)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(result), encoding="utf-8")
    (args.out.parent / "b2_robustness_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
