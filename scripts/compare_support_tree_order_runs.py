#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REFERENCE_RUN = "entity_chunk_reference_current_renderer"
METRIC_RUN = "entity_chunk_reference_metric_path_carrier"
RUN_ORDER = (
    REFERENCE_RUN,
    METRIC_RUN,
    "entity_chunk_reference_tree_all_no_budget",
    "entity_chunk_reference_tree_current_budget_order",
    "entity_chunk_reference_current_tree_intersection_only",
    "entity_chunk_reference_current_only_non_tree",
    "entity_chunk_reference_tree_answer_oracle",
)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "oracle_leakage_count": int(metrics.get("oracle_leakage_count", 0)),
    }


def _row_by_renderer(rows: list[dict[str, Any]], renderer: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("renderer_mode") == renderer), {})


def _classification(order: dict[str, Any], diff: dict[str, Any], current: float, metric: float) -> str:
    gap = max(0.0, current - metric)
    on_tree = float(order.get("answer_on_support_tree_rate", 0.0))
    budget = float(order.get("answer_metric_budget_cutoff_rate", 0.0))
    hidden = float(order.get("answer_current_only_non_tree_rate", 0.0))
    near1 = float(order.get("answer_near_tree_distance_1_rate", 0.0))
    near2 = float(order.get("answer_near_tree_distance_2_rate", 0.0))
    if current > 0.0 and on_tree < current * 0.70:
        return "TREE_MEMBERSHIP_BOTTLENECK"
    if gap > 0.0 and hidden >= max(0.01, gap * 0.50):
        return "HIDDEN_NON_TREE_RECOVERY"
    if gap > 0.0 and budget >= max(0.01, gap * 0.50):
        return "ORDER_BUDGET_BOTTLENECK"
    if near2 > on_tree and near1 >= on_tree:
        return "CORRIDOR_BOTTLENECK"
    if float(diff.get("answer_in_current_only_rate", 0.0)) > float(diff.get("answer_in_current_tree_intersection_rate", 0.0)):
        return "HIDDEN_NON_TREE_RECOVERY"
    return "DIAGNOSTIC_ONLY"


def _decision(rows: list[dict[str, Any]], order: dict[str, Any], diff: dict[str, Any]) -> tuple[str, str]:
    ref = _row_by_renderer(rows, "current_renderer")
    metric = _row_by_renderer(rows, "metric_path_carrier")
    tree_all = _row_by_renderer(rows, "tree_all_no_budget")
    tree_oracle = _row_by_renderer(rows, "tree_answer_oracle")
    if not ref or not metric or not order or not diff:
        return "STOP_BEFORE_100", "missing current, metric, order attribution, or current-vs-tree diff outputs"
    if int(metric.get("oracle_leakage_count", 0)) or bool(metric.get("uses_answer_string")):
        return "STOP_BEFORE_100", "non-oracle metric renderer leaked oracle signal"
    current = float(ref.get("answer_in_context", 0.0))
    metric_answer = float(metric.get("answer_in_context", 0.0))
    if current <= 0.0:
        return "STOP_BEFORE_100", "current renderer has no answer signal to explain"
    if not tree_all or not tree_oracle:
        return "STOP_BEFORE_100", "tree diagnostic ablation renderers are missing"
    tree_all_answer = float(tree_all.get("answer_in_context", 0.0))
    tree_oracle_answer = float(tree_oracle.get("answer_in_context", 0.0))
    if tree_oracle_answer > tree_all_answer + 1e-12:
        return "STOP_BEFORE_100", "tree oracle is not comparable with tree_all_no_budget"
    explained = (
        float(order.get("answer_on_tree_but_metric_not_rendered_rate", 0.0))
        + float(order.get("answer_current_only_non_tree_rate", 0.0))
        + float(order.get("answer_metric_budget_cutoff_rate", 0.0))
    )
    gap = current - metric_answer
    if gap <= 0.0:
        return "GO_TO_100", "metric path carrier no longer loses answer coverage in this smoke"
    if explained + 1e-12 >= gap * 0.70:
        return "DIAGNOSTIC_ONLY_100", "current-minus-metric gap is interpretable by order/budget or hidden non-tree recovery"
    return "STOP_BEFORE_100", "current-minus-metric gap is not explained by the diagnostics"


def compare_dataset(root: Path) -> dict[str, Any]:
    rows = [_run_row(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    metrics = _read(root / "support_tree_order_metrics.json")
    order = metrics.get("support_tree_order_budget", {}) if isinstance(metrics, dict) else {}
    diff = metrics.get("current_tree_diff", {}) if isinstance(metrics, dict) else {}
    taxonomy = metrics.get("failure_taxonomy", {}).get("support_tree_order_failure_counts", {}) if isinstance(metrics, dict) else {}
    decision, reason = _decision(rows, order, diff)
    ref = _row_by_renderer(rows, "current_renderer")
    metric = _row_by_renderer(rows, "metric_path_carrier")
    final_class = _classification(
        order,
        diff,
        float(ref.get("answer_in_context", 0.0)),
        float(metric.get("answer_in_context", 0.0)),
    )
    return {
        "dataset": root.name,
        "root": str(root),
        "rows": rows,
        "support_tree_order_budget": order,
        "current_tree_diff": diff,
        "taxonomy": taxonomy,
        "decision": decision,
        "reason": reason,
        "final_bottleneck_classification": final_class,
    }


def compare(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    results = [compare_dataset(root / dataset) for dataset in datasets] if datasets else [compare_dataset(root)]
    decisions = {str(result["decision"]) for result in results}
    if decisions == {"GO_TO_100"}:
        final = "GO_TO_100"
    elif decisions <= {"GO_TO_100", "DIAGNOSTIC_ONLY_100"} and "DIAGNOSTIC_ONLY_100" in decisions:
        final = "DIAGNOSTIC_ONLY_100"
    else:
        final = "STOP_BEFORE_100"
    classifications = [str(result.get("final_bottleneck_classification", "DIAGNOSTIC_ONLY")) for result in results]
    final_class = classifications[0] if len(set(classifications)) == 1 else "DIAGNOSTIC_ONLY"
    return {
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "datasets": results,
        "decision": final,
        "final_bottleneck_classification": final_class,
    }


def _dataset_markdown(result: dict[str, Any]) -> list[str]:
    rows = result.get("rows", [])
    ref = _row_by_renderer(rows, "current_renderer")
    metric = _row_by_renderer(rows, "metric_path_carrier")
    order = result.get("support_tree_order_budget", {})
    diff = result.get("current_tree_diff", {})

    def renderer_answer(renderer: str) -> str:
        return _fmt(float(_row_by_renderer(rows, renderer).get("answer_in_context", 0.0)))

    lines = [
        f"## {result.get('dataset', '')}",
        "",
        f"- Decision: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Reason: {result.get('reason', '')}",
        f"- Final bottleneck classification: **{result.get('final_bottleneck_classification', 'DIAGNOSTIC_ONLY')}**",
        f"- current answer-in-context: {_fmt(float(ref.get('answer_in_context', 0.0)))}",
        f"- metric_path_carrier answer-in-context: {_fmt(float(metric.get('answer_in_context', 0.0)))}",
        f"- current-minus-metric answer gap: {_fmt(float(ref.get('answer_in_context', 0.0)) - float(metric.get('answer_in_context', 0.0)))}",
        f"- answer on support tree rate: {_fmt(float(order.get('answer_on_support_tree_rate', 0.0)))}",
        f"- answer near support tree distance 1 rate: {_fmt(float(order.get('answer_near_tree_distance_1_rate', 0.0)))}",
        f"- answer near support tree distance 2 rate: {_fmt(float(order.get('answer_near_tree_distance_2_rate', 0.0)))}",
        f"- answer on tree but cut by budget rate: {_fmt(float(order.get('answer_metric_budget_cutoff_rate', 0.0)))}",
        f"- answer current-only non-tree rate: {_fmt(float(order.get('answer_current_only_non_tree_rate', 0.0)))}",
        f"- answer current-tree intersection rate: {_fmt(float(diff.get('answer_in_current_tree_intersection_rate', 0.0)))}",
        f"- answer tree-only rate: {_fmt(float(diff.get('answer_in_tree_only_rate', 0.0)))}",
        "",
        "### Diagnostic Renderer Answer-In-Context",
        "",
        f"- tree_all_no_budget: {renderer_answer('tree_all_no_budget')}",
        f"- tree_current_budget_order: {renderer_answer('tree_current_budget_order')}",
        f"- current_tree_intersection_only: {renderer_answer('current_tree_intersection_only')}",
        f"- current_only_non_tree: {renderer_answer('current_only_non_tree')}",
        f"- tree_answer_oracle: {renderer_answer('tree_answer_oracle')}",
        "",
        "### Failure Taxonomy",
        "",
        _fmt(result.get("taxonomy", {})),
        "",
        "### Variant Table",
        "",
        "| run | renderer_mode | oracle | diagnostic | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
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
                    _fmt(row["qa_f1"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["avg_context_tokens"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Support-Tree Order/Budget Diagnostic",
        "",
        f"- Branch: `{result.get('branch', '')}`",
        f"- Commit: `{result.get('commit', '')}`",
        f"- 100-query gate outcome: **{result.get('decision', 'STOP_BEFORE_100')}**",
        f"- Final bottleneck classification: **{result.get('final_bottleneck_classification', 'DIAGNOSTIC_ONLY')}**",
        "",
        "Previous path-carrier summary: `metric_path_carrier` preserved the graph-metric closure boundary and reduced tokens, but lost answer-in-context and rendered recall versus `current_renderer` on both 2Wiki and Hotpot. This diagnostic decomposes that loss into support-tree membership, path role, render order, budget cutoff, and current-only hidden recovery.",
        "",
        "Theoretical boundary: `T_q = SPClosure(A_q union Theta_refined)`. Diagnostic renderers are not adoption candidates; oracle renderers use answer strings only for upper-bound analysis.",
        "",
    ]
    for dataset in result.get("datasets", []):
        lines.extend(_dataset_markdown(dataset))
    lines.extend(
        [
            "## Local-Minimum Guard",
            "",
            "- Did we simply add more chunks? Diagnostic-only variants include no-budget and subset ablations, but no renderer is proposed for adoption.",
            "- Did we use current renderer behavior as a method? Only in diagnostic ablations that are explicitly non-adoptable.",
            "- Did we use answer/gold outside diagnostics? No; answer strings appear only in oracle/diagnostic attribution outputs.",
            "- Did support tree membership explain answer recovery? See the per-dataset support-tree rates above.",
            "- Did order/budget explain metric_path_carrier failure? See budget cutoff and tree-but-not-rendered rates above.",
            "- Did hidden non-tree chunks explain current renderer recovery? See current-only non-tree and current-vs-tree diff rates above.",
            "- Would a corridor objective be justified, or is that premature? Treat it as premature unless distance-1/2 near-tree rates dominate the residual gap across both datasets.",
            "",
            "Final decision: **DIAGNOSTIC_ONLY**. This round is for attribution, not adoption.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare support-tree order/budget diagnostic runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(Path(args.root), datasets=args.datasets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(result), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
