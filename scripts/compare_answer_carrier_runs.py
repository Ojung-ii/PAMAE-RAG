#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REFERENCE_RUN = "entity_chunk_reference_current_renderer"
RUN_ORDER = (
    REFERENCE_RUN,
    "entity_chunk_reference_projected_answer_chunk_oracle",
    "entity_chunk_reference_selected_basin_answer_chunk_oracle",
    "entity_chunk_reference_current_answer_role_oracle",
    "entity_chunk_reference_gold_chunk_role_oracle",
)
ORACLE_F1_FIELDS = {
    "projected_answer_chunk_oracle": "projected_answer_chunk_oracle_f1",
    "selected_basin_answer_chunk_oracle": "selected_basin_answer_chunk_oracle_f1",
    "current_answer_role_oracle": "current_answer_role_oracle_f1",
    "gold_chunk_role_oracle": "gold_chunk_role_oracle_f1",
}


def _read(path: Path) -> dict[str, Any]:
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


def _row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "answer_carrier_metrics.json")
    carrier = metrics.get("answer_carrier_attribution")
    if not isinstance(carrier, dict):
        carrier = {}
    return {
        "run": run_dir.name,
        "renderer_mode": metrics.get("renderer_mode", ""),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "local_objective_invalid_count": int(metrics.get("local_objective_invalid_count", 0)),
        "answer_chunk_candidate_rate": float(carrier.get("answer_chunk_candidate_rate", 0.0)),
        "answer_chunk_projected_rate": float(carrier.get("answer_chunk_projected_rate", 0.0)),
        "answer_chunk_selected_medoid_rate": float(carrier.get("answer_chunk_selected_medoid_rate", 0.0)),
        "answer_chunk_post_refine_medoid_rate": float(carrier.get("answer_chunk_post_refine_medoid_rate", 0.0)),
        "answer_chunk_selected_basin_rate": float(carrier.get("answer_chunk_selected_basin_rate", 0.0)),
        "answer_chunk_support_tree_rate": float(carrier.get("answer_chunk_support_tree_rate", 0.0)),
        "answer_chunk_bridge_rate": float(carrier.get("answer_chunk_bridge_rate", 0.0)),
        "answer_chunk_current_rendered_rate": float(carrier.get("answer_chunk_current_rendered_rate", 0.0)),
        "answer_chunk_rendered_nonmedoid_rate": float(carrier.get("answer_chunk_rendered_nonmedoid_rate", 0.0)),
        "answer_chunk_budget_cutoff_rate": float(carrier.get("answer_chunk_budget_cutoff_rate", 0.0)),
        "current_answer_in_context": float(carrier.get("current_answer_in_context", metrics.get("answer_in_context", 0.0))),
        "selected_medoid_answer_availability": float(carrier.get("selected_medoid_answer_availability", 0.0)),
        "current_minus_medoid_answer_gap": float(carrier.get("current_minus_medoid_answer_gap", 0.0)),
        "answer_render_role_distribution": carrier.get("answer_render_role_distribution", {}),
        "gold_render_role_distribution": carrier.get("gold_render_role_distribution", {}),
        "answer_carrier_failure_counts": carrier.get("answer_carrier_failure_counts", {}),
    }


def compare(root: Path) -> dict[str, Any]:
    rows = [_row(root / run) for run in RUN_ORDER if (root / run / "answer_carrier_metrics.json").exists()]
    if not rows:
        return {"rows": [], "decision": "STOP_BEFORE_100", "reason": "no answer carrier runs found"}
    ref = next((row for row in rows if row["run"] == REFERENCE_RUN), rows[0])
    oracle_f1 = {
        ORACLE_F1_FIELDS[row["renderer_mode"]]: row["qa_f1"]
        for row in rows
        if row["renderer_mode"] in ORACLE_F1_FIELDS
    }
    decision, reason = _decision(ref, rows)
    return {
        "rows": rows,
        "reference": ref,
        "oracle_f1": oracle_f1,
        "decision": decision,
        "reason": reason,
    }


def _decision(ref: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str]:
    if ref["triangle_inequality_violation_count"] != 0 or ref["local_objective_invalid_count"] != 0:
        return "STOP_BEFORE_100", "triangle/objective invariant failed"
    if ref["answer_chunk_candidate_rate"] <= 0.0:
        return "STOP_BEFORE_100", "answer chunk mapping/candidate attribution is empty"
    role_dist = ref.get("answer_render_role_distribution", {})
    role_total = sum(int(value or 0) for value in role_dist.values()) if isinstance(role_dist, dict) else 0
    if role_total <= 0 and ref["current_answer_in_context"] > 0.0:
        return "STOP_BEFORE_100", "current renderer answer context is not explained by role tracing"
    if ref["current_minus_medoid_answer_gap"] > 0.05 and role_total > 0:
        return "GO_TO_100", "current-minus-medoid answer gap is explained by rendered non-medoid/path roles"
    if ref["answer_chunk_projected_rate"] > ref["answer_chunk_post_refine_medoid_rate"] + 0.05:
        return "DIAGNOSTIC_ONLY_100", "answer chunks are projected more often than they are medoids; 100-query stability check is diagnostic"
    return "STOP_BEFORE_100", "50-query attribution did not reveal an interpretable continuation signal"


def markdown(summary: dict[str, Any]) -> str:
    branch = _git_value(["branch", "--show-current"])
    commit = _git_value(["rev-parse", "--short", "HEAD"])
    ref = summary.get("reference", {})
    oracle_f1 = summary.get("oracle_f1", {})
    lines = [
        "# Answer Carrier Comparison 50",
        "",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Final decision: **{summary.get('decision', 'STOP_BEFORE_100')}**",
        f"- Reason: {summary.get('reason', '')}",
        "",
        "## Required Headline Metrics",
        "",
        f"- current_renderer answer-in-context: {_fmt(float(ref.get('current_answer_in_context', 0.0)))}",
        f"- selected medoid chunk answer availability: {_fmt(float(ref.get('selected_medoid_answer_availability', 0.0)))}",
        f"- current_minus_medoid_answer_gap: {_fmt(float(ref.get('current_minus_medoid_answer_gap', 0.0)))}",
        "",
        "## Answer Carrier Stage Table",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "answer_chunk_candidate_rate",
        "answer_chunk_projected_rate",
        "answer_chunk_selected_medoid_rate",
        "answer_chunk_post_refine_medoid_rate",
        "answer_chunk_selected_basin_rate",
        "answer_chunk_support_tree_rate",
        "answer_chunk_bridge_rate",
        "answer_chunk_current_rendered_rate",
        "answer_chunk_rendered_nonmedoid_rate",
        "answer_chunk_budget_cutoff_rate",
    ):
        lines.append(f"| {key} | {_fmt(float(ref.get(key, 0.0)))} |")
    lines.extend(
        [
            "",
            "## Render Role Distribution",
            "",
            f"- answer render roles: {_fmt(ref.get('answer_render_role_distribution', {}))}",
            f"- gold render roles: {_fmt(ref.get('gold_render_role_distribution', {}))}",
            "",
            "## Failure Taxonomy",
            "",
            f"{_fmt(ref.get('answer_carrier_failure_counts', {}))}",
            "",
            "## Oracle QA F1",
            "",
            "| oracle | qa_f1 |",
            "| --- | ---: |",
        ]
    )
    for key in (
        "projected_answer_chunk_oracle_f1",
        "selected_basin_answer_chunk_oracle_f1",
        "current_answer_role_oracle_f1",
        "gold_chunk_role_oracle_f1",
    ):
        lines.append(f"| {key} | {_fmt(float(oracle_f1.get(key, 0.0)))} |")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | renderer_mode | oracle | answer_in_context | qa_f1 | avg_context_tokens | triangle_inequality_violation_count | local_objective_invalid_count |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["renderer_mode"],
                    str(row["oracle_renderer"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["qa_f1"]),
                    _fmt(row["avg_context_tokens"]),
                    _fmt(row["triangle_inequality_violation_count"]),
                    _fmt(row["local_objective_invalid_count"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answer carrier attribution diagnostic runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = compare(Path(args.root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
