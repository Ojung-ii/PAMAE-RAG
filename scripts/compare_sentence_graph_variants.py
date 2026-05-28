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
    "entity_sentence_sentence_only",
    "entity_sentence_sentence_path",
    "entity_sentence_chunk_hier_sentence_only",
    "entity_sentence_chunk_hier_sentence_path",
    "entity_sentence_chunk_hier_sentence_parent_title",
    "entity_sentence_chunk_hier_sentence_local_window",
    "entity_sentence_chunk_hier_sentence_parent_chunk",
)
GATE_FIELDS = (
    "answer_projected",
    "answer_rendered",
    "answer_in_context",
    "rendered_recall",
    "context_f1",
    "qa_f1",
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


def _mean_bool(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if bool(value)) / len(values))


def _path_answer_trace(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    path = diagnostics.get("path_realizability")
    if not isinstance(path, dict):
        return {}
    answer = path.get("answer_trace")
    return answer if isinstance(answer, dict) else {}


def _reference_row(run_dir: Path) -> dict[str, Any]:
    qa = _read(run_dir / "qa_metrics.json")
    retrieval = _read(run_dir / "retrieval_metrics.json")
    rows = _read_jsonl(run_dir / "retrieval_trace.jsonl")
    answer_traces = [_path_answer_trace(row) for row in rows]
    return {
        "run": run_dir.name,
        "graph_variant": "entity_chunk_reference",
        "renderer_mode": "current_renderer",
        "diagnostic_only": False,
        "num_queries": int(qa.get("num_queries", retrieval.get("num_queries", 0))),
        "gold_projection": float(retrieval.get("mean_context_recall", 0.0)),
        "answer_projected": _mean_bool([trace.get("answer_chunk_in_projected") for trace in answer_traces]),
        "answer_selected": _mean_bool([trace.get("answer_chunk_on_support_tree") for trace in answer_traces]),
        "answer_rendered": _mean_bool([trace.get("answer_chunk_rendered") for trace in answer_traces]),
        "answer_in_context": float(qa.get("mean_answer_coverage", 0.0)),
        "rendered_recall": float(qa.get("mean_context_recall", retrieval.get("mean_context_recall", 0.0))),
        "context_f1": float(qa.get("mean_context_f1", retrieval.get("mean_context_f1", 0.0)) or 0.0),
        "qa_f1": float(qa.get("mean_f1", 0.0)),
        "avg_context_tokens": float(qa.get("avg_context_tokens", retrieval.get("avg_context_tokens", 0.0))),
        "objective_increase_count": 0,
        "triangle_inequality_violation_count": 0,
        "gold_support_sentence_mapping_rate": None,
        "answer_containing_sentence_found_rate": None,
    }


def _sentence_row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "sentence_metrics.json")
    return {
        "run": run_dir.name,
        "graph_variant": metrics.get("graph_variant"),
        "renderer_mode": metrics.get("renderer_mode"),
        "diagnostic_only": bool(metrics.get("diagnostic_only", False)),
        "num_queries": int(metrics.get("num_queries", 0)),
        "gold_projection": float(metrics.get("gold_sentence_projection_rate", 0.0)),
        "answer_projected": float(metrics.get("answer_sentence_projection_rate", 0.0)),
        "answer_selected": float(metrics.get("answer_sentence_selected_rate", 0.0)),
        "answer_rendered": float(metrics.get("answer_sentence_rendered_rate", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context_rate", 0.0)),
        "rendered_recall": float(metrics.get("rendered_recall", 0.0)),
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "objective_increase_count": int(metrics.get("objective_increase_count", 0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "gold_support_sentence_mapping_rate": metrics.get("gold_support_sentence_mapping_rate"),
        "answer_containing_sentence_found_rate": metrics.get("answer_containing_sentence_found_rate"),
    }


def _row(run_dir: Path) -> dict[str, Any]:
    if run_dir.name == REFERENCE_RUN:
        return _reference_row(run_dir)
    return _sentence_row(run_dir)


def _gate(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    if row["run"] == ref["run"]:
        return {"run": row["run"], "decision": "REFERENCE", "blockers": [], "strong": False}
    blockers: list[str] = []
    for field in GATE_FIELDS:
        if row[field] < ref[field] - 1e-12:
            blockers.append(f"{field}_regression")
    if row["avg_context_tokens"] > ref["avg_context_tokens"] * 1.10 + 1e-12:
        blockers.append("context_tokens_over_110pct_reference")
    if row["objective_increase_count"] != 0:
        blockers.append("objective_monotonicity_violation")
    if row["triangle_inequality_violation_count"] != 0:
        blockers.append("triangle_inequality_violation")
    strong = (
        row["qa_f1"] > ref["qa_f1"] + 1e-12
        and row["answer_in_context"] > ref["answer_in_context"] + 1e-12
        and row["avg_context_tokens"] <= ref["avg_context_tokens"] + 1e-12
    )
    decision = "PASS" if not blockers else "STOP"
    if row.get("diagnostic_only"):
        decision = "DIAGNOSTIC_ONLY" if not blockers else "DIAGNOSTIC_ONLY_STOP"
    return {"run": row["run"], "decision": decision, "blockers": blockers, "strong": strong}


def _datasets_from_root(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if (root / REFERENCE_RUN).exists():
        return [root.name]
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def compare(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in _datasets_from_root(root, datasets):
        ds_root = root if (root / REFERENCE_RUN).exists() and root.name == dataset else root / dataset
        if not ds_root.exists():
            continue
        rows = [_row(ds_root / run) for run in RUN_ORDER if (ds_root / run).exists()]
        if not rows:
            continue
        ref = next((row for row in rows if row["run"] == REFERENCE_RUN), rows[0])
        gates = [_gate(row, ref) for row in rows]
        summary[dataset] = {"rows": rows, "gates": gates}
    summary["_final"] = _final_decision(summary)
    return summary


def _final_decision(summary: dict[str, Any]) -> dict[str, Any]:
    datasets = [key for key in summary if not key.startswith("_")]
    if not datasets:
        return {"decision": "STOP", "reason": "no datasets found"}
    runs = set.intersection(
        *[
            {gate["run"] for gate in summary[dataset]["gates"] if gate["decision"] in {"PASS", "REFERENCE"}}
            for dataset in datasets
        ]
    )
    runs.discard(REFERENCE_RUN)
    if not runs:
        return {"decision": "STOP", "reason": "no non-diagnostic variant passed all available datasets"}
    strong_runs = set.intersection(
        *[
            {gate["run"] for gate in summary[dataset]["gates"] if gate.get("strong")}
            for dataset in datasets
        ]
    )
    if strong_runs:
        return {"decision": "ADOPT_SENTENCE_PRIMARY", "strong_runs": sorted(strong_runs)}
    return {"decision": "DIAGNOSTIC_ONLY", "passing_runs": sorted(runs)}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def markdown(summary: dict[str, Any]) -> str:
    branch = _git_value(["branch", "--show-current"])
    commit = _git_value(["rev-parse", "--short", "HEAD"])
    final = summary.get("_final", {})
    lines = [
        "# Sentence Graph Granularity Diagnostic Report",
        "",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Final decision: **{final.get('decision', 'STOP')}**",
        "",
        "## Why This Was Tested",
        "",
        "Previous diagnostics suggested that graph evidence could be projected or path-reachable while answer-bearing surface evidence still failed to render or help QA. This run tests whether making sentences, rather than chunks, the primary PAMAE medoids improves that bottleneck.",
        "",
        "## PAMAE Principle Check",
        "",
        "- Primary objects are sentence nodes.",
        "- Selected medoids are sentence nodes.",
        "- PPR is used only to define query-conditioned sentence mass.",
        "- Shortest-path distance is the PAMAE metric.",
        "- No dense reranking, LLM reranking, answer-aware retrieval, or scalar score mixing is used.",
        "- `entity_sentence_chunk_hier` stores chunk parents for metadata/rendering; parent edges are excluded from the main metric.",
        "",
    ]
    for dataset, payload in summary.items():
        if dataset.startswith("_"):
            continue
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| run | graph | renderer | ans proj | ans rendered | ans ctx | rendered recall | context F1 | QA F1 | ctx tok | map rate | obj inc | tri viol | decision |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        gates = {gate["run"]: gate for gate in payload["gates"]}
        for row in payload["rows"]:
            gate = gates[row["run"]]
            decision = gate["decision"]
            if gate["blockers"]:
                decision += " (" + ", ".join(gate["blockers"]) + ")"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["run"],
                        _fmt(row["graph_variant"]),
                        _fmt(row["renderer_mode"]),
                        _fmt(row["answer_projected"]),
                        _fmt(row["answer_rendered"]),
                        _fmt(row["answer_in_context"]),
                        _fmt(row["rendered_recall"]),
                        _fmt(row["context_f1"]),
                        _fmt(row["qa_f1"]),
                        _fmt(row["avg_context_tokens"]),
                        _fmt(row["gold_support_sentence_mapping_rate"]),
                        _fmt(row["objective_increase_count"]),
                        _fmt(row["triangle_inequality_violation_count"]),
                        decision,
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Adoption Gate Decision",
            "",
            json.dumps(final, ensure_ascii=False, indent=2),
            "",
            "## Next Recommendation",
            "",
            "If sentence projection improves without QA improvement, focus next on context formatting and narrower parent-context rendering. If neither projection nor rendering improves on Hotpot, graph granularity alone is insufficient and the next diagnostic should target non-gold answer-bearing fact proxies.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sentence-primary graph variants.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = compare(Path(args.root), args.datasets or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(summary), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
