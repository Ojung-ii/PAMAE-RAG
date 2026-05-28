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
    "entity_chunk_reference_local_sentence_medoid",
    "entity_chunk_reference_fact_mediated_sentence",
    "entity_chunk_reference_selected_chunk_answer_sentence_oracle",
    "entity_chunk_reference_selected_chunk_gold_sentence_oracle",
)
GATE_FIELDS = (
    "qa_f1",
    "answer_in_context",
    "rendered_recall",
    "context_f1",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(run_dir: Path) -> dict[str, Any]:
    metrics = _read(run_dir / "local_surface_metrics.json")
    return {
        "run": run_dir.name,
        "graph_variant": metrics.get("graph_variant", "entity_chunk_reference"),
        "renderer_mode": metrics.get("renderer_mode"),
        "oracle_renderer": bool(metrics.get("oracle_renderer", False)),
        "answer_sentence_available": float(metrics.get("answer_sentence_available_in_selected_chunks", 0.0)),
        "gold_sentence_available": float(metrics.get("gold_sentence_available_in_selected_chunks", 0.0)),
        "answer_sentence_rendered": float(metrics.get("answer_sentence_rendered", 0.0)),
        "gold_sentence_rendered": float(metrics.get("gold_sentence_rendered", 0.0)),
        "answer_in_context": float(metrics.get("answer_in_context", 0.0)),
        "rendered_recall": float(metrics.get("rendered_recall", 0.0)),
        "context_f1": float(metrics.get("context_f1", 0.0)),
        "qa_f1": float(metrics.get("qa_f1", 0.0)),
        "oracle_gap_answer_containing": float(metrics.get("oracle_gap_answer_containing", 0.0)),
        "avg_context_tokens": float(metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0)),
        "triangle_inequality_violation_count": int(metrics.get("triangle_inequality_violation_count", 0)),
        "local_objective_invalid_count": int(metrics.get("local_objective_invalid_count", 0)),
        "answer_rendered_given_available": float(metrics.get("answer_rendered_given_available", 0.0)),
        "gold_rendered_given_available": float(metrics.get("gold_rendered_given_available", 0.0)),
        "qa_success_given_answer_rendered": float(metrics.get("qa_success_given_answer_rendered", 0.0)),
    }


def _datasets_from_root(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if (root / REFERENCE_RUN).exists():
        return [root.name]
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _gate(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    if row["run"] == ref["run"]:
        return {"run": row["run"], "decision": "REFERENCE", "blockers": [], "strong": False}
    blockers: list[str] = []
    if row["oracle_renderer"]:
        blockers.append("oracle_renderer_excluded")
    for field in GATE_FIELDS:
        if row[field] < ref[field] - 1e-12:
            blockers.append(f"{field}_regression")
    if row["avg_context_tokens"] > ref["avg_context_tokens"] + 1e-12:
        blockers.append("context_tokens_increase")
    if row["triangle_inequality_violation_count"] != 0:
        blockers.append("triangle_inequality_violation")
    if row["local_objective_invalid_count"] != 0:
        blockers.append("local_objective_invalid")
    if row["qa_f1"] > ref["qa_f1"] + 1e-12 and row["answer_in_context"] < ref["answer_in_context"] - 1e-12:
        blockers.append("qa_improved_but_answer_in_context_decreased")
    if row["qa_f1"] > ref["qa_f1"] + 1e-12 and row["rendered_recall"] < ref["rendered_recall"] - 1e-12:
        blockers.append("qa_improved_but_rendered_recall_decreased")
    strong = (
        not row["oracle_renderer"]
        and row["qa_f1"] > ref["qa_f1"] + 1e-12
        and row["answer_in_context"] > ref["answer_in_context"] + 1e-12
        and row["avg_context_tokens"] < ref["avg_context_tokens"] - 1e-12
    )
    return {
        "run": row["run"],
        "decision": "PASS" if not blockers else "STOP",
        "blockers": blockers,
        "strong": strong,
    }


def compare(root: Path, datasets: list[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in _datasets_from_root(root, datasets):
        ds_root = root if (root / REFERENCE_RUN).exists() and root.name == dataset else root / dataset
        rows = [_row(ds_root / run) for run in RUN_ORDER if (ds_root / run / "local_surface_metrics.json").exists()]
        if not rows:
            continue
        ref = next(row for row in rows if row["run"] == REFERENCE_RUN)
        answer_oracle = next((row for row in rows if row["renderer_mode"] == "selected_chunk_answer_sentence_oracle"), None)
        if answer_oracle is not None:
            for row in rows:
                if not row["oracle_renderer"]:
                    row["oracle_gap_answer_containing"] = max(
                        0.0,
                        answer_oracle["answer_in_context"] - row["answer_in_context"],
                    )
        gates = [_gate(row, ref) for row in rows]
        summary[dataset] = {"rows": rows, "gates": gates}
    summary["_final"] = _final_decision(summary)
    return summary


def _final_decision(summary: dict[str, Any]) -> dict[str, Any]:
    datasets = [key for key in summary if not key.startswith("_")]
    if not datasets:
        return {"decision": "STOP", "reason": "no datasets found"}
    passing = set.intersection(
        *[
            {
                gate["run"]
                for gate in summary[dataset]["gates"]
                if gate["decision"] in {"PASS", "REFERENCE"}
            }
            for dataset in datasets
        ]
    )
    passing.discard(REFERENCE_RUN)
    if not passing:
        return {"decision": "STOP", "reason": "no non-oracle local renderer passed all available datasets"}
    strong = set.intersection(
        *[
            {gate["run"] for gate in summary[dataset]["gates"] if gate.get("strong")}
            for dataset in datasets
        ]
    )
    if strong:
        return {"decision": "ADOPT_LOCAL_SURFACE_RENDERER", "strong_runs": sorted(strong)}
    return {"decision": "DIAGNOSTIC_ONLY", "passing_runs": sorted(passing)}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _analysis(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = [
        "## Previous Sentence-Primary STOP Summary",
        "",
        "The previous sentence-primary global graph round stopped because neither `entity_sentence` nor `entity_sentence_chunk_hier` beat the entity--chunk reference on answer projection, answer rendering, answer-in-context, rendered recall, or QA F1 across 2Wiki and Hotpot.",
        "",
        "## Reason For Returning To Chunk Backbone",
        "",
        "This round keeps the stronger entity--chunk retrieval backbone fixed and tests whether answer-bearing sentence surfaces can be selected inside the already selected chunks.",
        "",
        "## PAMAE Principle Check",
        "",
        "- Global retrieval object remains the chunk.",
        "- Local rendering objects are sentences/fact-grounded sentences inside selected chunks.",
        "- Local medoids use graph shortest-path distance with PPR only as sentence mass.",
        "- Fact-mediated rendering uses deterministic graph closure, not scalar score mixing.",
        "- Oracle renderers use answer/gold labels only for diagnostics and are excluded from adoption gates.",
        "",
        "## Selected-Chunk Answer Surface Availability",
        "",
    ]
    for dataset, payload in summary.items():
        if dataset.startswith("_"):
            continue
        ref = next(row for row in payload["rows"] if row["run"] == REFERENCE_RUN)
        lines.append(
            f"- `{dataset}` selected chunks contain answer sentences in {_fmt(ref['answer_sentence_available'])} of queries "
            f"and gold support sentences in {_fmt(ref['gold_sentence_available'])}."
        )
    lines.extend(["", "## Local-Minimum Guard Answers", ""])
    final = summary.get("_final", {})
    lines.extend(
        [
            "- Did this preserve the PAMAE principle? Yes: the global retrieval object stayed chunk-level and local sentence selection used graph distance.",
            "- Did it avoid scalar score mixing? Yes.",
            "- Did it improve both answer-bearing recovery and QA? See tables; adoption requires both datasets and all non-oracle gates.",
            "- Did it reduce or increase context tokens? See `ctx tok`; token increases block adoption.",
            "- Did it reveal selected chunks already contain answer sentences? See availability rates above.",
            "- Did it reveal non-gold local rendering cannot identify those sentences? Compare non-oracle recovery to answer oracle gaps.",
            f"- Final decision: **{final.get('decision', 'STOP')}**.",
        ]
    )
    return lines


def markdown(summary: dict[str, Any]) -> str:
    branch = _git_value(["branch", "--show-current"])
    commit = _git_value(["rev-parse", "--short", "HEAD"])
    final = summary.get("_final", {})
    lines = [
        "# Local Surface Diagnostic Report",
        "",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Final decision: **{final.get('decision', 'STOP')}**",
        "",
        *_analysis(summary),
        "",
    ]
    for dataset, payload in summary.items():
        if dataset.startswith("_"):
            continue
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| run | renderer_mode | answer_sentence_available_in_selected_chunks | gold_sentence_available_in_selected_chunks | answer_sentence_rendered | gold_sentence_rendered | answer_in_context | rendered_recall | context_f1 | qa_f1 | oracle_gap_answer_containing | avg_context_tokens | triangle_inequality_violation_count | local_objective_invalid_count | answer_rendered_given_available | gold_rendered_given_available | qa_success_given_answer_rendered | decision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                        row["renderer_mode"],
                        _fmt(row["answer_sentence_available"]),
                        _fmt(row["gold_sentence_available"]),
                        _fmt(row["answer_sentence_rendered"]),
                        _fmt(row["gold_sentence_rendered"]),
                        _fmt(row["answer_in_context"]),
                        _fmt(row["rendered_recall"]),
                        _fmt(row["context_f1"]),
                        _fmt(row["qa_f1"]),
                        _fmt(row["oracle_gap_answer_containing"]),
                        _fmt(row["avg_context_tokens"]),
                        _fmt(row["triangle_inequality_violation_count"]),
                        _fmt(row["local_objective_invalid_count"]),
                        _fmt(row["answer_rendered_given_available"]),
                        _fmt(row["gold_rendered_given_available"]),
                        _fmt(row["qa_success_given_answer_rendered"]),
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
            "If selected chunks contain answer sentences but non-oracle local renderers miss them while the answer oracle is strong, the next step is a principled non-gold answer-bearing proxy rather than another graph backbone change.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk-backbone local surface runs.")
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
