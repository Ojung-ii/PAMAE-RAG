#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_mean(metrics: dict[str, Any], stage: str, key: str) -> float | None:
    value = metrics.get("stage_diagnostics", {}).get(stage, {}).get("mean", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _failure_count(run_dir: Path, failure_type: str) -> int:
    path = run_dir / "failure_taxonomy.json"
    if not path.exists():
        return 0
    return int(_read(path).get("failure_type_counts", {}).get(failure_type, 0))


def _selected_basin_hit_rate(run_dir: Path) -> float:
    path = run_dir / "failure_taxonomy.json"
    if not path.exists():
        return 0.0
    rows = _read(path).get("rows", [])
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get("selected_basin_hit")) / len(rows)


def _row(run_dir: Path, oracle_f1: float | None) -> dict[str, Any]:
    qa = _read(run_dir / "qa_metrics.json")
    retrieval_path = run_dir / "retrieval_metrics.json"
    retrieval = _read(retrieval_path) if retrieval_path.exists() else {}
    f1 = float(qa.get("mean_f1", 0.0))
    return {
        "run": run_dir.name,
        "F1": f1,
        "oracle_gap": None if oracle_f1 is None else oracle_f1 - f1,
        "TypeB": _failure_count(run_dir, "selection_miss"),
        "selected_basin_hit": _selected_basin_hit_rate(run_dir),
        "rendered_recall": _stage_mean(qa, "context_rendering", "rendered_recall")
        or float(retrieval.get("mean_context_recall", 0.0)),
        "context_f1": float(qa.get("mean_context_f1", retrieval.get("mean_context_f1", 0.0))),
        "answer_in_context": float(qa.get("mean_answer_coverage", 0.0)),
        "retrieval_ms": float(qa.get("avg_retrieval_ms", 0.0)),
        "generation_ms": float(qa.get("avg_generation_ms", 0.0)),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _gate(row: dict[str, Any], ref: dict[str, Any], oracle_ok: bool) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if not oracle_ok:
        blockers.append("oracle_dominance_invalid")
    if row["F1"] < ref["F1"] - 1e-12:
        blockers.append("f1_regression")
    if row["oracle_gap"] is not None and ref["oracle_gap"] is not None:
        if row["oracle_gap"] > ref["oracle_gap"] + 1e-12:
            blockers.append("oracle_gap_regression")
    if row["TypeB"] >= ref["TypeB"]:
        blockers.append("type_b_not_reduced")
    if row["selected_basin_hit"] < ref["selected_basin_hit"] - 1e-12:
        blockers.append("selected_basin_hit_regression")
    if row["rendered_recall"] < ref["rendered_recall"] - 1e-12:
        blockers.append("rendered_recall_regression")
    if row["context_f1"] < ref["context_f1"] - 1e-12:
        blockers.append("context_f1_regression")
    if row["answer_in_context"] < ref["answer_in_context"] - 1e-12:
        blockers.append("answer_in_context_regression")
    return ("PASS" if not blockers else "STOP", blockers)


def compare(run_dirs: list[Path], oracle_path: Path | None) -> dict[str, Any]:
    oracle = _read(oracle_path) if oracle_path and oracle_path.exists() else {}
    oracle_f1 = oracle.get("gold_support_f1")
    oracle_f1 = float(oracle_f1) if isinstance(oracle_f1, (int, float)) else None
    oracle_ok = bool(oracle.get("oracle_dominance_valid", True))
    rows = [_row(run_dir, oracle_f1) for run_dir in run_dirs]
    if not rows:
        raise ValueError("At least one run directory is required")
    ref = rows[0]
    gates = []
    for row in rows:
        if row is ref:
            gates.append({"run": row["run"], "decision": "REFERENCE", "blockers": []})
            continue
        decision, blockers = _gate(row, ref, oracle_ok)
        gates.append({"run": row["run"], "decision": decision, "blockers": blockers})
    return {
        "oracle": oracle,
        "rows": rows,
        "risk_gates": gates,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Basin Selection Comparison",
        "",
        "| run | F1 | oracle_gap | TypeB | selected_basin_hit | rendered_recall | context_f1 | answer_in_context | retrieval_ms | generation_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    _fmt(row["F1"]),
                    _fmt(row["oracle_gap"]),
                    _fmt(row["TypeB"]),
                    _fmt(row["selected_basin_hit"]),
                    _fmt(row["rendered_recall"]),
                    _fmt(row["context_f1"]),
                    _fmt(row["answer_in_context"]),
                    _fmt(row["retrieval_ms"]),
                    _fmt(row["generation_ms"]),
                ]
            )
            + " |"
        )
    oracle = summary.get("oracle", {})
    lines.extend(
        [
            "",
            "## Oracle Diagnostics",
            "",
            f"- gold_support_f1: `{_fmt(oracle.get('gold_support_f1'))}`",
            f"- answer_containing_f1: `{_fmt(oracle.get('answer_containing_f1'))}`",
            f"- answer_copy_f1: `{_fmt(oracle.get('answer_copy_f1'))}`",
            f"- oracle_dominance_valid: `{str(oracle.get('oracle_dominance_valid')).lower()}`",
            f"- diagnosis: `{oracle.get('diagnosis', 'n/a')}`",
            "",
            "## Risk Gates",
            "",
        ]
    )
    for gate in summary["risk_gates"]:
        blockers = ", ".join(gate["blockers"]) if gate["blockers"] else "none"
        lines.append(f"- `{gate['run']}`: `{gate['decision']}` blockers=`{blockers}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare basin-preserving selection smoke runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--oracle", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = compare([Path(path) for path in args.runs], Path(args.oracle) if args.oracle else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(summary), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
