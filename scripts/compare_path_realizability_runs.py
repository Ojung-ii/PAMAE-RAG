#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_ORDER = (
    "current_content_current_renderer",
    "current_content_gold_path_oracle_renderer",
    "current_content_path_neighborhood_renderer",
    "basin_preserving_selection_current_renderer",
    "basin_preserving_selection_gold_path_oracle_renderer",
    "basin_preserving_selection_path_neighborhood_renderer",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_mean(metrics: dict[str, Any], stage: str, key: str) -> float:
    value = metrics.get("stage_diagnostics", {}).get(stage, {}).get("mean", {}).get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {
        str(key): int(value)
        for key, value in _read(path).get("representative_failure_counts", {}).items()
        if isinstance(value, int)
    }


def _row(run_dir: Path) -> dict[str, Any]:
    qa = _read(run_dir / "qa_metrics.json")
    retrieval = _read(run_dir / "retrieval_metrics.json")
    taxonomy = _read(run_dir / "representative_taxonomy.json")
    oracle = _read(run_dir / "oracle_diagnostics.json")
    f1 = float(qa.get("mean_f1", 0.0))
    oracle_f1 = oracle.get("gold_support_f1")
    oracle_gap = None
    if isinstance(oracle_f1, (int, float)) and not isinstance(oracle_f1, bool):
        oracle_gap = float(oracle_f1) - f1
    counts = _counts(run_dir / "representative_taxonomy.json")
    return {
        "run": run_dir.name,
        "F1": f1,
        "EM": float(qa.get("mean_exact_match", 0.0)),
        "oracle_gap": oracle_gap,
        "rendered_recall": _stage_mean(qa, "context_rendering", "rendered_recall")
        or float(retrieval.get("mean_context_recall", 0.0)),
        "context_f1": float(qa.get("mean_context_f1", retrieval.get("mean_context_f1", 0.0))),
        "answer_in_context": float(qa.get("mean_answer_coverage", 0.0)),
        "avg_context_tokens": float(qa.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(qa.get("avg_retrieval_ms", 0.0)),
        "generation_ms": float(qa.get("avg_generation_ms", 0.0)),
        "A": counts.get("A_projection_miss", 0),
        "B": counts.get("B_basin_partition_miss", 0),
        "C": counts.get("C_representative_mismatch", 0),
        "D": counts.get("D_renderer_sparsity", 0),
        "E": counts.get("E_budget_cutoff", 0),
        "F": counts.get("F_generator_fail", 0),
        "G": counts.get("G_success", 0),
        "mean_d_medoid_gold": taxonomy.get("mean_d_medoid_gold"),
        "mean_gold_distance_percentile_within_basin": taxonomy.get(
            "mean_gold_distance_percentile_within_basin"
        ),
        "gold_on_support_tree_rate": float(taxonomy.get("gold_on_support_tree_rate", 0.0)),
        "gold_path_exists_but_not_rendered_rate": float(
            taxonomy.get("gold_path_exists_but_not_rendered_rate", 0.0)
        ),
        "answer_chunk_projected_but_not_rendered_rate": float(
            taxonomy.get("answer_chunk_projected_but_not_rendered_rate", 0.0)
        ),
        "oracle": oracle,
    }


def _gate(row: dict[str, Any], ref: dict[str, Any]) -> tuple[str, list[str]]:
    if row["run"] == ref["run"]:
        return "REFERENCE", []
    if "gold_path_oracle" in row["run"]:
        return "DIAGNOSTIC_ONLY", []
    if "path_neighborhood" not in row["run"]:
        return "DIAGNOSTIC_ONLY", []
    blockers: list[str] = []
    if not row["oracle"].get("oracle_dominance_valid", True):
        blockers.append("oracle_dominance_invalid")
    if row["F1"] < ref["F1"] - 1e-12:
        blockers.append("f1_regression")
    if row["oracle_gap"] is not None and ref["oracle_gap"] is not None:
        if row["oracle_gap"] > ref["oracle_gap"] + 1e-12:
            blockers.append("oracle_gap_regression")
    if row["rendered_recall"] < ref["rendered_recall"] - 1e-12:
        blockers.append("rendered_recall_regression")
    if row["answer_in_context"] < ref["answer_in_context"] - 1e-12:
        blockers.append("answer_in_context_regression")
    if row["context_f1"] < ref["context_f1"] - 1e-12:
        blockers.append("context_f1_regression")
    if row["C"] > ref["C"]:
        blockers.append("representative_mismatch_increase")
    if row["D"] >= ref["D"]:
        blockers.append("renderer_sparsity_not_reduced")
    return ("PASS" if not blockers else "STOP", blockers)


def compare(root: Path, datasets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in datasets:
        ds_root = root / dataset
        rows = [_row(ds_root / name) for name in RUN_ORDER if (ds_root / name).exists()]
        if not rows:
            continue
        ref = next((row for row in rows if row["run"] == "current_content_current_renderer"), rows[0])
        gates = []
        for row in rows:
            decision, blockers = _gate(row, ref)
            gates.append({"run": row["run"], "decision": decision, "blockers": blockers})
        out[dataset] = {"rows": rows, "risk_gates": gates}
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# Path Realizability Comparison", ""]
    for dataset, payload in summary.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| run | F1 | oracle_gap | rendered_recall | context_f1 | answer_in_context | C | D | E | F | G | mean_d_medoid_gold | support_tree_rate | retrieval_ms | decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        gates = {gate["run"]: gate for gate in payload["risk_gates"]}
        for row in payload["rows"]:
            gate = gates.get(row["run"], {"decision": "n/a", "blockers": []})
            decision = gate["decision"]
            if gate["blockers"]:
                decision += " (" + ", ".join(gate["blockers"]) + ")"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["run"],
                        _fmt(row["F1"]),
                        _fmt(row["oracle_gap"]),
                        _fmt(row["rendered_recall"]),
                        _fmt(row["context_f1"]),
                        _fmt(row["answer_in_context"]),
                        _fmt(row["C"]),
                        _fmt(row["D"]),
                        _fmt(row["E"]),
                        _fmt(row["F"]),
                        _fmt(row["G"]),
                        _fmt(row["mean_d_medoid_gold"]),
                        _fmt(row["gold_on_support_tree_rate"]),
                        _fmt(row["retrieval_ms"]),
                        decision,
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare path-realizability smoke runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = compare(Path(args.root), args.datasets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(summary), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

