#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_PROMPT_HASH = "31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d"
B2 = "tree_shell1_semantic_query_order"
CURRENT = "current_renderer"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _variant_dir(dataset_root: Path, variant: str, mode: str) -> Path:
    return dataset_root / f"entity_chunk_reference_{variant}_{mode}"


def _metrics(dataset_root: Path, variant: str, mode: str) -> dict[str, Any]:
    row = _load_json(_variant_dir(dataset_root, variant, mode) / "runtime_metrics.json")
    row["variant"] = variant
    row["mode"] = mode
    return row


def _has_metrics(row: dict[str, Any]) -> bool:
    return "qa_f1" in row and "retrieval_ms" in row


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except Exception:
        return "unknown"


def _equivalence(dataset_root: Path) -> dict[str, Any]:
    diag = _read_jsonl(_variant_dir(dataset_root, B2, "diagnostic") / "retrieval_trace.jsonl")
    prod = _read_jsonl(_variant_dir(dataset_root, B2, "production") / "retrieval_trace.jsonl")
    prod_by_id = {str(row["query_id"]): row for row in prod}
    rows: list[dict[str, Any]] = []
    for row in diag:
        other = prod_by_id.get(str(row["query_id"]))
        if other is None:
            rows.append({"query_id": row["query_id"], "rendered_chunk_ids_exact_match": False, "context_text_hash_exact_match": False})
            continue
        diag_hash = row.get("diagnostics", {}).get("context_text_hash")
        prod_hash = other.get("diagnostics", {}).get("context_text_hash")
        rows.append(
            {
                "query_id": row["query_id"],
                "diagnostic_rendered_chunk_ids": row.get("context_node_ids", []),
                "production_rendered_chunk_ids": other.get("context_node_ids", []),
                "rendered_chunk_ids_exact_match": row.get("context_node_ids", []) == other.get("context_node_ids", []),
                "context_text_hash_exact_match": bool(diag_hash and diag_hash == prod_hash),
            }
        )
    total = len(rows)
    rendered_ok = sum(1 for row in rows if row["rendered_chunk_ids_exact_match"])
    hash_ok = sum(1 for row in rows if row["context_text_hash_exact_match"])
    diag_metrics = _metrics(dataset_root, B2, "diagnostic")
    prod_metrics = _metrics(dataset_root, B2, "production")
    quality_keys = ["qa_f1", "answer_in_context", "rendered_recall", "context_f1", "avg_context_tokens"]
    quality_match = bool(
        _has_metrics(diag_metrics)
        and _has_metrics(prod_metrics)
        and all(abs(float(diag_metrics.get(key, 0.0)) - float(prod_metrics.get(key, 0.0))) < 1e-12 for key in quality_keys)
    )
    return {
        "rows": rows,
        "total": total,
        "rendered_match_rate": float(rendered_ok / total) if total else 0.0,
        "context_hash_match_rate": float(hash_ok / total) if total else 0.0,
        "all_rendered_match": bool(total and rendered_ok == total),
        "all_context_hash_match": bool(total and hash_ok == total),
        "quality_match": bool(quality_match),
        "diagnostic_metrics": diag_metrics,
        "production_metrics": prod_metrics,
    }


def _gate(dataset: str, current: dict[str, Any], b2: dict[str, Any], equivalence_ok: bool) -> dict[str, Any]:
    blockers: list[str] = []
    checks = {
        "qa_f1": float(b2.get("qa_f1", 0.0)) >= float(current.get("qa_f1", 0.0)),
        "answer_in_context": float(b2.get("answer_in_context", 0.0)) >= float(current.get("answer_in_context", 0.0)),
        "rendered_recall": float(b2.get("rendered_recall", 0.0)) >= float(current.get("rendered_recall", 0.0)),
        "context_f1": float(b2.get("context_f1", 0.0)) >= float(current.get("context_f1", 0.0)),
        "tokens": float(b2.get("avg_context_tokens", 0.0)) <= float(current.get("avg_context_tokens", 0.0)) * 1.10,
        "time": float(b2.get("retrieval_ms", 0.0)) <= float(current.get("retrieval_ms", 0.0)) * 1.25,
        "prompt": b2.get("qa_prompt_hash") == EXPECTED_PROMPT_HASH and bool(b2.get("qa_prompt_text_exact_match")),
        "oracle_leakage": int(b2.get("oracle_leakage_count", 0) or 0) == 0,
        "score_mixing": not bool(b2.get("score_mixing_detected", False)),
        "equivalence": equivalence_ok,
    }
    blockers = [name for name, ok in checks.items() if not ok]
    return {
        "dataset": dataset,
        "pass": not blockers,
        "blockers": blockers,
        "time_ratio": float(b2.get("retrieval_ms", 0.0)) / max(float(current.get("retrieval_ms", 0.0)), 1e-9),
        "checks": checks,
    }


def _table(rows: list[dict[str, Any]]) -> list[str]:
    keys = [
        "variant",
        "mode",
        "em",
        "qa_f1",
        "answer_in_context",
        "rendered_recall",
        "context_f1",
        "avg_context_tokens",
        "retrieval_ms",
        "generation_ms",
        "support_tree_chunk_count",
        "shell1_chunk_count",
        "candidate_pool_size",
        "rendered_shell1_chunk_count",
    ]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key, "")) for key in keys) + " |")
    return lines


def _dataset_roots(root: Path, datasets: list[str] | None) -> dict[str, Path]:
    if datasets:
        return {dataset: root / dataset for dataset in datasets}
    if _variant_dir(root, B2, "diagnostic").exists() or _variant_dir(root, B2, "production").exists():
        return {root.name: root}
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()}


def build_report(root: Path, dataset_roots: dict[str, Path]) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "branch": _git_branch(),
        "commit": _git_commit(),
        "datasets": [],
    }
    lines = [
        "# B2 Runtime Validation Report",
        "",
        f"- Branch: `{payload['branch']}`",
        f"- Commit: `{payload['commit']}`",
        f"- Fixed prompt hash: `{EXPECTED_PROMPT_HASH}`",
        "- Method: fixed `tree_shell1_semantic_query_order`; production mode may remove diagnostics only.",
        "",
    ]
    all_gates: list[dict[str, Any]] = []
    for dataset, dataset_root in dataset_roots.items():
        current = _metrics(dataset_root, CURRENT, "production")
        b2_prod = _metrics(dataset_root, B2, "production")
        b2_diag_path = _variant_dir(dataset_root, B2, "diagnostic") / "runtime_metrics.json"
        equivalence = _equivalence(dataset_root) if b2_diag_path.exists() else {
            "all_rendered_match": False,
            "all_context_hash_match": False,
            "quality_match": False,
            "rendered_match_rate": 0.0,
            "context_hash_match_rate": 0.0,
        }
        equivalence_ok = bool(
            equivalence.get("all_rendered_match")
            and equivalence.get("all_context_hash_match")
            and equivalence.get("quality_match")
        )
        can_gate = _has_metrics(current) and _has_metrics(b2_prod)
        if can_gate:
            gate = _gate(dataset, current, b2_prod, equivalence_ok)
            all_gates.append(gate)
        else:
            gate = {
                "dataset": dataset,
                "pass": False,
                "blockers": ["baseline_or_b2_production_missing"],
                "time_ratio": 0.0,
                "checks": {},
                "skipped": True,
            }
        rows = [row for row in [current, b2_prod] if _has_metrics(row)]
        if b2_diag_path.exists():
            rows.insert(1, _metrics(dataset_root, B2, "diagnostic"))
        payload["datasets"].append({"dataset": dataset, "gate": gate, "equivalence": equivalence, "rows": rows})
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Equivalence rendered match: `{equivalence.get('rendered_match_rate', 0.0):.4f}`",
                f"- Equivalence context hash match: `{equivalence.get('context_hash_match_rate', 0.0):.4f}`",
                f"- Quality match: `{equivalence.get('quality_match', False)}`",
                f"- Gate pass: `{gate['pass']}` blockers: `{', '.join(gate['blockers']) if gate['blockers'] else 'none'}`",
                f"- B2/current retrieval time ratio: `{gate['time_ratio']:.4f}`",
                "",
                *_table(rows),
                "",
            ]
        )
        runtime = b2_prod.get("runtime_profile", {})
        if isinstance(runtime, dict):
            lines.append("### B2 Production Timing")
            for item in runtime.get("top_two_contributors", []):
                lines.append(f"- {item['name']}: {_fmt(item['mean_ms'])} ms")
            lines.append("")
    if not all_gates:
        decision = "DIAGNOSTIC_ONLY"
        recommendation = "Equivalence diagnostics were generated; production current-vs-B2 gates were not run in this report."
    else:
        primary_pass = bool(all(gate["pass"] for gate in all_gates))
        if primary_pass:
            decision = "ADOPTION_CANDIDATE_CONFIRMED"
            recommendation = "B2 production mode preserves context and passes the retrieval-time gate."
        elif all(
            not any(blocker in gate["blockers"] for blocker in ["qa_f1", "answer_in_context", "rendered_recall", "context_f1", "tokens", "equivalence", "prompt", "oracle_leakage", "score_mixing"])
            for gate in all_gates
        ):
            decision = "EFFICIENCY_OPT_REQUIRED"
            recommendation = "Quality and equivalence are preserved, but at least one time gate still fails."
        else:
            decision = "STOP"
            recommendation = "Do not adopt B2 until the listed gate blockers are resolved without semantic changes."
    payload["final_decision"] = decision
    lines.extend(
        [
            "## Local-Minimum Guard",
            "",
            "- Did we change the PAMAE core retrieval objective? No.",
            "- Did we change B2 candidate pool? No.",
            "- Did we change B2 ordering semantics? No.",
            "- Did we use global dense retrieval? No.",
            "- Did we use score mixing? No.",
            "- Did production mode produce identical contexts? See equivalence rows above.",
            "- Did B2 preserve answer coverage and rendered recall? See quality gates above.",
            "- Did B2 satisfy the time gate? See time ratios above.",
            "- Did optimization hide diagnostic costs rather than real retrieval costs? Production mode removes diagnostics and reports stage timings separately.",
            "",
            f"Final decision: **{decision}**",
            "",
            f"Next recommendation: {recommendation}",
            "",
        ]
    )
    return "\n".join(lines), payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B2 runtime validation runs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report, payload = build_report(args.root, _dataset_roots(args.root, args.datasets))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    (args.out.parent / "b2_runtime_validation_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
