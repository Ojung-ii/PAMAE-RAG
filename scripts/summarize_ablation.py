from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "dataset",
    "variant",
    "retrieval_variant",
    "renderer",
    "relevance_mode",
    "k",
    "k_max",
    "max_context_nodes",
    "max_context_tokens",
    "mean_context_recall",
    "mean_context_precision",
    "mean_context_f1",
    "mean_context_hit",
    "mean_anchor_recall",
    "mean_anchor_precision",
    "mean_anchor_f1",
    "mean_anchor_hit",
    "avg_context_size",
    "avg_context_tokens",
    "mean_context_recall_per_node",
    "mean_context_recall_per_1k_tokens",
    "avg_latency_ms",
    "objective_before_refinement_mean",
    "objective_after_refinement_mean",
    "refinement_accept_rate",
    "objective_support_spearman",
    "context_node_budget_satisfied_rate",
    "context_token_budget_satisfied_rate",
    "missing_prediction_count",
    "missing_anchor_key_count",
    "anchor_non_empty_ratio",
]


def _row(dataset: str, variant: str, metrics: dict[str, Any]) -> dict[str, Any]:
    out = {column: metrics.get(column) for column in SUMMARY_COLUMNS}
    out["dataset"] = metrics.get("dataset", dataset)
    out["variant"] = metrics.get("variant", variant)
    return out


def summarize(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    dataset = dataset_path.name
    rows = []
    for metrics_path in sorted(dataset_path.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_row(dataset, metrics_path.parent.name, metrics))
    return rows


def write_summary(
    dataset_dir: str | Path,
    *,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
) -> None:
    dataset_path = Path(dataset_dir)
    rows = summarize(dataset_path)
    csv_path = Path(output_csv) if output_csv else dataset_path / "summary.csv"
    md_path = Path(output_md) if output_md else dataset_path / "summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(col) is None else str(row.get(col)) for col in SUMMARY_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PAMAE-RAG ablation metrics")
    parser.add_argument("dataset_dir", nargs="?", help="Directory such as data/runs/popqa")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--runs-root", default="data/runs")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = str(Path(args.runs_root) / args.dataset) if args.dataset else args.runs_root
    write_summary(dataset_dir, output_csv=args.output_csv, output_md=args.output_md)


if __name__ == "__main__":
    main()
