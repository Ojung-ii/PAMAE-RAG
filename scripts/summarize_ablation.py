from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "dataset",
    "variant",
    "relevance_mode",
    "renderer",
    "retrieval_variant",
    "k",
    "k_max",
    "mean_context_recall",
    "mean_context_hit",
    "mean_anchor_recall",
    "mean_anchor_hit",
    "avg_context_size",
    "avg_latency_ms",
    "objective_before_refinement_mean",
    "objective_after_refinement_mean",
    "refinement_accept_rate",
    "missing_anchor_key_count",
    "missing_prediction_count",
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


def write_summary(dataset_dir: str | Path) -> None:
    dataset_path = Path(dataset_dir)
    rows = summarize(dataset_path)
    csv_path = dataset_path / "summary.csv"
    md_path = dataset_path / "summary.md"

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        if args.dataset is None:
            raise SystemExit("Provide dataset_dir or --dataset")
        dataset_dir = str(Path(args.runs_root) / args.dataset)
    write_summary(dataset_dir)


if __name__ == "__main__":
    main()
