#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "dataset",
    "variant",
    "num_queries",
    "avg_num_nodes",
    "avg_num_edges",
    "avg_symbolic_edges",
    "avg_backbone_edges",
    "avg_degree",
    "max_degree_mean",
    "avg_num_connected_components",
    "avg_largest_component_ratio",
    "avg_connected_pair_rate",
    "avg_disconnected_pair_rate",
    "gold_support_connected_rate",
    "gold_support_same_component_rate",
    "gold_support_avg_shortest_path_distance",
    "backbone_missing_embedding_count",
    "avg_edge_counts_by_type",
    "avg_edge_length_by_type",
]


def _json_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def summarize(runs_root: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_root)
    dataset = root.parent.name if root.name == "dq_connectivity" else root.name
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/graph_diagnostics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {column: metrics.get(column) for column in SUMMARY_COLUMNS}
        row["dataset"] = dataset
        row["variant"] = path.parent.name
        rows.append(row)
    return rows


def write_summary(runs_root: str | Path, output_csv: str | Path, output_md: str | Path) -> None:
    rows = summarize(runs_root)
    csv_path = Path(output_csv)
    md_path = Path(output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_cell(row.get(column)) for column in SUMMARY_COLUMNS})

    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_json_cell(row.get(column)) for column in SUMMARY_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PAMAE query-graph diagnostics")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    write_summary(args.runs_root, args.output_csv, args.output_md)


if __name__ == "__main__":
    main()
