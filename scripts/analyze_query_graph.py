#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.graph.graph_distance import (
    average_edge_counts,
    graph_diagnostics,
    graph_shortest_path_distance,
)
from pamae_rag.graph.query_graph import build_minimal_query_graph
from pamae_rag.graph.universe import select_universe_by_mass


def _gold_pair_stats(example, nodes, graph, sp: np.ndarray, disconnected_distance: float) -> tuple[float | None, bool | None]:
    id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}
    gold = [id_to_idx[node_id] for node_id in sorted(example.gold_node_ids) if node_id in id_to_idx]
    if len(gold) < 2:
        return None, None
    values: list[float] = []
    all_connected = True
    for pos, i in enumerate(gold):
        for j in gold[pos + 1 :]:
            value = float(sp[int(i), int(j)])
            if value >= disconnected_distance - 1e-12:
                all_connected = False
            else:
                values.append(value)
    return (mean(values) if values else None), all_connected


def analyze_query_graphs(input_path: str | Path, config_path: str | Path, limit: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit or cfg.experiment.limit_queries)
    node_counts: list[int] = []
    edge_counts: list[int] = []
    edge_counts_by_type: list[dict[str, int]] = []
    component_counts: list[int] = []
    largest_ratios: list[float] = []
    disconnected_rates: list[float] = []
    gold_connected: list[bool] = []
    gold_distances: list[float] = []
    problematic: list[dict[str, Any]] = []

    for example in examples:
        nodes = select_universe_by_mass(
            example.nodes,
            max_nodes=cfg.universe.max_nodes,
            min_relevance_mass=cfg.universe.min_relevance_mass,
        )
        graph = build_minimal_query_graph(
            nodes,
            example.query,
            edge_lengths=dict(cfg.pamae.graph.edge_lengths.__dict__),
            max_edges_per_node=cfg.pamae.graph.max_edges_per_node,
        )
        sp = graph_shortest_path_distance(graph, cfg.pamae.graph.disconnected_distance)
        diag = graph_diagnostics(graph, sp, cfg.pamae.graph.disconnected_distance)
        gold_avg_distance, gold_is_connected = _gold_pair_stats(
            example, nodes, graph, sp, cfg.pamae.graph.disconnected_distance
        )

        node_counts.append(len(nodes))
        edge_counts.append(graph.num_edges)
        edge_counts_by_type.append(graph.edge_counts_by_type)
        component_counts.append(int(diag["num_connected_components"]))
        largest_ratios.append(float(diag["largest_component_ratio"]))
        disconnected_rates.append(float(diag["disconnected_pair_rate"]))
        if gold_is_connected is not None:
            gold_connected.append(bool(gold_is_connected))
        if gold_avg_distance is not None:
            gold_distances.append(float(gold_avg_distance))
        if float(diag["disconnected_pair_rate"]) >= 0.8 and len(problematic) < 10:
            problematic.append(
                {
                    "query_id": example.query_id,
                    "disconnected_pair_rate": float(diag["disconnected_pair_rate"]),
                    "num_edges": graph.num_edges,
                    "largest_component_ratio": float(diag["largest_component_ratio"]),
                }
            )

    return {
        "num_queries": len(examples),
        "avg_num_nodes": mean(node_counts) if node_counts else 0.0,
        "avg_num_edges": mean(edge_counts) if edge_counts else 0.0,
        "avg_edge_counts_by_type": average_edge_counts(edge_counts_by_type),
        "avg_num_connected_components": mean(component_counts) if component_counts else 0.0,
        "avg_largest_component_ratio": mean(largest_ratios) if largest_ratios else 0.0,
        "avg_disconnected_pair_rate": mean(disconnected_rates) if disconnected_rates else 0.0,
        "gold_support_connected_rate": mean(gold_connected) if gold_connected else None,
        "gold_support_avg_shortest_path_distance": mean(gold_distances) if gold_distances else None,
        "problematic_queries": problematic,
    }


def _markdown(metrics: dict[str, Any], input_path: str, config_path: str) -> str:
    lines = [
        "# Query Graph Diagnostic",
        "",
        f"- input: `{input_path}`",
        f"- config: `{config_path}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if key in {"problematic_queries", "avg_edge_counts_by_type"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Average Edge Counts", "", "| edge_type | count |", "| --- | ---: |"])
    for key, value in metrics["avg_edge_counts_by_type"].items():
        lines.append(f"| {key} | {value} |")
    if metrics["problematic_queries"]:
        lines.extend(["", "## Problematic Queries", ""])
        for row in metrics["problematic_queries"]:
            lines.append(
                f"- `{row['query_id']}`: disconnected_pair_rate={row['disconnected_pair_rate']}, "
                f"num_edges={row['num_edges']}, largest_component_ratio={row['largest_component_ratio']}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    metrics = analyze_query_graphs(args.input, args.config, args.limit)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(metrics, args.input, args.config), encoding="utf-8")


if __name__ == "__main__":
    main()
