#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.graph.distances import build_distance_matrix
from pamae_rag.graph.graph_distance import (
    average_edge_counts,
    average_edge_lengths,
    connected_components,
)
from pamae_rag.graph.query_graph import build_minimal_query_graph
from pamae_rag.graph.universe import select_universe_by_mass


def _semantic_distance_matrix(nodes, metric: str) -> np.ndarray | None:
    try:
        embeddings = np.vstack([np.asarray(node.embedding, dtype=np.float64) for node in nodes])
    except (TypeError, ValueError):
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] != len(nodes) or not np.isfinite(embeddings).all():
        return None
    return build_distance_matrix(embeddings, metric=metric)


def _edge_length_means(graph) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for edge in graph.edges:
        totals[edge.edge_type] = totals.get(edge.edge_type, 0.0) + float(edge.length)
        counts[edge.edge_type] = counts.get(edge.edge_type, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def _adjacency(graph) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(graph.num_nodes)]
    for edge in graph.edges:
        adj[edge.source].append((edge.target, float(edge.length)))
        adj[edge.target].append((edge.source, float(edge.length)))
    for row in adj:
        row.sort(key=lambda item: (item[0], item[1]))
    return adj


def _shortest_paths_from(source: int, adj: list[list[tuple[int, float]]]) -> list[float]:
    dist = [float("inf")] * len(adj)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        value, node = heapq.heappop(heap)
        if value > dist[node]:
            continue
        for nxt, length in adj[node]:
            candidate = value + length
            if candidate < dist[nxt]:
                dist[nxt] = candidate
                heapq.heappush(heap, (candidate, nxt))
    return dist


def _fast_graph_diagnostics(graph, disconnected_distance: float) -> dict[str, Any]:
    components = connected_components(graph)
    n = max(graph.num_nodes, 1)
    offdiag = graph.num_nodes * max(graph.num_nodes - 1, 0)
    connected_pairs = sum(len(component) * max(len(component) - 1, 0) for component in components)
    connected_rate = connected_pairs / max(offdiag, 1)
    degrees = [0 for _ in range(graph.num_nodes)]
    for edge in graph.edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    largest = max((len(component) for component in components), default=0)
    return {
        "num_connected_components": len(components),
        "largest_component_ratio": largest / n,
        "connected_pair_rate": connected_rate,
        "disconnected_pair_rate": 1.0 - connected_rate,
        "avg_degree": float(mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees, default=0)),
        "graph_disconnected_distance": float(disconnected_distance),
        "backbone_missing_embedding_count": int(graph.backbone_missing_embedding_count),
    }


def _gold_pair_stats(example, nodes, graph, disconnected_distance: float) -> dict[str, Any]:
    id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}
    gold = [id_to_idx[node_id] for node_id in sorted(example.gold_node_ids) if node_id in id_to_idx]
    if len(gold) < 2:
        return {"pairs": 0, "connected_pairs": 0, "distances": [], "same_component": None}
    components = connected_components(graph)
    component_by_node = {
        node: component_idx
        for component_idx, component in enumerate(components)
        for node in component
    }
    values: list[float] = []
    pair_count = 0
    connected_count = 0
    adj = _adjacency(graph)
    for pos, i in enumerate(gold):
        dist = _shortest_paths_from(int(i), adj)
        for j in gold[pos + 1 :]:
            pair_count += 1
            value = float(dist[int(j)])
            if value < float("inf"):
                connected_count += 1
                values.append(value)
    component_ids = {component_by_node.get(idx) for idx in gold}
    return {
        "pairs": pair_count,
        "connected_pairs": connected_count,
        "distances": values,
        "same_component": len(component_ids) == 1,
    }


def analyze_query_graphs(input_path: str | Path, config_path: str | Path, limit: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit or cfg.experiment.limit_queries)
    node_counts: list[int] = []
    edge_counts: list[int] = []
    edge_counts_by_type: list[dict[str, int]] = []
    edge_lengths_by_type: list[dict[str, float]] = []
    degrees: list[float] = []
    max_degrees: list[int] = []
    component_counts: list[int] = []
    largest_ratios: list[float] = []
    connected_rates: list[float] = []
    disconnected_rates: list[float] = []
    gold_connected_pairs = 0
    gold_total_pairs = 0
    gold_same_component: list[bool] = []
    gold_distances: list[float] = []
    missing_embeddings: list[int] = []
    problematic: list[dict[str, Any]] = []

    for example in examples:
        nodes = select_universe_by_mass(
            example.nodes,
            max_nodes=cfg.universe.max_nodes,
            min_relevance_mass=cfg.universe.min_relevance_mass,
        )
        semantic = _semantic_distance_matrix(nodes, cfg.distance.metric)
        graph = build_minimal_query_graph(
            nodes,
            example.query,
            edge_lengths=dict(cfg.pamae.graph.edge_lengths.__dict__),
            max_edges_per_node=cfg.pamae.graph.max_edges_per_node,
            semantic_distance_matrix=semantic,
            backbone_config=cfg.pamae.graph.backbone,
        )
        diag = _fast_graph_diagnostics(graph, cfg.pamae.graph.disconnected_distance)
        gold_stats = _gold_pair_stats(example, nodes, graph, cfg.pamae.graph.disconnected_distance)

        node_counts.append(len(nodes))
        edge_counts.append(graph.num_edges)
        edge_counts_by_type.append(graph.edge_counts_by_type)
        edge_lengths_by_type.append(_edge_length_means(graph))
        degrees.append(float(diag["avg_degree"]))
        max_degrees.append(int(diag["max_degree"]))
        component_counts.append(int(diag["num_connected_components"]))
        largest_ratios.append(float(diag["largest_component_ratio"]))
        connected_rates.append(float(diag["connected_pair_rate"]))
        disconnected_rates.append(float(diag["disconnected_pair_rate"]))
        missing_embeddings.append(int(diag["backbone_missing_embedding_count"]))
        gold_total_pairs += int(gold_stats["pairs"])
        gold_connected_pairs += int(gold_stats["connected_pairs"])
        if gold_stats["same_component"] is not None:
            gold_same_component.append(bool(gold_stats["same_component"]))
        gold_distances.extend(float(value) for value in gold_stats["distances"])
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
        "avg_edge_length_by_type": average_edge_lengths(edge_lengths_by_type),
        "avg_symbolic_edges": mean(
            [
                sum(row.get(key, 0) for key in ("same_canonical_title", "title_mention", "shared_query_span"))
                for row in edge_counts_by_type
            ]
        )
        if edge_counts_by_type
        else 0.0,
        "avg_backbone_edges": mean(
            [sum(row.get(key, 0) for key in ("semantic_knn", "mutual_semantic_knn")) for row in edge_counts_by_type]
        )
        if edge_counts_by_type
        else 0.0,
        "avg_degree": mean(degrees) if degrees else 0.0,
        "max_degree_mean": mean(max_degrees) if max_degrees else 0.0,
        "avg_num_connected_components": mean(component_counts) if component_counts else 0.0,
        "avg_largest_component_ratio": mean(largest_ratios) if largest_ratios else 0.0,
        "avg_connected_pair_rate": mean(connected_rates) if connected_rates else 0.0,
        "avg_disconnected_pair_rate": mean(disconnected_rates) if disconnected_rates else 0.0,
        "gold_support_connected_rate": gold_connected_pairs / gold_total_pairs if gold_total_pairs else None,
        "gold_support_same_component_rate": mean(gold_same_component) if gold_same_component else None,
        "gold_support_avg_shortest_path_distance": mean(gold_distances) if gold_distances else None,
        "backbone_missing_embedding_count": mean(missing_embeddings) if missing_embeddings else 0.0,
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
        if key in {"problematic_queries", "avg_edge_counts_by_type", "avg_edge_length_by_type"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Average Edge Counts", "", "| edge_type | count |", "| --- | ---: |"])
    for key, value in metrics["avg_edge_counts_by_type"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Average Edge Lengths", "", "| edge_type | length |", "| --- | ---: |"])
    for key, value in metrics["avg_edge_length_by_type"].items():
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
