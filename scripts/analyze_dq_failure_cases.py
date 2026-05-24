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
from pamae_rag.graph.distances import build_distance_matrix
from pamae_rag.graph.graph_distance import graph_diagnostics, graph_shortest_path_distance
from pamae_rag.graph.query_graph import build_minimal_query_graph, extract_query_spans
from pamae_rag.graph.universe import select_universe_by_mass


def _read_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def _success(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("support_hit"))


def _overlap(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def _gold_graph_stats(example, nodes, cfg) -> dict[str, Any]:
    semantic = build_distance_matrix(np.vstack([node.embedding for node in nodes]), metric=cfg.distance.metric)
    graph = build_minimal_query_graph(
        nodes,
        example.query,
        edge_lengths=dict(cfg.pamae.graph.edge_lengths.__dict__),
        max_edges_per_node=cfg.pamae.graph.max_edges_per_node,
        semantic_distance_matrix=semantic,
        backbone_config=cfg.pamae.graph.backbone,
    )
    sp = graph_shortest_path_distance(graph, cfg.pamae.graph.disconnected_distance)
    diag = graph_diagnostics(graph, sp, cfg.pamae.graph.disconnected_distance)
    id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}
    gold = [id_to_idx[node_id] for node_id in sorted(example.gold_node_ids) if node_id in id_to_idx]
    gold_pair_distances: list[float] = []
    for pos, i in enumerate(gold):
        for j in gold[pos + 1 :]:
            value = float(sp[int(i), int(j)])
            if value < cfg.pamae.graph.disconnected_distance - 1e-12:
                gold_pair_distances.append(value)
    return {
        "num_edges": graph.num_edges,
        "largest_component_ratio": diag["largest_component_ratio"],
        "disconnected_pair_rate": diag["disconnected_pair_rate"],
        "gold_support_avg_shortest_path_distance": mean(gold_pair_distances) if gold_pair_distances else None,
        "query_span_count": len(extract_query_spans(example.query)),
    }


def analyze_failure_cases(
    input_path: str | Path,
    config_path: str | Path,
    top_rho_predictions: str | Path,
    semantic_predictions: str | Path,
    graph_predictions: str | Path,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=cfg.experiment.limit_queries)
    top = _read_predictions(top_rho_predictions)
    semantic = _read_predictions(semantic_predictions)
    graph = _read_predictions(graph_predictions)
    groups: dict[str, list[dict[str, Any]]] = {
        "top_rho_success_graph_refine_cell_fail": [],
        "top_rho_fail_graph_refine_cell_success": [],
        "both_success": [],
        "both_fail": [],
    }
    for example in examples:
        top_row = top.get(example.query_id)
        sem_row = semantic.get(example.query_id)
        graph_row = graph.get(example.query_id)
        top_ok = _success(top_row)
        graph_ok = _success(graph_row)
        if top_ok and not graph_ok:
            group = "top_rho_success_graph_refine_cell_fail"
        elif not top_ok and graph_ok:
            group = "top_rho_fail_graph_refine_cell_success"
        elif top_ok and graph_ok:
            group = "both_success"
        else:
            group = "both_fail"
        nodes = select_universe_by_mass(
            example.nodes,
            max_nodes=cfg.universe.max_nodes,
            min_relevance_mass=cfg.universe.min_relevance_mass,
        )
        stats = _gold_graph_stats(example, nodes, cfg)
        context_overlap = _overlap(
            top_row.get("context_node_ids", []) if top_row else [],
            graph_row.get("context_node_ids", []) if graph_row else [],
        )
        groups[group].append(
            {
                "query_id": example.query_id,
                "top_rho_hit": top_ok,
                "semantic_refine_cell_hit": _success(sem_row),
                "graph_refine_cell_hit": graph_ok,
                "context_overlap": context_overlap,
                **stats,
            }
        )
    summary: dict[str, Any] = {}
    for key, rows in groups.items():
        summary[key] = {
            "count": len(rows),
            "avg_context_overlap": mean([row["context_overlap"] for row in rows]) if rows else None,
            "avg_query_span_count": mean([row["query_span_count"] for row in rows]) if rows else None,
            "avg_largest_component_ratio": mean([row["largest_component_ratio"] for row in rows]) if rows else None,
            "avg_disconnected_pair_rate": mean([row["disconnected_pair_rate"] for row in rows]) if rows else None,
            "sample_query_ids": [row["query_id"] for row in rows[:10]],
        }
    return {"summary": summary, "groups": groups}


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# d_q Failure Case Analysis",
        "",
        "| group | count | avg_context_overlap | avg_query_span_count | avg_largest_component_ratio | avg_disconnected_pair_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["summary"].items():
        lines.append(
            f"| {key} | {row['count']} | {row['avg_context_overlap']} | {row['avg_query_span_count']} | "
            f"{row['avg_largest_component_ratio']} | {row['avg_disconnected_pair_rate']} |"
        )
    lines.extend(["", "## Sample Query IDs", ""])
    for key, row in result["summary"].items():
        samples = ", ".join(f"`{qid}`" for qid in row["sample_query_ids"])
        lines.append(f"- {key}: {samples}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-rho-predictions", required=True)
    parser.add_argument("--semantic-predictions", required=True)
    parser.add_argument("--graph-predictions", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    result = analyze_failure_cases(
        args.input,
        args.config,
        args.top_rho_predictions,
        args.semantic_predictions,
        args.graph_predictions,
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
