#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.diagnostics.current_tree_diff import aggregate_current_tree_diff, current_tree_diff_row
from pamae_rag.diagnostics.support_tree_order_budget import (
    aggregate_support_tree_order_budget,
    gold_support_tree_order_budget_rows,
    support_tree_order_budget_rows,
)
from pamae_rag.diagnostics.support_tree_order_taxonomy import aggregate_support_tree_order_taxonomy
from pamae_rag.graph.distances import build_distance_matrix
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass


REFERENCE_RUN = "entity_chunk_reference_current_renderer"
METRIC_RUN = "entity_chunk_reference_metric_path_carrier"


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def _distance_lookup(example: Any, cfg: Any) -> Callable[[str, str], float | None]:
    nodes = select_universe_by_mass(
        example.nodes,
        max_nodes=cfg.universe.max_nodes,
        min_relevance_mass=cfg.universe.min_relevance_mass,
    )
    if not nodes:
        return lambda _left, _right: None
    embeddings = np.vstack([node.embedding for node in nodes])
    semantic = build_distance_matrix(embeddings, metric=cfg.distance.metric)
    graph_result = build_graph_aware_distance_matrix(
        nodes,
        example.query,
        semantic,
        distance_mode=cfg.pamae.distance_mode,
        distance_weights={
            "semantic": cfg.pamae.distance_weights.semantic,
            "graph": cfg.pamae.distance_weights.graph,
        },
        graph_config=cfg.pamae.graph,
    )
    matrix = graph_result.graph_distance_matrix
    if matrix is None:
        matrix = graph_result.distance_matrix
    by_id = {str(node.node_id): idx for idx, node in enumerate(nodes)}
    disconnected = float(graph_result.diagnostics.get("graph_disconnected_distance", float("inf")))

    def lookup(left: str, right: str) -> float | None:
        left_idx = by_id.get(str(left))
        right_idx = by_id.get(str(right))
        if left_idx is None or right_idx is None:
            return None
        value = float(matrix[int(left_idx), int(right_idx)])
        if not np.isfinite(value) or value >= disconnected - 1e-9:
            return None
        return value

    return lookup


def build_outputs(*, config_path: Path, input_path: Path, root: Path, limit: int | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit)
    current_rows = _read_jsonl(root / REFERENCE_RUN / "retrieval_trace.jsonl")
    metric_rows = _read_jsonl(root / METRIC_RUN / "retrieval_trace.jsonl")
    current_qa = _read_jsonl(root / REFERENCE_RUN / "qa.jsonl")
    metric_qa = _read_jsonl(root / METRIC_RUN / "qa.jsonl")

    answer_order_rows: list[dict[str, Any]] = []
    gold_order_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    missing_queries: list[str] = []
    for example in examples:
        query_id = str(example.query_id)
        current = current_rows.get(query_id)
        metric = metric_rows.get(query_id)
        if current is None or metric is None:
            missing_queries.append(query_id)
            continue
        lookup = _distance_lookup(example, cfg)
        answer_order_rows.extend(
            support_tree_order_budget_rows(
                example=example,
                current_row=current,
                metric_row=metric,
                current_qa=current_qa.get(query_id),
                metric_qa=metric_qa.get(query_id),
                distance_lookup=lookup,
            )
        )
        gold_order_rows.extend(
            gold_support_tree_order_budget_rows(
                example=example,
                current_row=current,
                metric_row=metric,
                current_qa=current_qa.get(query_id),
                metric_qa=metric_qa.get(query_id),
                distance_lookup=lookup,
            )
        )
        diff_rows.append(current_tree_diff_row(example=example, current_row=current, metric_row=metric))

    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "support_tree_order_trace.jsonl", answer_order_rows)
    write_jsonl(root / "gold_support_tree_order_trace.jsonl", gold_order_rows)
    write_jsonl(root / "current_tree_diff_trace.jsonl", diff_rows)
    metrics = {
        "support_tree_order_budget": aggregate_support_tree_order_budget(answer_order_rows),
        "current_tree_diff": aggregate_current_tree_diff(diff_rows),
        "failure_taxonomy": aggregate_support_tree_order_taxonomy(answer_order_rows),
        "missing_query_ids": missing_queries,
    }
    (root / "support_tree_order_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build support-tree order/budget attribution outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_outputs(
        config_path=Path(args.config),
        input_path=Path(args.input),
        root=Path(args.root),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
