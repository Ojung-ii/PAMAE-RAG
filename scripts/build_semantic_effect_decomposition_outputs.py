#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids
from pamae_rag.diagnostics.semantic_hidden_carrier import (
    aggregate_semantic_hidden_carrier,
    semantic_hidden_carrier_rows,
)
from pamae_rag.graph.distances import build_distance_matrix
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass
from pamae_rag.objective.relevance_mass import normalize_relevance_scores, relevance_scores
from pamae_rag.selection.basin_preserving import query_anchor_indices
from pamae_rag.semantic.compatible_embedding_cache import CompatibleEmbeddingCache
from pamae_rag.semantic.semantic_candidate_ordering import build_semantic_graph_pool, node_id


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


def _indices_from_node_ids(nodes, node_ids: Iterable[str]) -> list[int]:
    by_id = {str(node.node_id): idx for idx, node in enumerate(nodes)}
    return [by_id[str(node_id)] for node_id in node_ids if str(node_id) in by_id]


def _graph_context(example, cfg, retrieval_row: dict[str, Any]) -> dict[str, Any]:
    nodes = select_universe_by_mass(
        example.nodes,
        max_nodes=cfg.universe.max_nodes,
        min_relevance_mass=cfg.universe.min_relevance_mass,
    )
    semantic_distance_matrix = build_distance_matrix(np.vstack([node.embedding for node in nodes]), metric=cfg.distance.metric)
    graph_result = build_graph_aware_distance_matrix(
        nodes,
        example.query,
        semantic_distance_matrix,
        distance_mode=cfg.pamae.distance_mode,
        distance_weights={
            "semantic": cfg.pamae.distance_weights.semantic,
            "graph": cfg.pamae.distance_weights.graph,
        },
        graph_config=cfg.pamae.graph,
    )
    graph_distance_matrix = graph_result.graph_distance_matrix
    if graph_distance_matrix is None:
        graph_distance_matrix = graph_result.distance_matrix
    graph_diagnostics = dict(graph_result.diagnostics)
    disconnected_distance = float(graph_diagnostics.get("graph_disconnected_distance", 2.0))
    candidates = [
        idx
        for idx, node in enumerate(nodes)
        if node.is_anchor_candidate and (not cfg.universe.anchor_node_types or node.node_type in cfg.universe.anchor_node_types)
    ] or list(range(len(nodes)))
    rho = normalize_relevance_scores(
        relevance_scores(
            nodes,
            mode=cfg.pamae.relevance_mode,
            query=example.query,
            query_metadata=example.metadata,
            weights=cfg.pamae.relevance_weights,
        )
    )
    diag = retrieval_row.get("diagnostics", {})
    selected = _indices_from_node_ids(nodes, retrieval_row.get("anchor_node_ids", []))
    query_anchors = _indices_from_node_ids(nodes, diag.get("diagnostic_query_anchor_node_ids", []))
    if not query_anchors:
        query_anchors = query_anchor_indices(candidates, rho, max(1, len(selected) or 1))
    pool = build_semantic_graph_pool(
        nodes=nodes,
        selected_medoids=selected,
        query_anchors=query_anchors,
        distance_matrix=graph_distance_matrix,
        disconnected_distance=disconnected_distance,
    )
    by_id = {str(node.node_id): idx for idx, node in enumerate(nodes)}

    def lookup(left: str, right: str) -> float | None:
        if left not in by_id or right not in by_id:
            return None
        value = float(graph_distance_matrix[by_id[left], by_id[right]])
        return value if np.isfinite(value) and value < disconnected_distance else None

    return {"nodes": nodes, "pool": pool, "distance_lookup": lookup}


def build_outputs(
    *,
    config_path: Path,
    input_path: Path,
    root: Path,
    cache_root: Path,
    limit: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit)
    cache = CompatibleEmbeddingCache(cache_root)
    current_rows = _read_jsonl(root / "entity_chunk_reference_current_renderer" / "retrieval_trace.jsonl")
    semantic_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    all_required: set[str] = set()
    for example in examples:
        current = current_rows.get(example.query_id)
        if current is None:
            continue
        context = _graph_context(example, cfg, current)
        nodes = context["nodes"]
        pool = context["pool"]
        support_ids = {node_id(nodes, idx) for idx in pool.support_tree_chunks}
        shell1_ids = {node_id(nodes, idx) for idx in pool.shell1_chunks}
        shell2_ids = {node_id(nodes, idx) for idx in pool.shell2_chunks}
        required = set(support_ids) | set(shell1_ids) | set(shell2_ids)
        all_required.update(required)
        store = cache.embedding_store_for_example(example, required)
        semantic_rows.extend(
            semantic_hidden_carrier_rows(
                example=example,
                current_row=current,
                store=store,
                distance_lookup=context["distance_lookup"],
                shell1_chunk_ids=shell1_ids,
            )
        )
        answer_ids = set(answer_containing_chunk_ids(example, nodes))
        pool_rows.append(
            {
                "query_id": example.query_id,
                "support_tree_chunk_count": len(pool.support_tree_chunks),
                "shell1_chunk_count": len(pool.shell1_chunks),
                "shell2_chunk_count": len(pool.shell2_chunks),
                "answer_on_support_tree": bool(answer_ids & support_ids),
                "answer_in_shell1": bool(answer_ids & shell1_ids),
                "answer_in_shell2": bool(answer_ids & shell2_ids),
            }
        )
    write_jsonl(root / "semantic_hidden_carrier_trace.jsonl", semantic_rows)
    write_jsonl(root / "semantic_pool_trace.jsonl", pool_rows)
    coverage = cache.coverage(
        query_ids=[example.query_id for example in examples],
        chunk_ids=sorted(all_required),
    )
    denom = max(len(pool_rows), 1)
    metrics = {
        "semantic_hidden_carrier": aggregate_semantic_hidden_carrier(semantic_rows),
        "pool_sizes": {
            "avg_support_tree_chunks": sum(row["support_tree_chunk_count"] for row in pool_rows) / denom,
            "avg_shell1_chunks": sum(row["shell1_chunk_count"] for row in pool_rows) / denom,
            "avg_shell2_chunks": sum(row["shell2_chunk_count"] for row in pool_rows) / denom,
            "answer_on_support_tree_rate": sum(1 for row in pool_rows if row["answer_on_support_tree"]) / denom,
            "answer_in_shell1_rate": sum(1 for row in pool_rows if row["answer_in_shell1"]) / denom,
            "answer_in_shell2_rate": sum(1 for row in pool_rows if row["answer_in_shell2"]) / denom,
        },
        "embedding": coverage,
    }
    for path in (root / "semantic_attribution_metrics.json", root / "semantic_effect_decomposition_metrics.json"):
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic effect decomposition attribution outputs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()

    metrics = build_outputs(
        config_path=args.config,
        input_path=args.input,
        root=args.root,
        cache_root=args.cache_root,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
