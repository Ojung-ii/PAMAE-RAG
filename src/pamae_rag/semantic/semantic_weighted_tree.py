from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.semantic.angular_distance import angular_distance
from pamae_rag.semantic.compatible_embedding_cache import cache_from_env
from pamae_rag.semantic.embedding_store import EmbeddingStore
from pamae_rag.semantic.semantic_candidate_ordering import dedupe_indices, is_chunk, node_id


@dataclass(frozen=True)
class SemanticWeightedTreeResult:
    indices: list[int]
    diagnostics: dict[str, Any]


@dataclass
class _Budget:
    nodes: Sequence[EvidenceNode]
    max_context_tokens: int
    max_context_nodes: int | None
    selected: list[int]
    used_tokens: int = 0

    def add(self, idx: int, *, force: bool = False) -> bool:
        idx = int(idx)
        if idx in self.selected:
            return False
        tokens = max(1, int(self.nodes[idx].token_count))
        if self.used_tokens + tokens > self.max_context_tokens:
            return False
        if not force and self.max_context_nodes and len(self.selected) >= self.max_context_nodes:
            return False
        self.selected.append(idx)
        self.used_tokens += tokens
        return True


def semantic_edge_length(
    *,
    left_idx: int,
    right_idx: int,
    nodes: Sequence[EvidenceNode],
    store: EmbeddingStore,
) -> float | None:
    left_embedding = store.node_embedding(node_id(nodes, left_idx))
    right_embedding = store.node_embedding(node_id(nodes, right_idx))
    if left_embedding is None or right_embedding is None:
        return None
    return float(1.0 + angular_distance(left_embedding, right_embedding))


def _adjacency_from_graph_distance(
    *,
    nodes: Sequence[EvidenceNode],
    distance_matrix: np.ndarray,
    store: EmbeddingStore,
    disconnected_distance: float,
    eps: float = 1e-9,
) -> tuple[dict[int, list[tuple[int, float]]], int]:
    matrix_size = min(len(nodes), int(distance_matrix.shape[0]))
    adjacency: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(matrix_size)}
    missing_edges = 0
    for left in range(matrix_size):
        for right in range(left + 1, matrix_size):
            value = float(distance_matrix[left, right])
            if not np.isfinite(value) or value >= float(disconnected_distance) - eps:
                continue
            if abs(value - 1.0) > eps:
                continue
            length = semantic_edge_length(left_idx=left, right_idx=right, nodes=nodes, store=store)
            if length is None:
                missing_edges += 1
                continue
            adjacency[left].append((right, length))
            adjacency[right].append((left, length))
    for idx in adjacency:
        adjacency[idx].sort(key=lambda item: (item[1], node_id(nodes, item[0]), item[0]))
    return adjacency, missing_edges


def _shortest_path(
    *,
    start: int,
    end: int,
    adjacency: dict[int, list[tuple[int, float]]],
    nodes: Sequence[EvidenceNode],
) -> list[int]:
    if int(start) == int(end):
        return [int(start)]
    queue: list[tuple[float, tuple[str, ...], int, list[int]]] = [(0.0, (node_id(nodes, start),), int(start), [int(start)])]
    best: dict[int, tuple[float, tuple[str, ...]]] = {int(start): (0.0, (node_id(nodes, start),))}
    while queue:
        distance, id_path, idx, path = heappop(queue)
        if idx == int(end):
            return path
        if best.get(idx) != (distance, id_path):
            continue
        for neighbor, length in adjacency.get(idx, []):
            next_distance = distance + float(length)
            next_id_path = (*id_path, node_id(nodes, neighbor))
            current = best.get(neighbor)
            if current is None or (next_distance, next_id_path) < current:
                best[neighbor] = (next_distance, next_id_path)
                heappush(queue, (next_distance, next_id_path, int(neighbor), [*path, int(neighbor)]))
    return []


def semantic_weighted_support_tree_indices(
    *,
    example: QueryExample,
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    disconnected_distance: float,
    embedding_store: EmbeddingStore | None = None,
) -> SemanticWeightedTreeResult:
    nodes = example.nodes
    matrix_size = min(len(nodes), int(distance_matrix.shape[0]))
    if embedding_store is None:
        compatible_cache = cache_from_env()
        if compatible_cache is not None:
            store = compatible_cache.embedding_store_for_example(
                example,
                [str(node.node_id) for node in nodes[:matrix_size]],
            )
        else:
            store = EmbeddingStore.from_example(example)
    else:
        store = embedding_store
    adjacency, missing_edge_count = _adjacency_from_graph_distance(
        nodes=nodes,
        distance_matrix=distance_matrix,
        store=store,
        disconnected_distance=disconnected_distance,
    )
    anchors = sorted(
        (idx for idx in dedupe_indices(query_anchors) if 0 <= int(idx) < matrix_size),
        key=lambda idx: (node_id(nodes, idx), idx),
    )
    medoids = sorted(
        (idx for idx in dedupe_indices(selected_medoids) if 0 <= int(idx) < matrix_size),
        key=lambda idx: (node_id(nodes, idx), idx),
    )
    tree: set[int] = set(medoids) | set(anchors)
    path_order: list[int] = []
    for anchor in anchors:
        for medoid in medoids:
            path = _shortest_path(start=anchor, end=medoid, adjacency=adjacency, nodes=nodes)
            tree.update(path)
            path_order.extend(path)
    for pos, left in enumerate(medoids):
        for right in medoids[pos + 1 :]:
            path = _shortest_path(start=left, end=right, adjacency=adjacency, nodes=nodes)
            tree.update(path)
            path_order.extend(path)

    tree_chunks = {idx for idx in tree if 0 <= int(idx) < matrix_size and is_chunk(nodes, idx)}
    ordered = sorted(
        tree_chunks,
        key=lambda idx: (
            0 if idx in medoids else 1,
            path_order.index(idx) if idx in path_order else 10**9,
            node_id(nodes, idx),
            int(idx),
        ),
    )
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    cutoff: list[int] = []
    for idx in ordered:
        added = budget.add(idx, force=bool(idx in medoids))
        if not added:
            cutoff.append(int(idx))

    diagnostics = {
        "renderer_mode": "semantic_weighted_tree_diagnostic",
        "diagnostic_renderer": True,
        "adoption_candidate": False,
        "graph_constraint": "semantic-weighted SPClosure over graph edges",
        "edge_length_formula": "1 + d_ang(i,j)",
        "positive_edge_lengths": True,
        "score_mixing_detected": False,
        "missing_semantic_edge_count": missing_edge_count,
        "support_tree_chunk_count": len(tree_chunks),
        "support_tree_chunk_ids": [node_id(nodes, idx) for idx in sorted(tree_chunks, key=lambda i: node_id(nodes, i))],
        "semantic_weighted_order_node_ids": [node_id(nodes, idx) for idx in ordered],
        "budget_cutoff_count": len(cutoff),
        "budget_cutoff_node_ids": [node_id(nodes, idx) for idx in cutoff],
        "context_tokens": int(sum(max(1, int(nodes[idx].token_count)) for idx in budget.selected)),
    }
    return SemanticWeightedTreeResult(indices=budget.selected, diagnostics=diagnostics)


__all__ = [
    "SemanticWeightedTreeResult",
    "semantic_edge_length",
    "semantic_weighted_support_tree_indices",
]
