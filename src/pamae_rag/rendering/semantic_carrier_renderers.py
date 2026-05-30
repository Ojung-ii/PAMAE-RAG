from __future__ import annotations

from dataclasses import dataclass
from math import inf
import time
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids
from pamae_rag.semantic.angular_distance import angular_distance, min_angular_distance
from pamae_rag.semantic.compatible_embedding_cache import cache_from_env
from pamae_rag.semantic.embedding_store import EmbeddingStore
from pamae_rag.semantic.semantic_candidate_ordering import build_semantic_graph_pool, node_id

TREE_SHELL1_GRAPH_ORDER = "tree_shell1_graph_order"
TREE_SHELL1_SEMANTIC_QUERY_ORDER = "tree_shell1_semantic_query_order"
TREE_SHELL1_SEMANTIC_TREE_ORDER = "tree_shell1_semantic_tree_order"
SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC = "semantic_weighted_tree_diagnostic"
SHELL1_ANSWER_ORACLE = "shell1_answer_oracle"

SEMANTIC_CARRIER_RENDERERS = {
    TREE_SHELL1_GRAPH_ORDER,
    TREE_SHELL1_SEMANTIC_QUERY_ORDER,
    TREE_SHELL1_SEMANTIC_TREE_ORDER,
    SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC,
}
SEMANTIC_ORACLE_RENDERERS = {SHELL1_ANSWER_ORACLE}
SEMANTIC_ADOPTION_CANDIDATE_RENDERERS = {
    TREE_SHELL1_GRAPH_ORDER,
    TREE_SHELL1_SEMANTIC_QUERY_ORDER,
    TREE_SHELL1_SEMANTIC_TREE_ORDER,
}


@dataclass(frozen=True)
class SemanticCarrierRenderResult:
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


def _embedding_distance_to_query(store: EmbeddingStore, idx: int, nodes: Sequence[EvidenceNode]) -> float:
    query_embedding = store.query_embedding
    chunk_embedding = store.node_embedding(node_id(nodes, idx))
    if query_embedding is None:
        raise ValueError("query embedding is required for tree_shell1_semantic_query_order")
    if chunk_embedding is None:
        return inf
    return angular_distance(query_embedding, chunk_embedding)


def semantic_query_order_keys(
    *,
    store: EmbeddingStore,
    candidate_indices: Sequence[int],
    nodes: Sequence[EvidenceNode],
) -> dict[int, float]:
    """Return angular-equivalent ordering keys for query-to-chunk distance.

    Embeddings are L2-normalized by the compatible cache. Since arccos is
    monotonic decreasing over cosine similarity, sorting by ``-dot(q, u)`` is
    exactly the same order as sorting by angular distance, with the existing
    node-id tie-breaks kept outside this helper.
    """

    query_embedding = store.query_embedding
    if query_embedding is None:
        raise ValueError("query embedding is required for tree_shell1_semantic_query_order")
    keys: dict[int, float] = {}
    for idx in candidate_indices:
        chunk_embedding = store.node_embedding(node_id(nodes, idx))
        keys[int(idx)] = inf if chunk_embedding is None else -float(np.dot(query_embedding, chunk_embedding))
    return keys


def _embedding_distance_to_tree(
    store: EmbeddingStore,
    idx: int,
    tree_chunks: set[int],
    nodes: Sequence[EvidenceNode],
) -> float:
    chunk_embedding = store.node_embedding(node_id(nodes, idx))
    if chunk_embedding is None:
        return inf
    tree_embeddings = [
        store.node_embedding(node_id(nodes, tree_idx))
        for tree_idx in sorted(tree_chunks, key=lambda tree_idx: node_id(nodes, tree_idx))
    ]
    tree_embeddings = [embedding for embedding in tree_embeddings if embedding is not None]
    distance = min_angular_distance(chunk_embedding, tree_embeddings) if tree_embeddings else None
    return inf if distance is None else float(distance)


def render_semantic_carrier_indices(
    *,
    example: QueryExample,
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    disconnected_distance: float,
    renderer_mode: str,
    embedding_store: EmbeddingStore | None = None,
    include_trace: bool = True,
) -> SemanticCarrierRenderResult:
    if renderer_mode not in SEMANTIC_CARRIER_RENDERERS and renderer_mode not in SEMANTIC_ORACLE_RENDERERS:
        raise ValueError(f"Unknown semantic carrier renderer: {renderer_mode}")
    if renderer_mode == SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC:
        raise ValueError("semantic_weighted_tree_diagnostic is rendered by semantic_weighted_tree.py")

    nodes = example.nodes
    started = time.perf_counter()
    pool_started = time.perf_counter()
    pool = build_semantic_graph_pool(
        nodes=nodes,
        selected_medoids=selected_medoids,
        query_anchors=query_anchors,
        distance_matrix=distance_matrix,
        disconnected_distance=disconnected_distance,
    )
    pool_ms = (time.perf_counter() - pool_started) * 1000.0
    candidates = set(pool.pool_chunks)
    lookup_started = time.perf_counter()
    if embedding_store is None:
        compatible_cache = cache_from_env()
        if compatible_cache is not None:
            candidate_ids = [node_id(nodes, idx) for idx in sorted(candidates)]
            store = compatible_cache.embedding_store_for_example(example, candidate_ids)
        else:
            store = EmbeddingStore.from_example(example)
    else:
        store = embedding_store
    lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
    oracle = renderer_mode in SEMANTIC_ORACLE_RENDERERS
    if renderer_mode == SHELL1_ANSWER_ORACLE:
        answer_ids = set(answer_containing_chunk_ids(example, nodes))
        candidates = {
            idx
            for idx in pool.shell1_chunks
            if node_id(nodes, idx) in answer_ids
        }

    semantic_missing = {node_id(nodes, idx) for idx in candidates if store.node_embedding(node_id(nodes, idx)) is None}
    role_order = {"medoid": 0, "anchor_medoid_path": 1, "medoid_medoid_path": 2, "strict_tree": 3, "shell1": 4}
    semantic_started = time.perf_counter()
    query_semantic_keys = (
        semantic_query_order_keys(store=store, candidate_indices=sorted(candidates), nodes=nodes)
        if renderer_mode == TREE_SHELL1_SEMANTIC_QUERY_ORDER
        else {}
    )
    tree_semantic_keys = (
        {
            int(idx): _embedding_distance_to_tree(store, idx, pool.support_tree_chunks, nodes)
            for idx in sorted(candidates)
        }
        if renderer_mode == TREE_SHELL1_SEMANTIC_TREE_ORDER
        else {}
    )

    def key(idx: int) -> tuple[Any, ...]:
        role = pool.roles.get(idx, {"role": "shell1", "path_position": 10**9, "node_id": node_id(nodes, idx)})
        graph_distance = float(pool.graph_distance_to_tree.get(idx, inf))
        base = (
            int(role_order.get(str(role["role"]), 9)),
            graph_distance,
        )
        if renderer_mode == TREE_SHELL1_SEMANTIC_QUERY_ORDER:
            return (*base, query_semantic_keys[int(idx)], node_id(nodes, idx), int(idx))
        if renderer_mode == TREE_SHELL1_SEMANTIC_TREE_ORDER:
            return (*base, tree_semantic_keys[int(idx)], node_id(nodes, idx), int(idx))
        return (
            *base,
            int(role_order.get(str(role["role"]), 9)),
            int(role.get("path_position", 10**9)),
            node_id(nodes, idx),
            int(idx),
        )

    ordered = sorted(candidates, key=key)
    semantic_ms = (time.perf_counter() - semantic_started) * 1000.0
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    cutoff: list[int] = []
    for idx in ordered:
        role = str(pool.roles.get(idx, {}).get("role", "shell1"))
        added = budget.add(idx, force=bool(role == "medoid"))
        if not added:
            cutoff.append(int(idx))

    rendered_shell1 = [idx for idx in budget.selected if idx in pool.shell1_chunks]
    diagnostics = {
        "renderer_mode": renderer_mode,
        "graph_constraint": "T_q union S1",
        "graph_defined_pool_only": True,
        "adoption_candidate": bool(renderer_mode in SEMANTIC_ADOPTION_CANDIDATE_RENDERERS),
        "diagnostic_renderer": bool(renderer_mode == SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC),
        "oracle_renderer": oracle,
        "uses_answer_string": bool(renderer_mode == SHELL1_ANSWER_ORACLE),
        "uses_gold_label": False,
        "score_mixing_detected": False,
        "oracle_leakage_count": 0 if not oracle else None,
        "support_tree_chunk_count": len(pool.support_tree_chunks),
        "shell1_chunk_count": len(pool.shell1_chunks),
        "shell2_chunk_count": len(pool.shell2_chunks),
        "pool_chunk_count": len(pool.pool_chunks),
        "candidate_pool_size": len(pool.pool_chunks),
        "rendered_shell1_chunk_count": len(rendered_shell1),
        "rendered_shell1_chunk_ids": [node_id(nodes, idx) for idx in rendered_shell1],
        "semantic_missing_node_count": len(semantic_missing),
        "embedding_missing_rate": float(len(semantic_missing) / max(len(candidates), 1)),
        "budget_cutoff_count": len(cutoff),
        "context_tokens": int(sum(max(1, int(nodes[idx].token_count)) for idx in budget.selected)),
        "time_support_tree_ms": pool_ms,
        "time_shell1_construction_ms": pool_ms,
        "time_embedding_lookup_ms": lookup_ms,
        "time_semantic_ordering_ms": semantic_ms,
        "candidate_embedding_lookup_count": len(candidates),
        "unique_candidate_embedding_count": len(candidates),
        "duplicate_embedding_lookup_avoided": 0,
        "query_embedding_cache_hit_rate": 1.0 if store.query_embedding is not None else 0.0,
        "query_embedding_cache_miss_count": 0 if store.query_embedding is not None else 1,
        "time_query_embedding_ms": 0.0,
        "time_semantic_renderer_total_ms": (time.perf_counter() - started) * 1000.0,
    }
    if include_trace:
        diagnostics.update(
            {
                "support_tree_chunk_ids": [
                    node_id(nodes, idx) for idx in sorted(pool.support_tree_chunks, key=lambda i: node_id(nodes, i))
                ],
                "shell1_chunk_ids": [node_id(nodes, idx) for idx in sorted(pool.shell1_chunks, key=lambda i: node_id(nodes, i))],
                "shell2_chunk_ids": [node_id(nodes, idx) for idx in sorted(pool.shell2_chunks, key=lambda i: node_id(nodes, i))],
                "semantic_carrier_order_node_ids": [node_id(nodes, idx) for idx in ordered],
                "budget_cutoff_node_ids": [node_id(nodes, idx) for idx in cutoff],
            }
        )
    return SemanticCarrierRenderResult(indices=budget.selected, diagnostics=diagnostics)


__all__ = [
    "SEMANTIC_ADOPTION_CANDIDATE_RENDERERS",
    "SEMANTIC_CARRIER_RENDERERS",
    "SEMANTIC_ORACLE_RENDERERS",
    "SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC",
    "SHELL1_ANSWER_ORACLE",
    "TREE_SHELL1_GRAPH_ORDER",
    "TREE_SHELL1_SEMANTIC_QUERY_ORDER",
    "TREE_SHELL1_SEMANTIC_TREE_ORDER",
    "SemanticCarrierRenderResult",
    "render_semantic_carrier_indices",
    "semantic_query_order_keys",
]
