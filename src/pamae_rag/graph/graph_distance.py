from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.graph.content_graph import project_content_graph_to_query_graph
from pamae_rag.graph.query_graph import QueryGraph, build_minimal_query_graph


@dataclass(frozen=True)
class GraphDistanceResult:
    distance_matrix: np.ndarray
    diagnostics: dict[str, Any]


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Expected mapping-like config, got {type(value).__name__}")


def _adjacency(graph: QueryGraph) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(graph.num_nodes)]
    for edge in graph.edges:
        adj[edge.source].append((edge.target, edge.length))
        adj[edge.target].append((edge.source, edge.length))
    for row in adj:
        row.sort(key=lambda item: (item[0], item[1]))
    return adj


def connected_components(graph: QueryGraph) -> list[list[int]]:
    adj = _adjacency(graph)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(graph.num_nodes):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt, _ in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(component))
    return components


def graph_shortest_path_distance(graph: QueryGraph, disconnected_distance: float) -> np.ndarray:
    if disconnected_distance < 0:
        raise ValueError("disconnected_distance must be nonnegative")
    n = graph.num_nodes
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if graph.edges:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for edge in graph.edges:
            rows.extend([edge.source, edge.target])
            cols.extend([edge.target, edge.source])
            data.extend([float(edge.length), float(edge.length)])
        matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        out = shortest_path(matrix, directed=False, unweighted=False)
        out = np.asarray(out, dtype=np.float64)
        out[~np.isfinite(out)] = float(disconnected_distance)
        out = np.minimum(out, float(disconnected_distance))
        out = np.minimum(out, out.T)
        np.fill_diagonal(out, 0.0)
        return out
    adj = _adjacency(graph)
    out = np.full((n, n), float(disconnected_distance), dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    for source in range(n):
        dist = [float("inf")] * n
        dist[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            value, node = heapq.heappop(heap)
            if value > dist[node]:
                continue
            for nxt, length in adj[node]:
                candidate = value + float(length)
                if candidate < dist[nxt]:
                    dist[nxt] = candidate
                    heapq.heappush(heap, (candidate, nxt))
        for target, value in enumerate(dist):
            if value < float("inf"):
                out[source, target] = value
    out = np.minimum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out


def graph_diagnostics(graph: QueryGraph, distance_matrix: np.ndarray, disconnected_distance: float) -> dict[str, Any]:
    components = connected_components(graph)
    n = max(graph.num_nodes, 1)
    largest = max((len(component) for component in components), default=0)
    offdiag = graph.num_nodes * max(graph.num_nodes - 1, 0)
    disconnected = 0
    if offdiag:
        disconnected = int(np.sum((distance_matrix >= disconnected_distance - 1e-12) & ~np.eye(graph.num_nodes, dtype=bool)))
    degrees = [0 for _ in range(graph.num_nodes)]
    for edge in graph.edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    return {
        "num_edges": graph.num_edges,
        "edge_counts_by_type": dict(graph.edge_counts_by_type),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees, default=0)),
        "num_connected_components": len(components),
        "largest_component_ratio": largest / n,
        "connected_pair_rate": 1.0 - (disconnected / max(offdiag, 1)),
        "disconnected_pair_rate": disconnected / max(offdiag, 1),
        "graph_disconnected_distance": float(disconnected_distance),
        "backbone_missing_embedding_count": int(graph.backbone_missing_embedding_count),
    }


def build_graph_aware_distance_matrix(
    nodes: tuple[EvidenceNode, ...] | list[EvidenceNode],
    query: str,
    semantic_distance_matrix: np.ndarray,
    *,
    distance_mode: str,
    distance_weights: dict[str, float],
    graph_config: Any,
) -> GraphDistanceResult:
    semantic = np.asarray(semantic_distance_matrix, dtype=np.float64)
    if distance_mode == "semantic":
        return GraphDistanceResult(
            semantic,
            {
                "distance_mode": "semantic",
                "graph_source": "none",
                "lambda_s": 1.0,
                "lambda_g": 0.0,
                "num_edges": 0,
                "edge_counts_by_type": {},
                "num_connected_components": len(nodes),
                "largest_component_ratio": 1.0 / max(len(nodes), 1),
                "disconnected_pair_rate": 1.0 if len(nodes) > 1 else 0.0,
                "graph_disconnected_distance": float(graph_config.disconnected_distance),
            },
        )

    graph_source = str(getattr(graph_config, "source", "legacy_query"))
    projected_node_ids: list[str] = []
    graph_start = time.perf_counter()
    if graph_source == "content":
        graph, content_index, projected_node_indices = project_content_graph_to_query_graph(
            nodes,
            edge_lengths=_as_plain_dict(graph_config.edge_lengths),
            max_edges_per_node=int(graph_config.max_edges_per_node),
        )
        content_diagnostics = dict(content_index.diagnostics)
        projected_node_ids = [nodes[int(idx)].node_id for idx in projected_node_indices]
    else:
        graph = build_minimal_query_graph(
            nodes,
            query,
            edge_lengths=_as_plain_dict(graph_config.edge_lengths),
            max_edges_per_node=int(graph_config.max_edges_per_node),
            semantic_distance_matrix=semantic,
            backbone_config=getattr(graph_config, "backbone", None),
        )
        content_diagnostics = {}
    graph_build_latency_ms = (time.perf_counter() - graph_start) * 1000.0
    graph_sp = graph_shortest_path_distance(graph, float(graph_config.disconnected_distance))
    diag = graph_diagnostics(graph, graph_sp, float(graph_config.disconnected_distance))
    diag.update(content_diagnostics)

    if distance_mode == "graph_sp":
        matrix = graph_sp
        lambda_s = 0.0
        lambda_g = 1.0
    elif distance_mode == "hybrid_sem_graph":
        weights = _as_plain_dict(distance_weights)
        lambda_s = float(weights.get("semantic", 0.7))
        lambda_g = float(weights.get("graph", 0.3))
        matrix = lambda_s * semantic + lambda_g * graph_sp
    else:
        raise ValueError(f"Unsupported pamae.distance_mode: {distance_mode}")

    matrix = np.maximum(matrix, 0.0)
    matrix = np.minimum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    diag.update(
        {
            "distance_mode": distance_mode,
            "graph_source": graph_source,
            "lambda_s": lambda_s,
            "lambda_g": lambda_g,
            "graph_build_latency_ms": graph_build_latency_ms,
            "projected_node_ids": projected_node_ids,
        }
    )
    return GraphDistanceResult(matrix, diag)


def average_edge_counts(values: list[dict[str, int]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    keys: set[str] = set()
    for value in values:
        keys |= set(value)
        for key, count in value.items():
            totals[key] += float(count)
    denom = max(len(values), 1)
    return {key: totals[key] / denom for key in sorted(keys)}


def average_edge_lengths(values: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    keys: set[str] = set()
    for value in values:
        keys |= set(value)
        for key, length in value.items():
            totals[key] += float(length)
            counts[key] += 1
    return {key: totals[key] / max(counts[key], 1) for key in sorted(keys)}
