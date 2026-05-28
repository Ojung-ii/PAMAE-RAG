from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised by integration runs when SciPy is installed.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
except ModuleNotFoundError:  # pragma: no cover - focused unit env may not include SciPy.
    csr_matrix = None
    shortest_path = None

from pamae_rag.sentence_graph.sentence_graph_builder import SentenceGraphIndex


@dataclass(frozen=True)
class SentenceMetricResult:
    sentence_ids: tuple[str, ...]
    distance_matrix: np.ndarray
    diagnostics: dict[str, Any]


def _dijkstra(
    adjacency: dict[str, list[tuple[str, float, str]]],
    source_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    distances: dict[str, float] = {source_id: 0.0}
    previous: dict[str, str] = {}
    heap: list[tuple[float, str]] = [(0.0, source_id)]
    while heap:
        value, node_id = heapq.heappop(heap)
        if value > distances.get(node_id, float("inf")):
            continue
        for neighbor_id, length, _edge_type in adjacency.get(node_id, []):
            candidate = value + float(length)
            if candidate < distances.get(neighbor_id, float("inf")):
                distances[neighbor_id] = candidate
                previous[neighbor_id] = node_id
                heapq.heappush(heap, (candidate, neighbor_id))
    return distances, previous


def sentence_shortest_path_distances(
    index: SentenceGraphIndex,
    sentence_ids: Iterable[str],
    *,
    use_chunk_parent_edges_in_metric: bool = False,
    disconnected_distance: float = 1_000_000.0,
) -> SentenceMetricResult:
    selected = tuple(dict.fromkeys(str(sentence_id) for sentence_id in sentence_ids))
    if disconnected_distance < 0:
        raise ValueError("disconnected_distance must be nonnegative")
    sentence_set = set(index.sentence_ids)
    unknown = [sentence_id for sentence_id in selected if sentence_id not in sentence_set]
    if unknown:
        raise ValueError(f"Unknown sentence ids in metric universe: {unknown[:3]}")

    adjacency = index.adjacency(include_chunk_parent_edges=use_chunk_parent_edges_in_metric)
    graph_node_ids = sorted(set(adjacency) | {neighbor for values in adjacency.values() for neighbor, _, _ in values} | set(selected))
    node_pos = {node_id: idx for idx, node_id in enumerate(graph_node_ids)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source_id, neighbors in adjacency.items():
        source_pos = node_pos[source_id]
        for target_id, length, _edge_type in neighbors:
            rows.append(source_pos)
            cols.append(node_pos[target_id])
            data.append(float(length))
    if graph_node_ids and data and csr_matrix is not None and shortest_path is not None:
        graph = csr_matrix((data, (rows, cols)), shape=(len(graph_node_ids), len(graph_node_ids)), dtype=np.float64)
        selected_pos = np.asarray([node_pos[sentence_id] for sentence_id in selected], dtype=np.int64)
        distances = shortest_path(graph, directed=False, unweighted=False, indices=selected_pos)
        distances = np.asarray(distances, dtype=np.float64)
        matrix = distances[:, selected_pos]
        matrix[~np.isfinite(matrix)] = float(disconnected_distance)
        matrix = np.minimum(matrix, float(disconnected_distance))
    else:
        matrix = np.full((len(selected), len(selected)), float(disconnected_distance), dtype=np.float64)
        for row, source_id in enumerate(selected):
            distances, _previous = _dijkstra(adjacency, source_id)
            for col, target_id in enumerate(selected):
                matrix[row, col] = min(
                    float(distances.get(target_id, disconnected_distance)),
                    disconnected_distance,
                )
    matrix = np.minimum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    finite_mask = matrix < float(disconnected_distance)
    offdiag = len(selected) * max(len(selected) - 1, 0)
    finite_pairs = int(np.sum(finite_mask & ~np.eye(len(selected), dtype=bool)))
    diagnostics = {
        "metric_sentence_count": len(selected),
        "metric_use_chunk_parent_edges": bool(use_chunk_parent_edges_in_metric),
        "metric_disconnected_distance": float(disconnected_distance),
        "metric_connected_pair_rate": finite_pairs / max(offdiag, 1),
    }
    return SentenceMetricResult(selected, matrix, diagnostics)


def shortest_path_node_ids(
    index: SentenceGraphIndex,
    source_id: str,
    target_id: str,
    *,
    use_chunk_parent_edges_in_metric: bool = False,
) -> tuple[str, ...]:
    adjacency = index.adjacency(include_chunk_parent_edges=use_chunk_parent_edges_in_metric)
    distances, previous = _dijkstra(adjacency, source_id)
    if target_id not in distances:
        return tuple()
    out = [target_id]
    while out[-1] != source_id:
        parent = previous.get(out[-1])
        if parent is None:
            return tuple()
        out.append(parent)
    out.reverse()
    return tuple(out)


def triangle_inequality_violation_count(
    distance_matrix: np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_nodes: int = 64,
    seed: int = 0,
) -> int:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("distance_matrix must be square")
    if n == 0:
        return 0
    if n <= max_nodes:
        sample = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(np.arange(n, dtype=np.int64), size=max_nodes, replace=False))
    violations = 0
    for i in sample:
        for j in sample:
            dij = matrix[int(i), int(j)]
            for k in sample:
                if dij > matrix[int(i), int(k)] + matrix[int(k), int(j)] + tolerance:
                    violations += 1
    return int(violations)
