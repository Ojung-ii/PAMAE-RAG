from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterable

from pamae_rag.local_surface.local_surface_graph import LocalSurfaceGraph


def shortest_path_distances(
    graph: LocalSurfaceGraph,
    sources: Iterable[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    all_nodes = set(graph.node_ids)
    for source in sources:
        source = str(source)
        distances = {node_id: float("inf") for node_id in all_nodes}
        if source not in all_nodes:
            out[source] = distances
            continue
        distances[source] = 0.0
        queue: deque[str] = deque([source])
        while queue:
            node_id = queue.popleft()
            for neighbor, length, _edge_type in graph.adjacency.get(node_id, ()):
                candidate = distances[node_id] + float(length)
                if candidate >= distances.get(neighbor, float("inf")):
                    continue
                distances[neighbor] = candidate
                queue.append(neighbor)
        out[source] = distances
    return out


def distance_between(graph: LocalSurfaceGraph, source: str, target: str) -> float:
    return shortest_path_distances(graph, [source])[str(source)].get(str(target), float("inf"))


def validate_triangle_inequality(
    graph: LocalSurfaceGraph,
    node_ids: Iterable[str] | None = None,
    *,
    max_triples: int = 512,
    tolerance: float = 1e-12,
) -> int:
    nodes = tuple(str(node_id) for node_id in (node_ids or graph.node_ids))
    if len(nodes) < 3:
        return 0
    distances = shortest_path_distances(graph, nodes)
    count = 0
    violations = 0
    for a, b, c in combinations(nodes, 3):
        count += 1
        dab = distances[a].get(b, float("inf"))
        dac = distances[a].get(c, float("inf"))
        dbc = distances[b].get(c, float("inf"))
        triples = ((dab, dac, dbc), (dac, dab, dbc), (dbc, dab, dac))
        for left, right_a, right_b in triples:
            if left < float("inf") and right_a < float("inf") and right_b < float("inf"):
                if left > right_a + right_b + tolerance:
                    violations += 1
        if count >= max_triples:
            break
    return violations


__all__ = ["distance_between", "shortest_path_distances", "validate_triangle_inequality"]
