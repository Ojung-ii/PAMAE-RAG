from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.diagnostics.path_realizability import path_nodes, support_tree_nodes


@dataclass(frozen=True)
class SemanticGraphPool:
    support_tree: set[int]
    support_tree_chunks: set[int]
    shell1_chunks: set[int]
    shell2_chunks: set[int]
    roles: dict[int, dict[str, Any]]
    graph_distance_to_tree: dict[int, float]

    @property
    def pool_chunks(self) -> set[int]:
        return set(self.support_tree_chunks) | set(self.shell1_chunks)


def node_id(nodes: Sequence[EvidenceNode], idx: int) -> str:
    return str(nodes[int(idx)].node_id)


def is_chunk(nodes: Sequence[EvidenceNode], idx: int) -> bool:
    return str(getattr(nodes[int(idx)], "node_type", "chunk")) == "chunk"


def dedupe_indices(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _finite_distance(value: float, disconnected_distance: float) -> bool:
    return bool(np.isfinite(value) and float(value) < float(disconnected_distance) - 1e-9)


def _distance_to_tree(
    *,
    idx: int,
    tree_chunks: set[int],
    distance_matrix: np.ndarray,
    disconnected_distance: float,
) -> float | None:
    if int(idx) in tree_chunks:
        return 0.0
    candidates = [
        float(distance_matrix[int(idx), int(tree_idx)])
        for tree_idx in sorted(tree_chunks)
        if _finite_distance(float(distance_matrix[int(idx), int(tree_idx)]), disconnected_distance)
    ]
    if not candidates:
        return None
    return float(min(candidates))


def _set_role(
    roles: dict[int, dict[str, Any]],
    nodes: Sequence[EvidenceNode],
    idx: int,
    role: str,
    role_priority: int,
    path_position: int,
) -> None:
    if not is_chunk(nodes, idx):
        return
    candidate = {
        "role": role,
        "role_priority": int(role_priority),
        "path_position": int(path_position),
        "node_id": node_id(nodes, idx),
    }
    current = roles.get(int(idx))
    if current is None or (
        candidate["role_priority"],
        candidate["path_position"],
        candidate["node_id"],
        int(idx),
    ) < (
        int(current["role_priority"]),
        int(current["path_position"]),
        str(current["node_id"]),
        int(idx),
    ):
        roles[int(idx)] = candidate


def build_semantic_graph_pool(
    *,
    nodes: Sequence[EvidenceNode],
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    disconnected_distance: float,
    eps: float = 1e-9,
) -> SemanticGraphPool:
    matrix_size = int(distance_matrix.shape[0])
    medoids = sorted(
        (idx for idx in dedupe_indices(selected_medoids) if 0 <= int(idx) < matrix_size),
        key=lambda idx: (node_id(nodes, idx), idx),
    )
    anchors = sorted(
        (idx for idx in dedupe_indices(query_anchors) if 0 <= int(idx) < matrix_size),
        key=lambda idx: (node_id(nodes, idx), idx),
    )
    support_tree = support_tree_nodes(
        query_anchors=anchors,
        selected_medoids=medoids,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    support_tree_chunks = {idx for idx in support_tree if 0 <= int(idx) < matrix_size and is_chunk(nodes, idx)}
    roles: dict[int, dict[str, Any]] = {}
    for medoid in medoids:
        _set_role(roles, nodes, medoid, "medoid", 0, 0)

    for anchor in anchors:
        for medoid in medoids:
            for pos, idx in enumerate(
                path_nodes(distance_matrix, anchor, medoid, nodes, disconnected_distance=disconnected_distance)
            ):
                if int(idx) in medoids:
                    continue
                _set_role(roles, nodes, idx, "anchor_medoid_path", 1, pos)

    for pos, left in enumerate(medoids):
        for right in medoids[pos + 1 :]:
            for path_pos, idx in enumerate(
                path_nodes(distance_matrix, left, right, nodes, disconnected_distance=disconnected_distance)
            ):
                if int(idx) in medoids:
                    continue
                _set_role(roles, nodes, idx, "medoid_medoid_path", 2, path_pos)

    for idx in sorted(support_tree_chunks):
        if idx not in roles:
            _set_role(roles, nodes, idx, "strict_tree", 3, 10**9)

    graph_distance_to_tree: dict[int, float] = {}
    shell1: set[int] = set()
    shell2: set[int] = set()
    for idx, node in enumerate(nodes[:matrix_size]):
        if str(getattr(node, "node_type", "chunk")) != "chunk":
            continue
        distance = _distance_to_tree(
            idx=idx,
            tree_chunks=support_tree_chunks,
            distance_matrix=distance_matrix,
            disconnected_distance=disconnected_distance,
        )
        if distance is None:
            continue
        graph_distance_to_tree[idx] = float(distance)
        if idx in support_tree_chunks:
            continue
        if abs(distance - 1.0) <= eps:
            shell1.add(idx)
            _set_role(roles, nodes, idx, "shell1", 4, 10**9)
        elif abs(distance - 2.0) <= eps:
            shell2.add(idx)

    return SemanticGraphPool(
        support_tree=support_tree,
        support_tree_chunks=support_tree_chunks,
        shell1_chunks=shell1,
        shell2_chunks=shell2,
        roles=roles,
        graph_distance_to_tree=graph_distance_to_tree,
    )


__all__ = [
    "SemanticGraphPool",
    "build_semantic_graph_pool",
    "dedupe_indices",
    "is_chunk",
    "node_id",
]
