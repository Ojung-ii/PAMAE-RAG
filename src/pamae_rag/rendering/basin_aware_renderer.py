from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode


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
        tok = max(1, int(self.nodes[idx].token_count))
        if self.used_tokens + tok > self.max_context_tokens:
            return False
        if not force and self.max_context_nodes and len(self.selected) >= self.max_context_nodes:
            return False
        self.selected.append(idx)
        self.used_tokens += tok
        return True


@dataclass(frozen=True)
class BasinRenderResult:
    indices: list[int]
    diagnostics: dict[str, Any]


def _node_id(nodes: Sequence[EvidenceNode], idx: int) -> str:
    return str(nodes[int(idx)].node_id)


def _path_nodes(
    query_anchor: int,
    medoid: int,
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
    *,
    eps: float = 1e-8,
) -> list[int]:
    start = int(query_anchor)
    end = int(medoid)
    target = float(distance_matrix[start, end])
    out: list[int] = []
    for idx in range(distance_matrix.shape[0]):
        via = float(distance_matrix[start, idx]) + float(distance_matrix[idx, end])
        if abs(via - target) <= eps:
            out.append(int(idx))
    return sorted(out, key=lambda idx: (float(distance_matrix[start, idx]), _node_id(nodes, idx), int(idx)))


def _basin_medoid(
    basin: int,
    node_to_basin: dict[int, int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    nodes: Sequence[EvidenceNode],
) -> int | None:
    members = sorted(node for node, owner in node_to_basin.items() if int(owner) == int(basin))
    if not members:
        return None
    rows = np.asarray(members, dtype=np.int64)
    return min(
        members,
        key=lambda idx: (
            float(np.dot(rho[rows], distance_matrix[np.ix_(rows, [int(idx)])].ravel())),
            _node_id(nodes, int(idx)),
            int(idx),
        ),
    )


def render_basin_path_closure_indices(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    *,
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    node_to_basin: dict[int, int],
    covered_basin_masses: dict[int, float],
) -> BasinRenderResult:
    anchor_list = list(dict.fromkeys(int(anchor) for anchor in anchors))
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    medoids_by_basin: dict[int, list[int]] = {}
    for medoid in anchor_list:
        basin = int(node_to_basin.get(int(medoid), int(medoid)))
        medoids_by_basin.setdefault(basin, []).append(int(medoid))

    basin_order = sorted(
        medoids_by_basin,
        key=lambda basin: (
            -float(covered_basin_masses.get(int(basin), 0.0)),
            _node_id(nodes, int(basin)) if 0 <= int(basin) < len(nodes) else str(basin),
            int(basin),
        ),
    )
    rendered_basins: set[int] = set()
    rendered_medoids: set[int] = set()
    rendered_bridge_chunks: set[int] = set()
    rendered_basin_medoids: set[int] = set()

    for basin in basin_order:
        rendered_basins.add(int(basin))
        for medoid in sorted(medoids_by_basin[basin], key=lambda idx: (_node_id(nodes, idx), idx)):
            if budget.add(medoid, force=True):
                rendered_medoids.add(medoid)
            path = _path_nodes(int(basin), int(medoid), distance_matrix, nodes)
            path_set = set(path)
            for idx in path:
                if idx == medoid:
                    continue
                if budget.add(idx):
                    rendered_bridge_chunks.add(idx)
            basin_medoid = _basin_medoid(int(basin), node_to_basin, distance_matrix, rho, nodes)
            if basin_medoid is not None and basin_medoid not in path_set and basin_medoid != medoid:
                if budget.add(basin_medoid):
                    rendered_basin_medoids.add(basin_medoid)

    diagnostics = {
        "renderer_mode": "basin_path_closure",
        "rendered_basin_count": len(rendered_basins),
        "rendered_medoid_count": len(rendered_medoids),
        "rendered_bridge_chunk_count": len(rendered_bridge_chunks),
        "rendered_basin_medoid_count": len(rendered_basin_medoids),
    }
    return BasinRenderResult(indices=budget.selected, diagnostics=diagnostics)
