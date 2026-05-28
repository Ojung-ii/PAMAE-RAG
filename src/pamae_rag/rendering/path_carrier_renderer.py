from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.diagnostics.path_realizability import path_nodes, support_tree_nodes

METRIC_PATH_CARRIER = "metric_path_carrier"
METRIC_PATH_CARRIER_NO_MEDOIDS = "metric_path_carrier_no_medoids"
METRIC_PATH_CARRIER_MEDOIDS_FIRST = "metric_path_carrier_medoids_first"
PATH_CARRIER_RENDERERS = {
    METRIC_PATH_CARRIER,
    METRIC_PATH_CARRIER_NO_MEDOIDS,
    METRIC_PATH_CARRIER_MEDOIDS_FIRST,
}


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


@dataclass(frozen=True)
class PathCarrierRenderResult:
    indices: list[int]
    diagnostics: dict[str, Any]


def _node_id(nodes: Sequence[EvidenceNode], idx: int) -> str:
    return str(nodes[int(idx)].node_id)


def _is_chunk(nodes: Sequence[EvidenceNode], idx: int) -> bool:
    return str(getattr(nodes[int(idx)], "node_type", "chunk")) == "chunk"


def _dedupe(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _path_members(
    *,
    start: int,
    end: int,
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
    disconnected_distance: float,
) -> list[int]:
    return path_nodes(
        distance_matrix,
        int(start),
        int(end),
        nodes,
        disconnected_distance=disconnected_distance,
    )


def _carrier_roles(
    *,
    query_anchors: Sequence[int],
    selected_medoids: Sequence[int],
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
    disconnected_distance: float,
) -> dict[int, dict[str, Any]]:
    anchors = sorted(_dedupe(query_anchors), key=lambda idx: (_node_id(nodes, idx), idx))
    medoids = sorted(_dedupe(selected_medoids), key=lambda idx: (_node_id(nodes, idx), idx))
    roles: dict[int, dict[str, Any]] = {}

    def add(idx: int, role: str, priority: int, path_position: int) -> None:
        idx = int(idx)
        if not _is_chunk(nodes, idx):
            return
        current = roles.get(idx)
        candidate = {
            "role": role,
            "role_priority": int(priority),
            "path_position": int(path_position),
            "node_id": _node_id(nodes, idx),
        }
        if current is None or (
            candidate["role_priority"],
            candidate["path_position"],
            candidate["node_id"],
        ) < (
            int(current["role_priority"]),
            int(current["path_position"]),
            str(current["node_id"]),
        ):
            roles[idx] = candidate

    for medoid in medoids:
        add(medoid, "medoid", 0, 0)

    for anchor in anchors:
        for medoid in medoids:
            for pos, idx in enumerate(
                _path_members(
                    start=anchor,
                    end=medoid,
                    distance_matrix=distance_matrix,
                    nodes=nodes,
                    disconnected_distance=disconnected_distance,
                )
            ):
                if int(idx) in medoids:
                    continue
                add(int(idx), "anchor_medoid_path", 1, pos)

    for left_pos, left in enumerate(medoids):
        for right in medoids[left_pos + 1 :]:
            for pos, idx in enumerate(
                _path_members(
                    start=left,
                    end=right,
                    distance_matrix=distance_matrix,
                    nodes=nodes,
                    disconnected_distance=disconnected_distance,
                )
            ):
                if int(idx) in medoids:
                    continue
                add(int(idx), "medoid_medoid_path", 2, pos)
    return roles


def render_metric_path_carrier_indices(
    *,
    nodes: Sequence[EvidenceNode],
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    disconnected_distance: float,
    renderer_mode: str = METRIC_PATH_CARRIER,
) -> PathCarrierRenderResult:
    if renderer_mode not in PATH_CARRIER_RENDERERS:
        raise ValueError(f"Unknown path carrier renderer: {renderer_mode}")
    medoids = _dedupe(selected_medoids)
    anchors = _dedupe(query_anchors)
    roles = _carrier_roles(
        query_anchors=anchors,
        selected_medoids=medoids,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    support_tree = support_tree_nodes(
        query_anchors=anchors,
        selected_medoids=medoids,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    support_tree_chunks = {idx for idx in support_tree if _is_chunk(nodes, idx)}
    include_medoids = renderer_mode != METRIC_PATH_CARRIER_NO_MEDOIDS
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    rendered_by_role = {"medoid": set(), "anchor_medoid_path": set(), "medoid_medoid_path": set()}
    cutoff_by_role = {"medoid": set(), "anchor_medoid_path": set(), "medoid_medoid_path": set()}
    ordered = sorted(
        roles,
        key=lambda idx: (
            int(roles[idx]["role_priority"]),
            int(roles[idx]["path_position"]),
            str(roles[idx]["node_id"]),
            int(idx),
        ),
    )
    for idx in ordered:
        role = str(roles[idx]["role"])
        if role == "medoid" and not include_medoids:
            continue
        added = budget.add(int(idx), force=bool(role == "medoid" and include_medoids))
        if added:
            rendered_by_role[role].add(int(idx))
        else:
            cutoff_by_role[role].add(int(idx))

    diagnostics = {
        "renderer_mode": renderer_mode,
        "metric_closure": "SPClosure(query_anchors + refined_medoids)",
        "metric_distance_only": True,
        "uses_score_mixing": False,
        "uses_rho_for_completion": False,
        "support_tree_chunk_count": len(support_tree_chunks),
        "support_tree_chunk_ids": [
            _node_id(nodes, idx) for idx in sorted(support_tree_chunks, key=lambda i: _node_id(nodes, i))
        ],
        "rendered_medoid_count": len(rendered_by_role["medoid"]),
        "rendered_anchor_medoid_path_chunk_count": len(rendered_by_role["anchor_medoid_path"]),
        "rendered_medoid_medoid_path_chunk_count": len(rendered_by_role["medoid_medoid_path"]),
        "budget_cutoff_count": sum(len(values) for values in cutoff_by_role.values()),
        "budget_cutoff_node_ids": [
            _node_id(nodes, idx)
            for role in ("medoid", "anchor_medoid_path", "medoid_medoid_path")
            for idx in sorted(cutoff_by_role[role], key=lambda i: _node_id(nodes, i))
        ],
        "rendered_path_role_node_ids": {
            role: [_node_id(nodes, idx) for idx in sorted(values, key=lambda i: _node_id(nodes, i))]
            for role, values in rendered_by_role.items()
        },
        "path_carrier_order_node_ids": [_node_id(nodes, idx) for idx in ordered],
        "context_tokens": int(sum(max(1, int(nodes[idx].token_count)) for idx in budget.selected)),
    }
    return PathCarrierRenderResult(indices=budget.selected, diagnostics=diagnostics)


__all__ = [
    "METRIC_PATH_CARRIER",
    "METRIC_PATH_CARRIER_MEDOIDS_FIRST",
    "METRIC_PATH_CARRIER_NO_MEDOIDS",
    "PATH_CARRIER_RENDERERS",
    "PathCarrierRenderResult",
    "render_metric_path_carrier_indices",
]
