from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode


@dataclass
class _Budget:
    nodes: Sequence[EvidenceNode]
    max_context_tokens: int
    max_context_nodes: int | None
    selected: list[int]
    used_tokens: int = 0

    def add(self, idx: int, *, force: bool = False) -> None:
        idx = int(idx)
        if idx in self.selected:
            return
        tok = max(1, int(self.nodes[idx].token_count))
        if self.used_tokens + tok > self.max_context_tokens:
            return
        if not force and self.max_context_nodes and len(self.selected) >= self.max_context_nodes:
            return
        self.selected.append(idx)
        self.used_tokens += tok


def _nearest_anchor_distances(anchors: Sequence[int], distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_arr = np.asarray(list(anchors), dtype=np.int64)
    nearest_pos = np.argmin(distance_matrix[:, anchor_arr], axis=1)
    nearest_dist = distance_matrix[np.arange(distance_matrix.shape[0]), anchor_arr[nearest_pos]]
    return nearest_pos, nearest_dist


def _score_fill_order(anchors: Sequence[int], distance_matrix: np.ndarray, rho: np.ndarray, gamma: float) -> list[int]:
    if not anchors:
        return []
    _, nearest_dist = _nearest_anchor_distances(anchors, distance_matrix)
    return sorted(
        range(distance_matrix.shape[0]),
        key=lambda i: (
            -(float(rho[i]) - float(gamma) * float(nearest_dist[i])),
            -float(rho[i]),
            float(nearest_dist[i]),
            int(i),
        ),
    )


def _dedupe(indices: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(idx) for idx in indices))


def _render_old_order(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    evidence_per_anchor: int,
    gamma: float,
) -> list[int]:
    anchor_list = _dedupe(anchors)
    order: list[int] = [*anchor_list]
    for anchor in anchor_list:
        nearest = np.argsort(distance_matrix[:, int(anchor)])
        candidate_window = nearest[: max(evidence_per_anchor * 4, evidence_per_anchor + 1)]
        ranked = sorted(
            candidate_window,
            key=lambda i: (float(distance_matrix[int(i), int(anchor)]), -float(rho[int(i)]), int(i)),
        )
        order.extend(int(idx) for idx in ranked[: evidence_per_anchor + 1])
    order.extend(_score_fill_order(anchor_list, distance_matrix, rho, gamma))
    return _dedupe(order)


def _render_nearest_order(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    evidence_per_anchor: int,
    gamma: float,
) -> list[int]:
    anchor_list = _dedupe(anchors)
    order: list[int] = [*anchor_list]
    for anchor in anchor_list:
        nearest = sorted(
            range(len(nodes)),
            key=lambda i: (float(distance_matrix[int(i), int(anchor)]), -float(rho[int(i)]), int(i)),
        )
        order.extend(int(idx) for idx in nearest[: evidence_per_anchor + 1])
    order.extend(_score_fill_order(anchor_list, distance_matrix, rho, gamma))
    return _dedupe(order)


def _render_cell_top_rho_order(
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    gamma: float,
) -> list[int]:
    anchor_list = _dedupe(anchors)
    if not anchor_list:
        return []
    order: list[int] = [*anchor_list]
    nearest_pos, nearest_dist = _nearest_anchor_distances(anchor_list, distance_matrix)
    for pos in range(len(anchor_list)):
        rows = np.where(nearest_pos == pos)[0]
        if rows.size == 0:
            continue
        best = max(
            (int(i) for i in rows.tolist()),
            key=lambda i: (float(rho[i]), -float(nearest_dist[i]), -int(i)),
        )
        order.append(best)
    order.extend(_score_fill_order(anchor_list, distance_matrix, rho, gamma))
    return _dedupe(order)


def _render_old(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    evidence_per_anchor: int,
    gamma: float,
) -> list[int]:
    anchor_list = list(dict.fromkeys(int(a) for a in anchors))
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    for anchor in anchors:
        budget.add(int(anchor), force=True)
    for anchor in anchor_list:
        nearest = np.argsort(distance_matrix[:, int(anchor)])
        candidate_window = nearest[: max(evidence_per_anchor * 4, evidence_per_anchor + 1)]
        ranked = sorted(
            candidate_window,
            key=lambda i: (float(distance_matrix[int(i), int(anchor)]), -float(rho[int(i)]), int(i)),
        )
        for idx in ranked[: evidence_per_anchor + 1]:
            budget.add(int(idx))
    for idx in _score_fill_order(anchor_list, distance_matrix, rho, gamma):
        budget.add(int(idx))
    return budget.selected


def _render_anchor_only(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    max_context_tokens: int,
    max_context_nodes: int | None,
) -> list[int]:
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    for anchor in anchors:
        budget.add(int(anchor), force=True)
    return budget.selected


def _render_nearest(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    evidence_per_anchor: int,
    rho: np.ndarray,
    gamma: float,
) -> list[int]:
    anchor_list = list(dict.fromkeys(int(a) for a in anchors))
    selected = _render_anchor_only(nodes, anchor_list, max_context_tokens, max_context_nodes)
    budget = _Budget(
        nodes,
        max_context_tokens,
        max_context_nodes,
        selected,
        sum(max(1, int(nodes[i].token_count)) for i in selected),
    )
    for anchor in anchor_list:
        nearest = sorted(
            range(len(nodes)),
            key=lambda i: (float(distance_matrix[int(i), int(anchor)]), -float(rho[int(i)]), int(i)),
        )
        for idx in nearest[: evidence_per_anchor + 1]:
            budget.add(int(idx))
    for idx in _score_fill_order(anchor_list, distance_matrix, rho, gamma):
        budget.add(int(idx))
    return budget.selected


def _render_global_top_rho(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    gamma: float,
) -> list[int]:
    anchor_list = list(dict.fromkeys(int(a) for a in anchors))
    selected = _render_anchor_only(nodes, anchor_list, max_context_tokens, max_context_nodes)
    budget = _Budget(
        nodes,
        max_context_tokens,
        max_context_nodes,
        selected,
        sum(max(1, int(nodes[i].token_count)) for i in selected),
    )
    for idx in _score_fill_order(anchor_list, distance_matrix, rho, gamma):
        budget.add(int(idx))
    return budget.selected


def _render_cell_top_rho(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    gamma: float,
) -> list[int]:
    anchor_list = list(dict.fromkeys(int(a) for a in anchors))
    if not anchor_list:
        return []
    selected = _render_anchor_only(nodes, anchor_list, max_context_tokens, max_context_nodes)
    budget = _Budget(
        nodes,
        max_context_tokens,
        max_context_nodes,
        selected,
        sum(max(1, int(nodes[i].token_count)) for i in selected),
    )

    nearest_pos, nearest_dist = _nearest_anchor_distances(anchor_list, distance_matrix)

    for pos in range(len(anchor_list)):
        rows = np.where(nearest_pos == pos)[0]
        if rows.size == 0:
            continue
        best = max(
            (int(i) for i in rows.tolist()),
            key=lambda i: (float(rho[i]), -float(nearest_dist[i]), -int(i)),
        )
        budget.add(best)

    for idx in _score_fill_order(anchor_list, distance_matrix, rho, gamma):
        budget.add(int(idx))
    return budget.selected


def render_context_indices(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None = None,
    evidence_per_anchor: int = 2,
    *,
    renderer: str = "old",
    gamma: float = 0.0,
) -> list[int]:
    if renderer == "old":
        return _render_old(
            nodes,
            anchors,
            distance_matrix,
            rho,
            max_context_tokens,
            max_context_nodes,
            evidence_per_anchor,
            gamma,
        )
    if renderer == "anchor_only":
        return _render_anchor_only(nodes, anchors, max_context_tokens, max_context_nodes)
    if renderer == "nearest":
        return _render_nearest(
            nodes,
            anchors,
            distance_matrix,
            max_context_tokens,
            max_context_nodes,
            evidence_per_anchor,
            rho,
            gamma,
        )
    if renderer == "cell_top_rho":
        return _render_cell_top_rho(
            nodes,
            anchors,
            distance_matrix,
            rho,
            max_context_tokens,
            max_context_nodes,
            gamma,
        )
    if renderer == "global_top_rho":
        return _render_global_top_rho(
            nodes,
            anchors,
            distance_matrix,
            rho,
            max_context_tokens,
            max_context_nodes,
            gamma,
        )
    raise ValueError(f"Unknown renderer: {renderer}")


def render_context_order_indices(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    evidence_per_anchor: int = 2,
    *,
    renderer: str = "old",
    gamma: float = 0.0,
) -> list[int]:
    if renderer == "old":
        return _render_old_order(nodes, anchors, distance_matrix, rho, evidence_per_anchor, gamma)
    if renderer == "anchor_only":
        return _dedupe(anchors)
    if renderer == "nearest":
        return _render_nearest_order(nodes, anchors, distance_matrix, rho, evidence_per_anchor, gamma)
    if renderer == "cell_top_rho":
        return _render_cell_top_rho_order(anchors, distance_matrix, rho, gamma)
    if renderer == "global_top_rho":
        return _dedupe([*anchors, *_score_fill_order(anchors, distance_matrix, rho, gamma)])
    raise ValueError(f"Unknown renderer: {renderer}")


def render_context_text(nodes: Sequence[EvidenceNode], indices: Sequence[int]) -> str:
    blocks = []
    for rank, idx in enumerate(indices, start=1):
        node = nodes[int(idx)]
        title = node.metadata.get("title") or node.node_id
        blocks.append(f"[{rank}] {title}\nnode_id: {node.node_id}\n{node.text}")
    return "\n\n".join(blocks)
