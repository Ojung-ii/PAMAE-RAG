from __future__ import annotations

from typing import Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode


def _append_with_budget(
    selected: list[int],
    nodes: Sequence[EvidenceNode],
    idx: int,
    used_tokens: int,
    max_context_tokens: int,
    *,
    force: bool = False,
) -> int:
    idx = int(idx)
    if idx in selected:
        return used_tokens
    tok = max(1, int(nodes[idx].token_count))
    if not force and selected and used_tokens + tok > max_context_tokens:
        return used_tokens
    selected.append(idx)
    return used_tokens + tok


def _render_old(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    evidence_per_anchor: int,
) -> list[int]:
    selected: list[int] = []
    used_tokens = 0
    for anchor in anchors:
        used_tokens = _append_with_budget(
            selected, nodes, int(anchor), used_tokens, max_context_tokens, force=True
        )
        nearest = np.argsort(distance_matrix[:, int(anchor)])
        candidate_window = nearest[: max(evidence_per_anchor * 4, evidence_per_anchor + 1)]
        ranked = sorted(candidate_window, key=lambda i: (distance_matrix[int(i), int(anchor)], -rho[int(i)]))
        for idx in ranked[: evidence_per_anchor + 1]:
            used_tokens = _append_with_budget(selected, nodes, int(idx), used_tokens, max_context_tokens)
    return selected


def _render_anchor_only(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    max_context_tokens: int,
) -> list[int]:
    selected: list[int] = []
    used_tokens = 0
    for anchor in anchors:
        used_tokens = _append_with_budget(
            selected, nodes, int(anchor), used_tokens, max_context_tokens, force=True
        )
    return selected


def _render_nearest(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    evidence_per_anchor: int,
) -> list[int]:
    selected = _render_anchor_only(nodes, anchors, max_context_tokens)
    used_tokens = sum(max(1, int(nodes[i].token_count)) for i in selected)
    for anchor in anchors:
        nearest = np.argsort(distance_matrix[:, int(anchor)])
        for idx in nearest[: evidence_per_anchor + 1]:
            used_tokens = _append_with_budget(selected, nodes, int(idx), used_tokens, max_context_tokens)
    return selected


def _render_global_top_rho(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    rho: np.ndarray,
    max_context_tokens: int,
) -> list[int]:
    selected = _render_anchor_only(nodes, anchors, max_context_tokens)
    used_tokens = sum(max(1, int(nodes[i].token_count)) for i in selected)
    for idx in np.argsort(-rho):
        used_tokens = _append_with_budget(selected, nodes, int(idx), used_tokens, max_context_tokens)
    return selected


def _render_cell_top_rho(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    gamma: float,
) -> list[int]:
    anchor_list = list(dict.fromkeys(int(a) for a in anchors))
    if not anchor_list:
        return []
    selected = _render_anchor_only(nodes, anchor_list, max_context_tokens)
    used_tokens = sum(max(1, int(nodes[i].token_count)) for i in selected)

    anchor_arr = np.asarray(anchor_list, dtype=np.int64)
    nearest_pos = np.argmin(distance_matrix[:, anchor_arr], axis=1)
    nearest_dist = distance_matrix[np.arange(len(nodes)), anchor_arr[nearest_pos]]

    for pos in range(len(anchor_list)):
        rows = np.where(nearest_pos == pos)[0]
        if rows.size == 0:
            continue
        best = max((int(i) for i in rows.tolist()), key=lambda i: (float(rho[i]), -float(nearest_dist[i])))
        used_tokens = _append_with_budget(selected, nodes, best, used_tokens, max_context_tokens)

    ranked = sorted(
        range(len(nodes)),
        key=lambda i: (
            -(float(rho[i]) - float(gamma) * float(nearest_dist[i])),
            -float(rho[i]),
            float(nearest_dist[i]),
        ),
    )
    for idx in ranked:
        used_tokens = _append_with_budget(selected, nodes, idx, used_tokens, max_context_tokens)
    return selected


def render_context_indices(
    nodes: Sequence[EvidenceNode],
    anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    evidence_per_anchor: int = 2,
    *,
    renderer: str = "old",
    gamma: float = 0.0,
) -> list[int]:
    if renderer == "old":
        return _render_old(nodes, anchors, distance_matrix, rho, max_context_tokens, evidence_per_anchor)
    if renderer == "anchor_only":
        return _render_anchor_only(nodes, anchors, max_context_tokens)
    if renderer == "nearest":
        return _render_nearest(nodes, anchors, distance_matrix, max_context_tokens, evidence_per_anchor)
    if renderer == "cell_top_rho":
        return _render_cell_top_rho(nodes, anchors, distance_matrix, rho, max_context_tokens, gamma)
    if renderer == "global_top_rho":
        return _render_global_top_rho(nodes, anchors, rho, max_context_tokens)
    raise ValueError(f"Unknown renderer: {renderer}")


def render_context_text(nodes: Sequence[EvidenceNode], indices: Sequence[int]) -> str:
    blocks = []
    for rank, idx in enumerate(indices, start=1):
        node = nodes[int(idx)]
        title = node.metadata.get("title") or node.node_id
        blocks.append(f"[{rank}] {title}\nnode_id: {node.node_id}\n{node.text}")
    return "\n\n".join(blocks)
