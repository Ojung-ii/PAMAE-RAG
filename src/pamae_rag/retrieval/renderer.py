from __future__ import annotations

from typing import Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode


def render_context_indices(nodes: Sequence[EvidenceNode], anchors: Sequence[int], distance_matrix: np.ndarray, rho: np.ndarray, max_context_tokens: int, evidence_per_anchor: int = 2) -> list[int]:
    selected: list[int] = []
    used_tokens = 0
    def try_add(idx: int) -> None:
        nonlocal used_tokens
        idx = int(idx)
        if idx in selected:
            return
        tok = max(1, int(nodes[idx].token_count))
        if selected and used_tokens + tok > max_context_tokens:
            return
        selected.append(idx)
        used_tokens += tok
    for anchor in anchors:
        try_add(int(anchor))
        nearest = np.argsort(distance_matrix[:, int(anchor)])
        candidate_window = nearest[: max(evidence_per_anchor * 4, evidence_per_anchor + 1)]
        ranked = sorted(candidate_window, key=lambda i: (distance_matrix[int(i), int(anchor)], -rho[int(i)]))
        for idx in ranked[: evidence_per_anchor + 1]:
            try_add(int(idx))
    return selected


def render_context_text(nodes: Sequence[EvidenceNode], indices: Sequence[int]) -> str:
    blocks = []
    for rank, idx in enumerate(indices, start=1):
        node = nodes[int(idx)]
        title = node.metadata.get("title") or node.node_id
        blocks.append(f"[{rank}] {title}\nnode_id: {node.node_id}\n{node.text}")
    return "\n\n".join(blocks)
