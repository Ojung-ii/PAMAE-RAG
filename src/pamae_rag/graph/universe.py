from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode


def select_universe_by_mass(nodes: tuple[EvidenceNode, ...] | list[EvidenceNode], max_nodes: int, min_relevance_mass: float) -> tuple[EvidenceNode, ...]:
    if not nodes:
        return tuple()
    scores = np.asarray([max(0.0, n.relevance) for n in nodes], dtype=np.float64)
    if scores.sum() <= 0:
        scores = np.ones(len(nodes), dtype=np.float64)
    order = np.argsort(-scores)
    total = float(scores.sum())
    selected: list[EvidenceNode] = []
    acc = 0.0
    for idx in order:
        selected.append(nodes[int(idx)])
        acc += float(scores[int(idx)])
        if len(selected) >= max_nodes or acc / total >= min_relevance_mass:
            break
    return tuple(selected)
