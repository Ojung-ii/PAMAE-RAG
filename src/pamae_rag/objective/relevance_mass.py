from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode


def relevance_mass(nodes: tuple[EvidenceNode, ...] | list[EvidenceNode], smoothing: float = 1e-8) -> np.ndarray:
    if not nodes:
        raise ValueError("nodes must not be empty")
    scores = np.asarray([max(0.0, n.relevance) for n in nodes], dtype=np.float64) + smoothing
    total = float(scores.sum())
    if total <= 0:
        return np.full(len(nodes), 1.0 / len(nodes), dtype=np.float64)
    return scores / total
