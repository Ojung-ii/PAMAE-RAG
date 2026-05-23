from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    text: str
    embedding: np.ndarray
    relevance: float = 1.0
    token_count: int = 0
    node_type: str = "chunk"
    is_anchor_candidate: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    query: str
    nodes: tuple[EvidenceNode, ...]
    gold_node_ids: frozenset[str] = field(default_factory=frozenset)
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    anchor_node_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    context_node_ids: tuple[str, ...]
    objective_before_refinement: float
    objective_after_refinement: float
    support_recall: float | None
    support_hit: float | None
    exact_phase1: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None

    def to_json(self) -> dict[str, Any]:
        anchor_node_ids = list(self.anchor_node_ids)
        anchor_ids = list(self.anchor_ids)
        if anchor_node_ids != anchor_ids:
            raise ValueError("anchor_node_ids and anchor_ids must match")
        return {
            "query_id": self.query_id,
            "anchor_node_ids": anchor_node_ids,
            "anchor_ids": anchor_ids,
            "context_node_ids": list(self.context_node_ids),
            "objective_before_refinement": self.objective_before_refinement,
            "objective_after_refinement": self.objective_after_refinement,
            "support_recall": self.support_recall,
            "support_hit": self.support_hit,
            "exact_phase1": self.exact_phase1,
            "diagnostics": self.diagnostics,
            "latency_ms": self.latency_ms,
        }
