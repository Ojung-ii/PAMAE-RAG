from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective
from pamae_rag.pamae.global_search import SearchResult


@dataclass(frozen=True)
class SelectionResult:
    anchors: list[int]
    full_objective: ObjectiveBreakdown
    sample_objective: ObjectiveBreakdown
    exact: bool
    sample_index: int


def select_by_full_objective(results: Sequence[SearchResult], distance_matrix: np.ndarray, rho: np.ndarray, token_costs: np.ndarray | None = None, token_weight: float = 0.0, anchor_penalty: float = 0.0) -> SelectionResult:
    if not results:
        raise ValueError("No Phase I search results to select from")
    best: SelectionResult | None = None
    for i, result in enumerate(results):
        obj = anchor_objective(result.anchors, distance_matrix, rho, token_costs, token_weight, anchor_penalty)
        current = SelectionResult(result.anchors, obj, result.objective, result.exact, i)
        if best is None or current.full_objective.total < best.full_objective.total:
            best = current
    assert best is not None
    return best
