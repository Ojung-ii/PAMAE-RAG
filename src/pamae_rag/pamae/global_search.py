from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Sequence

import numpy as np

from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective


@dataclass(frozen=True)
class SearchResult:
    anchors: list[int]
    objective: ObjectiveBreakdown
    exact: bool
    num_combinations: int


def exact_k_medoids_on_sample(sample_indices: Sequence[int], k: int, distance_matrix: np.ndarray, rho: np.ndarray, token_costs: np.ndarray | None = None, token_weight: float = 0.0, anchor_penalty: float = 0.0, max_combinations: int = 1_000_000, require_exact: bool = True) -> SearchResult:
    sample = sorted(set(int(i) for i in sample_indices))
    if len(sample) < k:
        raise ValueError(f"sample has {len(sample)} candidates but k={k}")
    count = comb(len(sample), k)
    if count > max_combinations:
        if require_exact:
            raise RuntimeError(f"Exact sample search budget exceeded: C({len(sample)},{k})={count} > {max_combinations}. Reduce sample size or raise max_combinations.")
        anchors = greedy_k_medoids_on_sample(sample, k, distance_matrix, rho, token_costs, token_weight, anchor_penalty)
        obj = anchor_objective(anchors, distance_matrix, rho, token_costs, token_weight, anchor_penalty, row_indices=sample)
        return SearchResult(anchors=anchors, objective=obj, exact=False, num_combinations=count)
    best_obj: ObjectiveBreakdown | None = None
    best: list[int] | None = None
    rows = np.asarray(sample, dtype=np.int64)
    for combo in combinations(sample, k):
        obj = anchor_objective(combo, distance_matrix, rho, token_costs, token_weight, anchor_penalty, row_indices=rows)
        if best_obj is None or obj.total < best_obj.total:
            best_obj = obj
            best = list(combo)
    assert best is not None and best_obj is not None
    return SearchResult(anchors=best, objective=best_obj, exact=True, num_combinations=count)


def greedy_k_medoids_on_sample(sample: Sequence[int], k: int, distance_matrix: np.ndarray, rho: np.ndarray, token_costs: np.ndarray | None = None, token_weight: float = 0.0, anchor_penalty: float = 0.0) -> list[int]:
    selected: list[int] = []
    remaining = set(int(i) for i in sample)
    rows = list(int(i) for i in sample)
    for _ in range(k):
        best_u: int | None = None
        best_val = float("inf")
        for u in sorted(remaining):
            anchors = selected + [u]
            val = anchor_objective(anchors, distance_matrix, rho, token_costs, token_weight, anchor_penalty, row_indices=rows).total
            if val < best_val:
                best_val = val
                best_u = u
        assert best_u is not None
        selected.append(best_u)
        remaining.remove(best_u)
    return sorted(selected)
