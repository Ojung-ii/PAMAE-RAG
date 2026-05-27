from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Sequence

import numpy as np

from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective

_EXACT_SEARCH_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class SearchResult:
    anchors: list[int]
    objective: ObjectiveBreakdown
    exact: bool
    num_combinations: int


def _best_exact_combo_batched(
    sample: Sequence[int],
    k: int,
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    token_costs: np.ndarray | None,
    token_weight: float,
    anchor_penalty: float,
    *,
    chunk_size: int = _EXACT_SEARCH_CHUNK_SIZE,
) -> tuple[list[int], ObjectiveBreakdown]:
    sample_arr = np.asarray(sample, dtype=np.int64)
    sample_distances = distance_matrix[np.ix_(sample_arr, sample_arr)]
    sample_rho = np.asarray(rho[sample_arr], dtype=np.float64)
    all_token_costs = None if token_costs is None else np.asarray(token_costs, dtype=np.float64)
    active_token_penalty = all_token_costs is not None and token_weight > 0.0
    k_pen = float(anchor_penalty * k)

    best_positions: np.ndarray | None = None
    best_total = float("inf")
    chunk: list[tuple[int, ...]] = []

    def evaluate_chunk(combos: list[tuple[int, ...]]) -> None:
        nonlocal best_positions, best_total
        combo_positions = np.asarray(combos, dtype=np.int64)
        nearest_distances = np.min(sample_distances[:, combo_positions], axis=2)
        coverage_values = sample_rho @ nearest_distances
        if active_token_penalty:
            assert all_token_costs is not None
            anchor_indices = sample_arr[combo_positions]
            token_penalties = float(token_weight) * np.sum(all_token_costs[anchor_indices], axis=1)
            totals = coverage_values + token_penalties + k_pen
        else:
            token_penalties = np.zeros(combo_positions.shape[0], dtype=np.float64)
            totals = coverage_values + k_pen

        local_best = int(np.argmin(totals))
        local_total = float(totals[local_best])
        if local_total < best_total:
            best_positions = np.array(combo_positions[local_best], dtype=np.int64)
            best_total = local_total

    for combo in combinations(range(len(sample)), k):
        chunk.append(combo)
        if len(chunk) >= chunk_size:
            evaluate_chunk(chunk)
            chunk = []
    if chunk:
        evaluate_chunk(chunk)

    assert best_positions is not None
    anchors = sample_arr[best_positions].astype(int).tolist()
    objective = anchor_objective(
        anchors,
        distance_matrix,
        rho,
        token_costs,
        token_weight,
        anchor_penalty,
        row_indices=sample_arr,
    )
    return anchors, objective


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
    best, best_obj = _best_exact_combo_batched(
        sample,
        k,
        distance_matrix,
        rho,
        token_costs,
        token_weight,
        anchor_penalty,
    )
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
