from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, Sequence

import numpy as np

from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective
from pamae_rag.pamae.global_search import SearchResult


@dataclass(frozen=True)
class BasinPreservingSelectionResult:
    anchors: list[int]
    full_objective: ObjectiveBreakdown
    sample_objective: ObjectiveBreakdown
    exact: bool
    sample_index: int
    diagnostics: dict[str, Any]
    node_to_basin: dict[int, int]
    covered_basin_ids: tuple[int, ...]


def query_anchor_indices(candidates: Sequence[int], rho: np.ndarray, k: int) -> list[int]:
    ranked = sorted(
        (int(idx) for idx in candidates),
        key=lambda idx: (-float(rho[int(idx)]), int(idx)),
    )
    return ranked[: max(1, min(k, len(ranked)))]


def assign_query_basins(
    candidates: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    node_ids: Sequence[str] | None = None,
) -> dict[int, int]:
    anchors = [int(anchor) for anchor in query_anchors]
    if not anchors:
        return {}
    ids = list(node_ids) if node_ids is not None else [str(i) for i in range(distance_matrix.shape[0])]
    out: dict[int, int] = {}
    for idx in sorted(int(value) for value in candidates):
        best_anchor = min(
            anchors,
            key=lambda anchor: (
                float(distance_matrix[int(idx), int(anchor)]),
                str(ids[int(anchor)]),
                int(anchor),
            ),
        )
        out[idx] = int(best_anchor)
    return out


def basin_masses(node_to_basin: dict[int, int], rho: np.ndarray) -> dict[int, float]:
    masses: dict[int, float] = defaultdict(float)
    for node, basin in node_to_basin.items():
        masses[int(basin)] += float(rho[int(node)])
    return {basin: float(masses[basin]) for basin in sorted(masses)}


def eligible_basins(masses: dict[int, float], sampling_budget: int, tau: float = 1.0) -> tuple[int, ...]:
    if sampling_budget <= 0:
        return tuple()
    return tuple(
        basin
        for basin, mass in sorted(masses.items())
        if float(sampling_budget) * float(mass) >= float(tau)
    )


def _nodes_by_basin(node_to_basin: dict[int, int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for node, basin in node_to_basin.items():
        out[int(basin)].append(int(node))
    return {basin: sorted(nodes) for basin, nodes in out.items()}


def _within_basin_medoid(
    nodes: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    node_ids: Sequence[str] | None,
) -> int:
    ids = list(node_ids) if node_ids is not None else [str(i) for i in range(distance_matrix.shape[0])]
    rows = np.asarray([int(node) for node in nodes], dtype=np.int64)
    best = min(
        (int(node) for node in nodes),
        key=lambda node: (
            float(np.dot(rho[rows], distance_matrix[np.ix_(rows, [int(node)])].ravel())),
            str(ids[int(node)]),
            int(node),
        ),
    )
    return int(best)


def _covered_basins(anchors: Sequence[int], node_to_basin: dict[int, int], eligible: set[int]) -> tuple[int, ...]:
    covered = {
        int(node_to_basin[int(anchor)])
        for anchor in anchors
        if int(anchor) in node_to_basin and int(node_to_basin[int(anchor)]) in eligible
    }
    return tuple(sorted(covered))


def _lex_key(
    anchors: Sequence[int],
    *,
    objective: ObjectiveBreakdown,
    node_to_basin: dict[int, int],
    eligible: set[int],
    masses: dict[int, float],
) -> tuple[Any, ...]:
    covered = _covered_basins(anchors, node_to_basin, eligible)
    covered_mass = sum(float(masses.get(basin, 0.0)) for basin in covered)
    return (
        len(covered),
        covered_mass,
        -float(objective.coverage),
        tuple(-int(anchor) for anchor in sorted(int(anchor) for anchor in anchors)),
    )


def _reduced_candidates(
    results: Sequence[SearchResult],
    *,
    k: int,
    candidates: Sequence[int],
    node_to_basin: dict[int, int],
    eligible: tuple[int, ...],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    node_ids: Sequence[str] | None,
) -> list[int]:
    pool = {int(anchor) for result in results for anchor in result.anchors}
    by_basin = _nodes_by_basin(node_to_basin)
    for basin in eligible:
        nodes = by_basin.get(int(basin), [])
        if nodes:
            pool.add(_within_basin_medoid(nodes, distance_matrix, rho, node_ids))
    for idx in query_anchor_indices(candidates, rho, k):
        pool.add(int(idx))
    if len(pool) < k:
        for idx in sorted((int(value) for value in candidates), key=lambda i: (-float(rho[int(i)]), int(i))):
            pool.add(int(idx))
            if len(pool) >= k:
                break
    return sorted(pool)


def _best_exact(
    pool: Sequence[int],
    k: int,
    *,
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    token_costs: np.ndarray | None,
    token_weight: float,
    anchor_penalty: float,
    node_to_basin: dict[int, int],
    eligible: set[int],
    masses: dict[int, float],
) -> tuple[list[int], ObjectiveBreakdown]:
    best_combo: tuple[int, ...] | None = None
    best_obj: ObjectiveBreakdown | None = None
    best_key: tuple[Any, ...] | None = None
    for combo in combinations(sorted(int(value) for value in pool), k):
        obj = anchor_objective(
            combo,
            distance_matrix=distance_matrix,
            rho=rho,
            token_costs=token_costs,
            token_weight=token_weight,
            anchor_penalty=anchor_penalty,
        )
        key = _lex_key(combo, objective=obj, node_to_basin=node_to_basin, eligible=eligible, masses=masses)
        if best_key is None or key > best_key:
            best_key = key
            best_combo = combo
            best_obj = obj
    assert best_combo is not None and best_obj is not None
    return list(best_combo), best_obj


def _best_greedy(
    pool: Sequence[int],
    k: int,
    *,
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    token_costs: np.ndarray | None,
    token_weight: float,
    anchor_penalty: float,
    node_to_basin: dict[int, int],
    eligible: set[int],
    masses: dict[int, float],
) -> tuple[list[int], ObjectiveBreakdown]:
    selected: list[int] = []
    remaining = set(int(value) for value in pool)
    while len(selected) < k and remaining:
        best_anchor: int | None = None
        best_obj: ObjectiveBreakdown | None = None
        best_key: tuple[Any, ...] | None = None
        for candidate in sorted(remaining):
            proposal = [*selected, int(candidate)]
            obj = anchor_objective(
                proposal,
                distance_matrix=distance_matrix,
                rho=rho,
                token_costs=token_costs,
                token_weight=token_weight,
                anchor_penalty=anchor_penalty,
            )
            key = _lex_key(proposal, objective=obj, node_to_basin=node_to_basin, eligible=eligible, masses=masses)
            if best_key is None or key > best_key:
                best_key = key
                best_anchor = int(candidate)
                best_obj = obj
        assert best_anchor is not None and best_obj is not None
        selected.append(best_anchor)
        remaining.remove(best_anchor)
    obj = anchor_objective(
        selected,
        distance_matrix=distance_matrix,
        rho=rho,
        token_costs=token_costs,
        token_weight=token_weight,
        anchor_penalty=anchor_penalty,
    )
    return sorted(selected), obj


def select_basin_preserving_medoids(
    results: Sequence[SearchResult],
    *,
    candidates: Sequence[int],
    k: int,
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    token_costs: np.ndarray | None,
    token_weight: float,
    anchor_penalty: float,
    sample_sizes: Sequence[int],
    max_combinations: int,
    node_ids: Sequence[str] | None = None,
    tau: float = 1.0,
) -> BasinPreservingSelectionResult:
    if not results:
        raise ValueError("No Phase I search results to select from")
    candidates = list(dict.fromkeys(int(candidate) for candidate in candidates))
    if len(candidates) < k:
        raise ValueError(f"candidate pool has {len(candidates)} candidates but k={k}")
    query_anchors = query_anchor_indices(candidates, rho, k)
    node_to_basin = assign_query_basins(candidates, query_anchors, distance_matrix, node_ids)
    masses = basin_masses(node_to_basin, rho)
    sampling_budget = int(sum(int(size) for size in sample_sizes))
    eligible = eligible_basins(masses, sampling_budget, tau=tau)
    eligible_set = set(eligible)
    pool = _reduced_candidates(
        results,
        k=k,
        candidates=candidates,
        node_to_basin=node_to_basin,
        eligible=eligible,
        distance_matrix=distance_matrix,
        rho=rho,
        node_ids=node_ids,
    )

    exact = comb(len(pool), k) <= int(max_combinations)
    if exact:
        anchors, objective = _best_exact(
            pool,
            k,
            distance_matrix=distance_matrix,
            rho=rho,
            token_costs=token_costs,
            token_weight=token_weight,
            anchor_penalty=anchor_penalty,
            node_to_basin=node_to_basin,
            eligible=eligible_set,
            masses=masses,
        )
    else:
        anchors, objective = _best_greedy(
            pool,
            k,
            distance_matrix=distance_matrix,
            rho=rho,
            token_costs=token_costs,
            token_weight=token_weight,
            anchor_penalty=anchor_penalty,
            node_to_basin=node_to_basin,
            eligible=eligible_set,
            masses=masses,
        )
    covered = _covered_basins(anchors, node_to_basin, eligible_set)
    covered_mass = sum(float(masses.get(basin, 0.0)) for basin in covered)
    diagnostics = {
        "selection_mode": "basin_preserving_medoids",
        "basin_count": len(masses),
        "eligible_basin_count": len(eligible),
        "covered_basin_count": len(covered),
        "covered_basin_mass": float(covered_mass),
        "phi_selected": float(objective.coverage),
        "basin_masses": {str(key): float(value) for key, value in sorted(masses.items())},
        "eligible_basin_ids": [str(value) for value in eligible],
        "covered_basin_ids": [str(value) for value in covered],
        "query_anchor_node_ids": (
            [str(node_ids[int(idx)]) for idx in query_anchors] if node_ids is not None else [str(idx) for idx in query_anchors]
        ),
        "reduced_candidate_count": len(pool),
        "basin_selection_exact": exact,
        "basin_sampling_budget": sampling_budget,
        "basin_min_expected_samples": float(tau),
    }
    return BasinPreservingSelectionResult(
        anchors=anchors,
        full_objective=objective,
        sample_objective=objective,
        exact=exact and all(result.exact for result in results),
        sample_index=-1,
        diagnostics=diagnostics,
        node_to_basin={int(k): int(v) for k, v in node_to_basin.items()},
        covered_basin_ids=covered,
    )


def gold_ids_in_selected_basins(
    *,
    gold_node_ids: set[str] | frozenset[str],
    nodes: Sequence[Any],
    node_to_basin: dict[int, int],
    covered_basin_ids: Sequence[int],
) -> list[str]:
    covered = {int(value) for value in covered_basin_ids}
    gold = {str(value) for value in gold_node_ids}
    out: list[str] = []
    for idx, node in enumerate(nodes):
        if str(node.node_id) in gold and int(node_to_basin.get(int(idx), -1)) in covered:
            out.append(str(node.node_id))
    return sorted(out)
