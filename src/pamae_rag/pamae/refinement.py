from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective, assign_to_anchors


@dataclass(frozen=True)
class RefinementResult:
    anchors: list[int]
    before: ObjectiveBreakdown
    after: ObjectiveBreakdown
    accepted: bool
    history: list[float]
    cluster_sizes: list[int]


def refine_medoids_monotone(anchors: Sequence[int], candidate_indices: Sequence[int], distance_matrix: np.ndarray, rho: np.ndarray, token_costs: np.ndarray | None = None, token_weight: float = 0.0, anchor_penalty: float = 0.0, max_iters: int = 1) -> RefinementResult:
    current = list(dict.fromkeys(int(a) for a in anchors))
    candidates = sorted(set(int(i) for i in candidate_indices))
    before = anchor_objective(current, distance_matrix, rho, token_costs, token_weight, anchor_penalty)
    best_obj = before
    history = [before.total]
    accepted_any = False
    cluster_sizes: list[int] = []
    if not current:
        return RefinementResult(current, before, before, False, history, cluster_sizes)
    for _ in range(max_iters):
        assignment_pos = assign_to_anchors(current, distance_matrix)
        proposal = list(current)
        cluster_sizes = []
        for pos, old_anchor in enumerate(current):
            rows = np.where(assignment_pos == pos)[0]
            cluster_sizes.append(int(rows.size))
            if rows.size == 0:
                continue
            row_set = set(rows.tolist())
            eligible = [u for u in candidates if u in row_set]
            if not eligible:
                eligible = [old_anchor]
            best_u = old_anchor
            best_cost = float("inf")
            for u in eligible:
                if u in proposal and u != old_anchor:
                    continue
                cost = float(np.dot(rho[rows], distance_matrix[np.ix_(rows, [u])].ravel()))
                if token_costs is not None and token_weight > 0:
                    cost += float(token_weight * token_costs[u])
                if cost < best_cost:
                    best_cost = cost
                    best_u = int(u)
            proposal[pos] = best_u
        deduped: list[int] = []
        for old_anchor, new_anchor in zip(current, proposal, strict=True):
            chosen = new_anchor if new_anchor not in deduped else old_anchor
            if chosen not in deduped:
                deduped.append(chosen)
        for u in candidates:
            if len(deduped) >= len(current):
                break
            if u not in deduped:
                deduped.append(u)
        proposal = deduped[: len(current)]
        proposed_obj = anchor_objective(proposal, distance_matrix, rho, token_costs, token_weight, anchor_penalty)
        if proposed_obj.total < best_obj.total - 1e-12:
            current = proposal
            best_obj = proposed_obj
            history.append(best_obj.total)
            accepted_any = True
        else:
            history.append(best_obj.total)
            break
    return RefinementResult(current, before, best_obj, accepted_any, history, cluster_sizes)
