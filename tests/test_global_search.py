from itertools import combinations

import numpy as np

from pamae_rag.objective.anchor_objective import anchor_objective
from pamae_rag.pamae.global_search import exact_k_medoids_on_sample


def test_exact_global_search_runs():
    d = np.array(
        [
            [0.0, 0.1, 0.9, 1.0],
            [0.1, 0.0, 0.8, 0.9],
            [0.9, 0.8, 0.0, 0.1],
            [1.0, 0.9, 0.1, 0.0],
        ]
    )
    rho = np.ones(4) / 4
    result = exact_k_medoids_on_sample([0, 1, 2, 3], 2, d, rho)
    assert result.exact
    assert len(result.anchors) == 2
    assert result.num_combinations == 6


def test_exact_global_search_matches_bruteforce_with_penalties():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 5))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = np.clip(1.0 - x @ x.T, 0.0, 2.0)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    rho = rng.random(8)
    rho = rho / rho.sum()
    token_costs = rng.random(8)
    sample = [0, 2, 3, 4, 6, 7]
    k = 3

    expected_anchors = None
    expected_objective = None
    for combo in combinations(sample, k):
        obj = anchor_objective(
            combo,
            d,
            rho,
            token_costs=token_costs,
            token_weight=0.03,
            anchor_penalty=0.02,
            row_indices=sample,
        )
        if expected_objective is None or obj.total < expected_objective.total:
            expected_anchors = list(combo)
            expected_objective = obj

    result = exact_k_medoids_on_sample(
        sample,
        k,
        d,
        rho,
        token_costs=token_costs,
        token_weight=0.03,
        anchor_penalty=0.02,
    )
    assert result.anchors == expected_anchors
    assert result.objective == expected_objective
