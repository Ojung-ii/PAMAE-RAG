import numpy as np

from pamae_rag.objective.anchor_objective import anchor_objective


def test_anchor_objective_prefers_covering_mass():
    d = np.array(
        [
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    rho = np.array([0.49, 0.49, 0.02])
    assert anchor_objective([0, 1], d, rho).total < anchor_objective([0, 2], d, rho).total
