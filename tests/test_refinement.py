import numpy as np

from pamae_rag.pamae.refinement import refine_medoids_monotone


def test_refinement_is_monotone():
    d = np.array(
        [
            [0.0, 0.2, 1.0, 1.0],
            [0.2, 0.0, 0.8, 0.9],
            [1.0, 0.8, 0.0, 0.1],
            [1.0, 0.9, 0.1, 0.0],
        ]
    )
    rho = np.array([0.4, 0.3, 0.2, 0.1])
    result = refine_medoids_monotone([1, 3], [0, 1, 2, 3], d, rho, max_iters=1)
    assert result.after.total <= result.before.total + 1e-12
