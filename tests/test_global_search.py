import numpy as np

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
