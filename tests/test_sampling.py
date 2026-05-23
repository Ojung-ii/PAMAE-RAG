import numpy as np

from pamae_rag.pamae.sampling import make_weighted_samples, sample_probabilities


def test_weighted_sampling_shapes():
    rho = np.array([0.5, 0.2, 0.2, 0.1])
    probs = sample_probabilities([0, 1, 2, 3], rho, uniform_mix=0.1)
    assert np.isclose(probs.sum(), 1.0)
    samples = make_weighted_samples([0, 1, 2, 3], rho, k=2, num_samples=3, sample_size_per_k=2, sample_size_cap=4, seed=1)
    assert len(samples) == 3
    assert all(len(s) == 4 for s in samples)
