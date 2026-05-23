from __future__ import annotations

from typing import Sequence

import numpy as np


def sample_probabilities(candidate_indices: Sequence[int], rho: np.ndarray, uniform_mix: float) -> np.ndarray:
    candidates = np.asarray(list(candidate_indices), dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("candidate_indices must not be empty")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    weights = np.maximum(rho[candidates], 0.0)
    if weights.sum() <= 0:
        mass = np.full(candidates.size, 1.0 / candidates.size)
    else:
        mass = weights / weights.sum()
    uniform = np.full(candidates.size, 1.0 / candidates.size)
    probs = (1.0 - uniform_mix) * mass + uniform_mix * uniform
    return probs / probs.sum()


def make_weighted_samples(candidate_indices: Sequence[int], rho: np.ndarray, k: int, num_samples: int, sample_size_per_k: int, sample_size_cap: int, seed: int, uniform_mix: float = 0.15) -> list[list[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    candidates = list(dict.fromkeys(int(i) for i in candidate_indices))
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    sample_size = min(len(candidates), max(k, min(sample_size_cap, sample_size_per_k * k)))
    probs = sample_probabilities(candidates, rho, uniform_mix)
    samples: list[list[int]] = []
    for _ in range(num_samples):
        draw = rng.choice(np.asarray(candidates, dtype=np.int64), size=sample_size, replace=False, p=probs)
        samples.append(sorted(int(x) for x in draw.tolist()))
    return samples
