from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def angular_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = normalize_rows(np.asarray(embeddings, dtype=np.float64))
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = np.arccos(sim) / np.pi
    np.fill_diagonal(dist, 0.0)
    return dist


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = normalize_rows(np.asarray(embeddings, dtype=np.float64))
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def build_distance_matrix(embeddings: np.ndarray, metric: str = "angular") -> np.ndarray:
    if metric == "angular":
        return angular_distance_matrix(embeddings)
    if metric == "cosine":
        return cosine_distance_matrix(embeddings)
    raise ValueError(f"Unsupported distance metric: {metric}")


def validate_square_distance_matrix(d: np.ndarray) -> None:
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance matrix must be square")
    if np.any(d < -1e-9):
        raise ValueError("distance matrix contains negative entries")
    if not np.allclose(d, d.T, atol=1e-6):
        raise ValueError("distance matrix must be symmetric")
