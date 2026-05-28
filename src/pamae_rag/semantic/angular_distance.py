from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_embedding(vector: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    """Return an L2-normalized copy, or None for missing/invalid vectors."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def angular_distance(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Normalized angular distance arccos(<z_l,z_r>) / pi for normalized embeddings."""
    left_norm = normalize_embedding(left)
    right_norm = normalize_embedding(right)
    if left_norm is None or right_norm is None:
        raise ValueError("angular_distance requires non-empty finite vectors")
    if left_norm.shape != right_norm.shape:
        raise ValueError(f"embedding dimensions differ: {left_norm.shape} != {right_norm.shape}")
    dot = float(np.dot(left_norm, right_norm))
    clamped = max(-1.0, min(1.0, dot))
    value = float(np.arccos(clamped) / np.pi)
    if abs(value) <= 1e-12:
        return 0.0
    if abs(value - 1.0) <= 1e-12:
        return 1.0
    return value


def min_angular_distance(
    vector: Sequence[float] | np.ndarray,
    candidates: Sequence[Sequence[float] | np.ndarray],
) -> float | None:
    distances: list[float] = []
    for candidate in candidates:
        try:
            distances.append(angular_distance(vector, candidate))
        except ValueError:
            continue
    if not distances:
        return None
    return float(min(distances))


__all__ = ["angular_distance", "min_angular_distance", "normalize_embedding"]
