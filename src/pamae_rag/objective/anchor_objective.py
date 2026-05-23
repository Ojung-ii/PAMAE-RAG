from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: float
    coverage: float
    token_penalty: float
    anchor_penalty: float


def anchor_objective(anchor_indices: Sequence[int], distance_matrix: np.ndarray, rho: np.ndarray, token_costs: np.ndarray | None = None, token_weight: float = 0.0, anchor_penalty: float = 0.0, row_indices: Sequence[int] | None = None) -> ObjectiveBreakdown:
    if len(anchor_indices) == 0:
        return ObjectiveBreakdown(float("inf"), float("inf"), 0.0, 0.0)
    anchors = np.asarray(list(anchor_indices), dtype=np.int64)
    if len(set(anchors.tolist())) != len(anchors):
        raise ValueError("anchor indices must be unique")
    if np.any(anchors < 0) or np.any(anchors >= distance_matrix.shape[1]):
        raise IndexError("anchor index out of range")
    rows = np.asarray(row_indices if row_indices is not None else np.arange(distance_matrix.shape[0]), dtype=np.int64)
    sub_d = distance_matrix[np.ix_(rows, anchors)]
    coverage = float(np.dot(rho[rows], np.min(sub_d, axis=1)))
    token_pen = 0.0
    if token_costs is not None and token_weight > 0.0:
        token_pen = float(token_weight * np.sum(token_costs[anchors]))
    k_pen = float(anchor_penalty * len(anchors))
    return ObjectiveBreakdown(total=coverage + token_pen + k_pen, coverage=coverage, token_penalty=token_pen, anchor_penalty=k_pen)


def assign_to_anchors(anchor_indices: Sequence[int], distance_matrix: np.ndarray) -> np.ndarray:
    if len(anchor_indices) == 0:
        raise ValueError("anchor_indices must be non-empty")
    anchors = np.asarray(list(anchor_indices), dtype=np.int64)
    nearest_pos = np.argmin(distance_matrix[:, anchors], axis=1)
    return nearest_pos
