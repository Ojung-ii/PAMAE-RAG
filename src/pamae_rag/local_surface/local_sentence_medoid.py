from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from pamae_rag.local_surface.local_metric_distance import (
    shortest_path_distances,
    validate_triangle_inequality,
)
from pamae_rag.local_surface.local_sentence_mass import LocalSentenceMass, local_query_sentence_mass
from pamae_rag.local_surface.local_surface_graph import LocalSurfaceGraph


@dataclass(frozen=True)
class LocalMedoidConfig:
    local_sentence_medoids: int = 4
    expected_hops: int = 2


@dataclass(frozen=True)
class LocalMedoidResult:
    selected_sentence_ids: tuple[str, ...]
    sentence_mass: dict[str, float]
    query_anchor_entities: tuple[str, ...]
    objective: float
    reachable_sentence_count: int
    diagnostics: dict[str, Any]


def local_medoid_objective(
    sentence_ids: tuple[str, ...],
    medoid_ids: tuple[str, ...],
    sentence_mass: dict[str, float],
    distances: dict[str, dict[str, float]],
) -> tuple[float, int]:
    total = 0.0
    reachable = 0
    for sentence_id in sentence_ids:
        best = min((distances[sentence_id].get(medoid, float("inf")) for medoid in medoid_ids), default=float("inf"))
        if best == float("inf"):
            continue
        reachable += 1
        total += float(sentence_mass.get(sentence_id, 0.0)) * best
    return total, reachable


def select_local_sentence_medoids(
    graph: LocalSurfaceGraph,
    query: str,
    *,
    config: LocalMedoidConfig | None = None,
) -> LocalMedoidResult:
    cfg = config or LocalMedoidConfig()
    mass: LocalSentenceMass = local_query_sentence_mass(
        graph,
        query,
        expected_hops=cfg.expected_hops,
    )
    sentence_ids = tuple(sorted(graph.sentence_ids))
    if not sentence_ids:
        return LocalMedoidResult(
            selected_sentence_ids=tuple(),
            sentence_mass={},
            query_anchor_entities=mass.query_anchor_entities,
            objective=0.0,
            reachable_sentence_count=0,
            diagnostics={
                **mass.diagnostics,
                "local_sentence_medoids": cfg.local_sentence_medoids,
                "local_objective_invalid": False,
                "triangle_inequality_violation_count": 0,
                "selected_medoids_are_sentence_nodes": True,
            },
        )
    k = min(int(cfg.local_sentence_medoids), len(sentence_ids))
    distances = shortest_path_distances(graph, sentence_ids)
    best_ids: tuple[str, ...] | None = None
    best_objective = float("inf")
    best_reachable = 0
    for candidate in combinations(sentence_ids, k):
        objective, reachable = local_medoid_objective(
            sentence_ids,
            tuple(candidate),
            mass.sentence_mass,
            distances,
        )
        key = (objective, tuple(candidate))
        best_key = (best_objective, best_ids or tuple("~" for _ in range(k)))
        if key < best_key:
            best_ids = tuple(candidate)
            best_objective = objective
            best_reachable = reachable
    assert best_ids is not None
    valid_sentences = set(sentence_ids)
    local_objective_invalid = best_objective == float("inf") or any(
        medoid not in valid_sentences for medoid in best_ids
    )
    diagnostics = {
        **mass.diagnostics,
        "local_sentence_medoids": cfg.local_sentence_medoids,
        "selected_local_sentence_medoids": list(best_ids),
        "local_objective": best_objective,
        "local_reachable_sentence_count": best_reachable,
        "local_unreachable_sentence_count": len(sentence_ids) - best_reachable,
        "local_objective_invalid": local_objective_invalid,
        "ppr_used_only_as_mass": True,
        "local_objective_uses_graph_distance": True,
        "deterministic_tie_break": "node_id",
        "selected_medoids_are_sentence_nodes": all(medoid in valid_sentences for medoid in best_ids),
        "triangle_inequality_violation_count": validate_triangle_inequality(graph, sentence_ids),
    }
    return LocalMedoidResult(
        selected_sentence_ids=best_ids,
        sentence_mass=mass.sentence_mass,
        query_anchor_entities=mass.query_anchor_entities,
        objective=best_objective,
        reachable_sentence_count=best_reachable,
        diagnostics=diagnostics,
    )


__all__ = [
    "LocalMedoidConfig",
    "LocalMedoidResult",
    "local_medoid_objective",
    "select_local_sentence_medoids",
]
