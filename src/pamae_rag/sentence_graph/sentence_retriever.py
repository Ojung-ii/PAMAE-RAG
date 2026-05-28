from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pamae_rag.objective.anchor_objective import assign_to_anchors
from pamae_rag.pamae.global_search import exact_k_medoids_on_sample
from pamae_rag.pamae.refinement import refine_medoids_monotone
from pamae_rag.pamae.sampling import make_weighted_samples
from pamae_rag.pamae.selection import select_by_full_objective
from pamae_rag.sentence_graph.sentence_graph_builder import SentenceGraphIndex
from pamae_rag.sentence_graph.sentence_metric_distance import (
    sentence_shortest_path_distances,
    triangle_inequality_violation_count,
)
from pamae_rag.sentence_graph.sentence_query_mass import sentence_mass_from_ppr


@dataclass(frozen=True)
class SentenceRetrieverConfig:
    expected_hops: int = 2
    active_mass_eta: float = 0.995
    include_chunk_parent_edges_in_ppr: bool = False
    use_chunk_parent_edges_in_metric: bool = False
    disconnected_distance: float = 1_000_000.0
    k: int = 3
    num_samples: int = 5
    sample_size_per_k: int = 12
    sample_size_cap: int = 48
    sample_uniform_mix: float = 0.15
    require_exact_sample_search: bool = True
    max_exact_combinations: int = 1_000_000
    refinement_iters: int = 1
    objective_tolerance: float = 1e-12


@dataclass(frozen=True)
class SentenceRetrievalResult:
    selected_sentence_ids: tuple[str, ...]
    pre_refinement_sentence_ids: tuple[str, ...]
    active_sentence_ids: tuple[str, ...]
    active_sentence_mass: np.ndarray
    distance_matrix: np.ndarray
    query_anchor_entity_ids: tuple[str, ...]
    query_anchor_entities: tuple[str, ...]
    anchor_fallback_used: bool
    node_to_basin: dict[str, int]
    phi_before_refine: float
    phi_after_refine: float
    objective_decreased: bool
    objective_increase: bool
    exact_phase1: bool
    diagnostics: dict[str, Any]


def _ids(local_ids: list[int], active_sentence_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(active_sentence_ids[int(idx)] for idx in local_ids)


def retrieve_sentence_medoids(
    index: SentenceGraphIndex,
    query: str,
    *,
    config: SentenceRetrieverConfig,
    seed: int,
) -> SentenceRetrievalResult:
    mass = sentence_mass_from_ppr(
        index,
        query,
        expected_hops=config.expected_hops,
        active_mass_eta=config.active_mass_eta,
        include_chunk_parent_edges_in_ppr=config.include_chunk_parent_edges_in_ppr,
    )
    if not mass.active_sentence_ids:
        raise ValueError("Sentence graph produced an empty active sentence universe")
    metric = sentence_shortest_path_distances(
        index,
        mass.active_sentence_ids,
        use_chunk_parent_edges_in_metric=config.use_chunk_parent_edges_in_metric,
        disconnected_distance=config.disconnected_distance,
    )
    rho = mass.active_sentence_mass
    candidates = list(range(len(mass.active_sentence_ids)))
    k = min(max(1, int(config.k)), len(candidates))
    samples = make_weighted_samples(
        candidates,
        rho,
        k=k,
        num_samples=config.num_samples,
        sample_size_per_k=config.sample_size_per_k,
        sample_size_cap=config.sample_size_cap,
        seed=seed,
        uniform_mix=config.sample_uniform_mix,
    )
    phase1 = [
        exact_k_medoids_on_sample(
            sample,
            k=k,
            distance_matrix=metric.distance_matrix,
            rho=rho,
            token_costs=None,
            token_weight=0.0,
            anchor_penalty=0.0,
            max_combinations=config.max_exact_combinations,
            require_exact=config.require_exact_sample_search,
        )
        for sample in samples
    ]
    selected = select_by_full_objective(
        phase1,
        distance_matrix=metric.distance_matrix,
        rho=rho,
        token_costs=None,
        token_weight=0.0,
        anchor_penalty=0.0,
    )
    refined = refine_medoids_monotone(
        selected.anchors,
        candidate_indices=candidates,
        distance_matrix=metric.distance_matrix,
        rho=rho,
        token_costs=None,
        token_weight=0.0,
        anchor_penalty=0.0,
        max_iters=config.refinement_iters,
    )
    objective_decreased = bool(refined.after.total <= refined.before.total + config.objective_tolerance)
    final_anchors = refined.anchors if objective_decreased else selected.anchors
    assignments = assign_to_anchors(final_anchors, metric.distance_matrix)
    node_to_basin = {
        sentence_id: int(assignments[pos])
        for pos, sentence_id in enumerate(mass.active_sentence_ids)
    }
    triangle_violations = triangle_inequality_violation_count(metric.distance_matrix)
    diagnostics = {
        "retrieval_variant": "sentence_primary_sample_full_validation_refine",
        "selected_sample_index": selected.sample_index,
        "num_samples": len(samples),
        "sample_sizes": [len(sample) for sample in samples],
        "phase1_num_combinations": [result.num_combinations for result in phase1],
        "phase1_exact": all(result.exact for result in phase1),
        "sample_objective": asdict(selected.sample_objective),
        "full_validation_objective": asdict(selected.full_objective),
        "refinement_before": asdict(refined.before),
        "refinement_after": asdict(refined.after),
        "refinement_accepted": refined.accepted,
        "refinement_history": refined.history,
        "cluster_sizes": refined.cluster_sizes,
        "selected_medoids_are_sentence_nodes": all(
            sentence_id.startswith("sent:") for sentence_id in _ids(final_anchors, mass.active_sentence_ids)
        ),
        "ppr_used_only_for_sentence_mass": True,
        "objective_increase_count": 0 if objective_decreased else 1,
        "triangle_inequality_violation_count": triangle_violations,
        **mass.diagnostics,
        **metric.diagnostics,
    }
    return SentenceRetrievalResult(
        selected_sentence_ids=_ids(final_anchors, mass.active_sentence_ids),
        pre_refinement_sentence_ids=_ids(selected.anchors, mass.active_sentence_ids),
        active_sentence_ids=mass.active_sentence_ids,
        active_sentence_mass=rho,
        distance_matrix=metric.distance_matrix,
        query_anchor_entity_ids=mass.query_anchors.entity_ids,
        query_anchor_entities=mass.query_anchors.query_anchor_entities,
        anchor_fallback_used=mass.query_anchors.anchor_fallback_used,
        node_to_basin=node_to_basin,
        phi_before_refine=float(refined.before.total),
        phi_after_refine=float(refined.after.total),
        objective_decreased=objective_decreased,
        objective_increase=not objective_decreased,
        exact_phase1=all(result.exact for result in phase1),
        diagnostics=diagnostics,
    )
