from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from pamae_rag.config import AppConfig
from pamae_rag.data.schema import QueryExample, RetrievalResult
from pamae_rag.eval.support_recall import hit, recall
from pamae_rag.graph.distances import build_distance_matrix, validate_square_distance_matrix
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass
from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective, assign_to_anchors
from pamae_rag.objective.relevance_mass import relevance_diagnostics, relevance_mass
from pamae_rag.pamae.global_search import SearchResult
from pamae_rag.pamae.global_search import exact_k_medoids_on_sample
from pamae_rag.pamae.refinement import RefinementResult, refine_medoids_monotone
from pamae_rag.pamae.sampling import make_weighted_samples
from pamae_rag.pamae.selection import select_by_full_objective
from pamae_rag.retrieval.renderer import render_context_indices


def candidate_indices(nodes, anchor_node_types: Iterable[str]) -> list[int]:
    allowed = set(anchor_node_types)
    idxs = [
        i
        for i, node in enumerate(nodes)
        if node.is_anchor_candidate and (not allowed or node.node_type in allowed)
    ]
    if not idxs:
        idxs = list(range(len(nodes)))
    return idxs


def _node_ids(nodes, idxs: Iterable[int]) -> tuple[str, ...]:
    return tuple(nodes[int(i)].node_id for i in idxs)


def _context_tokens(nodes, idxs: Iterable[int]) -> int:
    return int(sum(max(1, int(nodes[int(i)].token_count)) for i in idxs))


def _cluster_sizes(anchor_indices: Iterable[int], distance_matrix: np.ndarray) -> list[int]:
    anchors = list(anchor_indices)
    if not anchors:
        return []
    assignments = assign_to_anchors(anchors, distance_matrix)
    return [int(np.sum(assignments == pos)) for pos in range(len(anchors))]


def _top_rho_candidates(candidates: list[int], rho: np.ndarray, k: int) -> list[int]:
    ranked = sorted(candidates, key=lambda i: (-float(rho[int(i)]), int(i)))
    return ranked[:k]


def _objective_json(obj: ObjectiveBreakdown | None) -> dict | None:
    return asdict(obj) if obj is not None else None


def _selected_from_sample_objective(
    phase1_results: list[SearchResult],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    token_costs: np.ndarray,
    token_weight: float,
    anchor_penalty: float,
):
    best_index = min(range(len(phase1_results)), key=lambda i: phase1_results[i].objective.total)
    result = phase1_results[best_index]
    full_obj = anchor_objective(
        result.anchors,
        distance_matrix=distance_matrix,
        rho=rho,
        token_costs=token_costs,
        token_weight=token_weight,
        anchor_penalty=anchor_penalty,
    )
    return result.anchors, result.objective, full_obj, best_index


def _identity_refinement(
    anchors: list[int],
    objective: ObjectiveBreakdown,
    distance_matrix: np.ndarray,
) -> RefinementResult:
    return RefinementResult(
        anchors=list(anchors),
        before=objective,
        after=objective,
        accepted=False,
        history=[objective.total],
        cluster_sizes=_cluster_sizes(anchors, distance_matrix),
    )


def _actual_renderer(cfg: AppConfig) -> str:
    if cfg.pamae.retrieval_variant == "sample_full_validation_refine_cell_renderer":
        return "cell_top_rho"
    if cfg.pamae.retrieval_variant == "top_rho":
        return "global_top_rho"
    return cfg.pamae.renderer


def _run_for_k(example: QueryExample, cfg: AppConfig, k: int, seed: int) -> RetrievalResult:
    nodes = select_universe_by_mass(
        example.nodes,
        max_nodes=cfg.universe.max_nodes,
        min_relevance_mass=cfg.universe.min_relevance_mass,
    )
    if not nodes:
        raise ValueError(f"Example {example.query_id!r} has empty universe")

    embeddings = np.vstack([node.embedding for node in nodes])
    semantic_distance_matrix = build_distance_matrix(embeddings, metric=cfg.distance.metric)
    graph_result = build_graph_aware_distance_matrix(
        nodes,
        example.query,
        semantic_distance_matrix,
        distance_mode=cfg.pamae.distance_mode,
        distance_weights={
            "semantic": cfg.pamae.distance_weights.semantic,
            "graph": cfg.pamae.distance_weights.graph,
        },
        graph_config=cfg.pamae.graph,
    )
    distance_matrix = graph_result.distance_matrix
    validate_square_distance_matrix(distance_matrix)

    rho = relevance_mass(
        nodes,
        mode=cfg.pamae.relevance_mode,
        query=example.query,
        query_metadata=example.metadata,
        weights=cfg.pamae.relevance_weights,
    )
    rho_diagnostics = relevance_diagnostics(
        nodes,
        mode=cfg.pamae.relevance_mode,
        query=example.query,
        query_metadata=example.metadata,
        weights=cfg.pamae.relevance_weights,
    )
    token_costs = np.asarray([max(1, node.token_count) / 1000.0 for node in nodes], dtype=np.float64)
    candidates = candidate_indices(nodes, cfg.universe.anchor_node_types)
    if len(candidates) < k:
        k = len(candidates)
    if k < 1:
        raise ValueError(f"Example {example.query_id!r} has no anchor candidates")

    retrieval_variant = cfg.pamae.retrieval_variant
    renderer = _actual_renderer(cfg)
    samples: list[list[int]] = []
    phase1_results: list[SearchResult] = []
    selected_sample_index: int | None = None
    sample_objective: ObjectiveBreakdown | None = None
    full_validation_objective: ObjectiveBreakdown | None = None

    if retrieval_variant == "top_rho":
        anchors = _top_rho_candidates(candidates, rho, k)
        full_validation_objective = anchor_objective(
            anchors,
            distance_matrix=distance_matrix,
            rho=rho,
            token_costs=token_costs,
            token_weight=cfg.pamae.token_weight,
            anchor_penalty=cfg.pamae.anchor_penalty,
        )
        refined = _identity_refinement(anchors, full_validation_objective, distance_matrix)
        exact_phase1 = True
    else:
        samples = make_weighted_samples(
            candidates,
            rho,
            k=k,
            num_samples=cfg.pamae.num_samples,
            sample_size_per_k=cfg.pamae.sample_size_per_k,
            sample_size_cap=cfg.pamae.sample_size_cap,
            seed=seed,
            uniform_mix=cfg.pamae.sample_uniform_mix,
        )
        if not samples:
            raise ValueError(f"Example {example.query_id!r} produced no samples")

        phase1_results = [
            exact_k_medoids_on_sample(
                sample,
                k=k,
                distance_matrix=distance_matrix,
                rho=rho,
                token_costs=token_costs,
                token_weight=cfg.pamae.token_weight,
                anchor_penalty=cfg.pamae.anchor_penalty,
                max_combinations=cfg.pamae.max_exact_combinations,
                require_exact=cfg.pamae.require_exact_sample_search,
            )
            for sample in samples
        ]

        if retrieval_variant == "sample_only":
            anchors, sample_objective, full_validation_objective, selected_sample_index = (
                _selected_from_sample_objective(
                    phase1_results,
                    distance_matrix=distance_matrix,
                    rho=rho,
                    token_costs=token_costs,
                    token_weight=cfg.pamae.token_weight,
                    anchor_penalty=cfg.pamae.anchor_penalty,
                )
            )
        else:
            selected = select_by_full_objective(
                phase1_results,
                distance_matrix=distance_matrix,
                rho=rho,
                token_costs=token_costs,
                token_weight=cfg.pamae.token_weight,
                anchor_penalty=cfg.pamae.anchor_penalty,
            )
            anchors = selected.anchors
            sample_objective = selected.sample_objective
            full_validation_objective = selected.full_objective
            selected_sample_index = selected.sample_index

        if retrieval_variant in {
            "sample_full_validation_refine",
            "sample_full_validation_refine_cell_renderer",
            "adaptive_k",
        }:
            refined = refine_medoids_monotone(
                anchors,
                candidate_indices=candidates,
                distance_matrix=distance_matrix,
                rho=rho,
                token_costs=token_costs,
                token_weight=cfg.pamae.token_weight,
                anchor_penalty=cfg.pamae.anchor_penalty,
                max_iters=cfg.pamae.refinement_iters,
            )
        else:
            refined = _identity_refinement(anchors, full_validation_objective, distance_matrix)
        exact_phase1 = all(result.exact for result in phase1_results)

    anchors = refined.anchors
    context_indices = render_context_indices(
        nodes,
        anchors,
        distance_matrix=distance_matrix,
        rho=rho,
        max_context_tokens=cfg.pamae.max_context_tokens,
        max_context_nodes=cfg.pamae.max_context_nodes,
        evidence_per_anchor=cfg.pamae.evidence_per_anchor,
        renderer=renderer,
        gamma=cfg.pamae.renderer_gamma,
    )

    anchor_ids = _node_ids(nodes, anchors)
    context_node_ids = _node_ids(nodes, context_indices)
    final_context_tokens = _context_tokens(nodes, context_indices)
    support_recall = recall(context_node_ids, example.gold_node_ids)
    support_hit = hit(context_node_ids, example.gold_node_ids)
    max_context_nodes = cfg.pamae.max_context_nodes
    node_budget_active = max_context_nodes is not None and max_context_nodes > 0
    unique_anchor_count = len(list(dict.fromkeys(int(a) for a in anchors)))
    node_budget_exceeded_by_anchors = bool(node_budget_active and unique_anchor_count > max_context_nodes)
    node_budget_satisfied = bool(not node_budget_active or len(context_indices) <= max_context_nodes)
    token_budget_satisfied = bool(final_context_tokens <= cfg.pamae.max_context_tokens)

    diagnostics = {
        "retrieval_variant": retrieval_variant,
        "renderer": renderer,
        "relevance_mode": cfg.pamae.relevance_mode,
        "relevance_weights": dict(cfg.pamae.relevance_weights),
        **graph_result.diagnostics,
        "semantic_component_available": rho_diagnostics["semantic_component_available"],
        "query_title_spans": rho_diagnostics["query_title_spans"],
        "top_relevance_node_ids": rho_diagnostics["top_relevance_node_ids"],
        "k": k,
        "k_max": cfg.pamae.k_max,
        "selected_k": k,
        "num_universe_nodes": len(nodes),
        "num_anchor_candidates": len(candidates),
        "num_samples": len(samples),
        "sample_sizes": [len(s) for s in samples],
        "selected_sample_index": selected_sample_index,
        "phase1_exact": exact_phase1,
        "phase1_num_combinations": [r.num_combinations for r in phase1_results],
        "sample_objective": _objective_json(sample_objective),
        "full_validation_objective": _objective_json(full_validation_objective),
        "refinement_before": asdict(refined.before),
        "refinement_after": asdict(refined.after),
        "refinement_accepted": refined.accepted,
        "refinement_history": refined.history,
        "cluster_sizes": refined.cluster_sizes,
        "fallback_used": not exact_phase1,
        "max_context_nodes": max_context_nodes,
        "max_context_tokens": cfg.pamae.max_context_tokens,
        "final_context_nodes": len(context_indices),
        "final_context_tokens": final_context_tokens,
        "node_budget_satisfied": node_budget_satisfied,
        "token_budget_satisfied": token_budget_satisfied,
        "node_budget_exceeded_by_anchors": node_budget_exceeded_by_anchors,
        "context_budget_policy": "anchors_then_cell_top_rho_then_score_fill",
        "max_context_nodes_less_than_k": bool(node_budget_active and max_context_nodes < k),
    }

    return RetrievalResult(
        query_id=example.query_id,
        anchor_node_ids=anchor_ids,
        anchor_ids=anchor_ids,
        context_node_ids=context_node_ids,
        objective_before_refinement=refined.before.total,
        objective_after_refinement=refined.after.total,
        support_recall=support_recall,
        support_hit=support_hit,
        exact_phase1=exact_phase1,
        diagnostics=diagnostics,
    )


def run_query_pamae(example: QueryExample, cfg: AppConfig, seed_offset: int = 0) -> RetrievalResult:
    if cfg.pamae.retrieval_variant == "adaptive_k":
        best: RetrievalResult | None = None
        best_score = float("inf")
        max_k = min(cfg.pamae.k_max, max(1, len(example.nodes)))
        for k in range(1, max_k + 1):
            result = _run_for_k(example, cfg, k=k, seed=cfg.seed + seed_offset + 1009 * k)
            score = result.objective_after_refinement + cfg.pamae.lambda_k * len(result.anchor_node_ids)
            if score < best_score:
                best_score = score
                best = result
        assert best is not None
        best.diagnostics["selected_k"] = len(best.anchor_node_ids)
        best.diagnostics["adaptive_k_score"] = best_score
        return best
    if cfg.pamae.auto_k:
        best: RetrievalResult | None = None
        max_k = min(cfg.pamae.k_max, max(1, len(example.nodes)))
        for k in range(1, max_k + 1):
            result = _run_for_k(example, cfg, k=k, seed=cfg.seed + seed_offset + 1009 * k)
            if best is None or result.objective_after_refinement < best.objective_after_refinement:
                best = result
        assert best is not None
        return best
    return _run_for_k(example, cfg, k=cfg.pamae.k, seed=cfg.seed + seed_offset)
