from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from pamae_rag.config import AppConfig
from pamae_rag.data.schema import QueryExample, RetrievalResult
from pamae_rag.eval.support_recall import hit, recall
from pamae_rag.graph.distances import build_distance_matrix, validate_square_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass
from pamae_rag.objective.relevance_mass import relevance_mass
from pamae_rag.pamae.global_search import exact_k_medoids_on_sample
from pamae_rag.pamae.refinement import refine_medoids_monotone
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


def _run_for_k(example: QueryExample, cfg: AppConfig, k: int, seed: int) -> RetrievalResult:
    nodes = select_universe_by_mass(
        example.nodes,
        max_nodes=cfg.universe.max_nodes,
        min_relevance_mass=cfg.universe.min_relevance_mass,
    )
    if not nodes:
        raise ValueError(f"Example {example.query_id!r} has empty universe")

    embeddings = np.vstack([node.embedding for node in nodes])
    distance_matrix = build_distance_matrix(embeddings, metric=cfg.distance.metric)
    validate_square_distance_matrix(distance_matrix)

    rho = relevance_mass(nodes)
    token_costs = np.asarray([max(1, node.token_count) / 1000.0 for node in nodes], dtype=np.float64)
    candidates = candidate_indices(nodes, cfg.universe.anchor_node_types)
    if len(candidates) < k:
        k = len(candidates)
    if k < 1:
        raise ValueError(f"Example {example.query_id!r} has no anchor candidates")

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

    selected = select_by_full_objective(
        phase1_results,
        distance_matrix=distance_matrix,
        rho=rho,
        token_costs=token_costs,
        token_weight=cfg.pamae.token_weight,
        anchor_penalty=cfg.pamae.anchor_penalty,
    )

    refined = refine_medoids_monotone(
        selected.anchors,
        candidate_indices=candidates,
        distance_matrix=distance_matrix,
        rho=rho,
        token_costs=token_costs,
        token_weight=cfg.pamae.token_weight,
        anchor_penalty=cfg.pamae.anchor_penalty,
        max_iters=cfg.pamae.refinement_iters,
    )

    anchors = refined.anchors
    context_indices = render_context_indices(
        nodes,
        anchors,
        distance_matrix=distance_matrix,
        rho=rho,
        max_context_tokens=cfg.pamae.max_context_tokens,
        evidence_per_anchor=cfg.pamae.evidence_per_anchor,
    )

    anchor_ids = _node_ids(nodes, anchors)
    context_node_ids = _node_ids(nodes, context_indices)
    support_recall = recall(context_node_ids, example.gold_node_ids)
    support_hit = hit(context_node_ids, example.gold_node_ids)

    exact_phase1 = all(result.exact for result in phase1_results)
    diagnostics = {
        "k": k,
        "num_universe_nodes": len(nodes),
        "num_anchor_candidates": len(candidates),
        "num_samples": len(samples),
        "sample_sizes": [len(s) for s in samples],
        "selected_sample_index": selected.sample_index,
        "phase1_exact": exact_phase1,
        "phase1_num_combinations": [r.num_combinations for r in phase1_results],
        "sample_objective": asdict(selected.sample_objective),
        "full_validation_objective": asdict(selected.full_objective),
        "refinement_before": asdict(refined.before),
        "refinement_after": asdict(refined.after),
        "refinement_accepted": refined.accepted,
        "refinement_history": refined.history,
        "cluster_sizes": refined.cluster_sizes,
    }

    return RetrievalResult(
        query_id=example.query_id,
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
