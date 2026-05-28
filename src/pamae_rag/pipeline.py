from __future__ import annotations

from dataclasses import asdict
import time
from typing import Iterable

import numpy as np

from pamae_rag.config import AppConfig
from pamae_rag.data.schema import QueryExample, RetrievalResult
from pamae_rag.eval.support_recall import hit, recall
from pamae_rag.eval.support_facts import support_fact_stage_metrics
from pamae_rag.eval.stage_diagnostics import make_stage_metrics
from pamae_rag.graph.distances import build_distance_matrix, validate_square_distance_matrix
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass
from pamae_rag.objective.anchor_objective import ObjectiveBreakdown, anchor_objective, assign_to_anchors
from pamae_rag.objective.relevance_mass import (
    normalize_relevance_scores,
    relevance_diagnostics,
    relevance_scores,
)
from pamae_rag.pamae.global_search import SearchResult
from pamae_rag.pamae.global_search import exact_k_medoids_on_sample
from pamae_rag.pamae.refinement import RefinementResult, refine_medoids_monotone
from pamae_rag.pamae.sampling import make_weighted_samples
from pamae_rag.pamae.selection import select_by_full_objective
from pamae_rag.qa.metrics import gold_answers, normalize_answer
from pamae_rag.rendering.basin_aware_renderer import render_basin_path_closure_indices
from pamae_rag.retrieval.renderer import render_context_indices
from pamae_rag.selection.basin_preserving import (
    BasinPreservingSelectionResult,
    gold_ids_in_selected_basins,
    select_basin_preserving_medoids,
)


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


def _all_node_ids(nodes) -> tuple[str, ...]:
    return tuple(node.node_id for node in nodes)


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


def _support_fact_extra(
    example: QueryExample,
    nodes,
    selected_node_ids: Iterable[str],
) -> dict:
    return support_fact_stage_metrics(
        nodes=nodes,
        selected_node_ids=selected_node_ids,
        metadata=example.metadata,
    )


def _prefix_metrics(prefix: str, values: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _answer_in_context(example: QueryExample, nodes, idxs: Iterable[int]) -> bool | None:
    answers = gold_answers(example)
    if not answers:
        return None
    context = " ".join(str(nodes[int(idx)].text) for idx in idxs)
    context_norm = normalize_answer(context)
    if not context_norm:
        return False
    padded = f" {context_norm} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded:
            return True
    return False


def _run_for_k(example: QueryExample, cfg: AppConfig, k: int, seed: int) -> RetrievalResult:
    stage_diagnostics: dict[str, dict] = {}
    stage_start = time.perf_counter()
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
    graph_diagnostics = dict(graph_result.diagnostics)
    projected_node_ids = tuple(str(x) for x in graph_diagnostics.pop("projected_node_ids", []))
    distance_matrix = graph_result.distance_matrix
    validate_square_distance_matrix(distance_matrix)

    rho_scores = relevance_scores(
        nodes,
        mode=cfg.pamae.relevance_mode,
        query=example.query,
        query_metadata=example.metadata,
        weights=cfg.pamae.relevance_weights,
    )
    rho = normalize_relevance_scores(rho_scores)
    rho_diagnostics = relevance_diagnostics(
        nodes,
        mode=cfg.pamae.relevance_mode,
        query=example.query,
        query_metadata=example.metadata,
        weights=cfg.pamae.relevance_weights,
        scores=rho_scores,
    )
    token_costs = np.asarray([max(1, node.token_count) / 1000.0 for node in nodes], dtype=np.float64)
    candidates = candidate_indices(nodes, cfg.universe.anchor_node_types)
    universe_token_count = _context_tokens(nodes, range(len(nodes)))
    stage_diagnostics["query_anchor_construction"] = make_stage_metrics(
        stage="query_anchor_construction",
        selected_node_ids=_all_node_ids(nodes),
        gold_node_ids=example.gold_node_ids,
        candidate_node_ids=_node_ids(nodes, candidates),
        token_count=universe_token_count,
        latency_ms=(time.perf_counter() - stage_start) * 1000.0,
        extra={
            "num_universe_nodes": len(nodes),
            "num_anchor_candidates": len(candidates),
            **_support_fact_extra(example, nodes, _all_node_ids(nodes)),
        },
    )
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
    basin_selection: BasinPreservingSelectionResult | None = None
    basin_selection_exact = True
    basin_diagnostics: dict = {}

    if retrieval_variant == "top_rho":
        stage_start = time.perf_counter()
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
        stage_diagnostics["candidate_generation"] = make_stage_metrics(
            stage="candidate_generation",
            selected_node_ids=_node_ids(nodes, anchors),
            gold_node_ids=example.gold_node_ids,
            candidate_node_ids=_node_ids(nodes, candidates),
            token_count=_context_tokens(nodes, anchors),
            latency_ms=(time.perf_counter() - stage_start) * 1000.0,
            extra={
                "candidate_strategy": "top_rho",
                **_support_fact_extra(example, nodes, _node_ids(nodes, anchors)),
            },
        )
    else:
        stage_start = time.perf_counter()
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
        sampled_ids = sorted({int(idx) for sample in samples for idx in sample})
        stage_diagnostics["candidate_generation"] = make_stage_metrics(
            stage="candidate_generation",
            selected_node_ids=_node_ids(nodes, sampled_ids),
            gold_node_ids=example.gold_node_ids,
            candidate_node_ids=_node_ids(nodes, candidates),
            token_count=_context_tokens(nodes, sampled_ids),
            latency_ms=(time.perf_counter() - stage_start) * 1000.0,
            extra={
                "candidate_strategy": "weighted_samples",
                "num_samples": len(samples),
                "sampled_candidate_count": len(sampled_ids),
                **_support_fact_extra(example, nodes, _node_ids(nodes, sampled_ids)),
            },
        )

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
        elif retrieval_variant == "basin_preserving_medoids":
            basin_selection = select_basin_preserving_medoids(
                phase1_results,
                candidates=candidates,
                k=k,
                distance_matrix=distance_matrix,
                rho=rho,
                token_costs=token_costs,
                token_weight=cfg.pamae.token_weight,
                anchor_penalty=cfg.pamae.anchor_penalty,
                sample_sizes=[len(sample) for sample in samples],
                max_combinations=cfg.pamae.max_exact_combinations,
                node_ids=[node.node_id for node in nodes],
                tau=cfg.pamae.basin_min_expected_samples,
            )
            anchors = basin_selection.anchors
            sample_objective = basin_selection.sample_objective
            full_validation_objective = basin_selection.full_objective
            selected_sample_index = basin_selection.sample_index
            basin_selection_exact = basin_selection.exact
            basin_diagnostics = dict(basin_selection.diagnostics)
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
            "basin_preserving_medoids",
            "adaptive_k",
        }:
            stage_start = time.perf_counter()
            pre_refine_anchor_ids = _node_ids(nodes, anchors)
            selected_basin_gold_ids = (
                gold_ids_in_selected_basins(
                    gold_node_ids=example.gold_node_ids,
                    nodes=nodes,
                    node_to_basin=basin_selection.node_to_basin,
                    covered_basin_ids=basin_selection.covered_basin_ids,
                )
                if basin_selection is not None
                else []
            )
            projected_gold_ids = sorted(set(projected_node_ids) & {str(x) for x in example.gold_node_ids})
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
            stage_diagnostics["local_refinement"] = make_stage_metrics(
                stage="local_refinement",
                selected_node_ids=_node_ids(nodes, refined.anchors),
                gold_node_ids=example.gold_node_ids,
                token_count=_context_tokens(nodes, refined.anchors),
                latency_ms=(time.perf_counter() - stage_start) * 1000.0,
                extra={
                    "refinement_accepted": refined.accepted,
                    "objective_before": refined.before.total,
                    "objective_after": refined.after.total,
                    **basin_diagnostics,
                    "projected_gold_hit": bool(projected_gold_ids),
                    "selected_basin_gold_hit": bool(selected_basin_gold_ids),
                    "selected_basin_gold_chunk_ids": selected_basin_gold_ids,
                    "selection_recovered_from_type_b": bool(
                        projected_gold_ids
                        and selected_basin_gold_ids
                        and not recall(pre_refine_anchor_ids, example.gold_node_ids)
                    ),
                    "pre_refinement_anchor_count": len(pre_refine_anchor_ids),
                    "pre_refinement_gold_supporting_evidence_survival": recall(
                        pre_refine_anchor_ids,
                        example.gold_node_ids,
                    ),
                    **_prefix_metrics(
                        "pre_refinement_",
                        _support_fact_extra(example, nodes, pre_refine_anchor_ids),
                    ),
                    **_support_fact_extra(example, nodes, _node_ids(nodes, refined.anchors)),
                },
            )
        else:
            refined = _identity_refinement(anchors, full_validation_objective, distance_matrix)
            pre_refine_anchor_ids = _node_ids(nodes, anchors)
            stage_diagnostics["local_refinement"] = make_stage_metrics(
                stage="local_refinement",
                selected_node_ids=_node_ids(nodes, refined.anchors),
                gold_node_ids=example.gold_node_ids,
                token_count=_context_tokens(nodes, refined.anchors),
                latency_ms=0.0,
                status="identity",
                extra={
                    "refinement_accepted": refined.accepted,
                    "objective_before": refined.before.total,
                    "objective_after": refined.after.total,
                    "pre_refinement_anchor_count": len(pre_refine_anchor_ids),
                    "pre_refinement_gold_supporting_evidence_survival": recall(
                        pre_refine_anchor_ids,
                        example.gold_node_ids,
                    ),
                    **_prefix_metrics(
                        "pre_refinement_",
                        _support_fact_extra(example, nodes, pre_refine_anchor_ids),
                    ),
                    **_support_fact_extra(example, nodes, _node_ids(nodes, refined.anchors)),
                },
            )
        exact_phase1 = all(result.exact for result in phase1_results) and basin_selection_exact

    anchors = refined.anchors
    stage_diagnostics.setdefault(
        "local_refinement",
        make_stage_metrics(
            stage="local_refinement",
            selected_node_ids=_node_ids(nodes, refined.anchors),
            gold_node_ids=example.gold_node_ids,
            token_count=_context_tokens(nodes, refined.anchors),
            latency_ms=0.0,
            status="identity",
            extra={
                "refinement_accepted": refined.accepted,
                "objective_before": refined.before.total,
                "objective_after": refined.after.total,
                "pre_refinement_anchor_count": len(_node_ids(nodes, refined.anchors)),
                "pre_refinement_gold_supporting_evidence_survival": recall(
                    _node_ids(nodes, refined.anchors),
                    example.gold_node_ids,
                ),
                **_prefix_metrics(
                    "pre_refinement_",
                    _support_fact_extra(example, nodes, _node_ids(nodes, refined.anchors)),
                ),
                **_support_fact_extra(example, nodes, _node_ids(nodes, refined.anchors)),
            },
        ),
    )
    content_projection_enabled = graph_diagnostics.get("graph_source") == "content"
    projection_node_ids = projected_node_ids if content_projection_enabled else _all_node_ids(nodes)
    stage_diagnostics["content_graph_projection"] = make_stage_metrics(
        stage="content_graph_projection",
        selected_node_ids=projection_node_ids,
        gold_node_ids=example.gold_node_ids,
        token_count=universe_token_count,
        latency_ms=graph_diagnostics.get("graph_build_latency_ms", 0.0)
        if content_projection_enabled
        else 0.0,
        status="ok" if content_projection_enabled else "not_configured",
        extra={
            "graph_source": graph_diagnostics.get("graph_source"),
            "projected_node_count": len(projected_node_ids) if content_projection_enabled else None,
            **_support_fact_extra(example, nodes, projection_node_ids),
        },
    )
    stage_diagnostics["reranking_scoring"] = make_stage_metrics(
        stage="reranking_scoring",
        selected_node_ids=_node_ids(nodes, anchors),
        gold_node_ids=example.gold_node_ids,
        token_count=_context_tokens(nodes, anchors),
        latency_ms=0.0,
        extra={
            "objective_after_refinement": refined.after.total,
            "anchor_count": len(anchors),
            **_support_fact_extra(example, nodes, _node_ids(nodes, anchors)),
        },
    )
    stage_start = time.perf_counter()
    basin_render_diagnostics: dict = {}
    if renderer == "basin_path_closure":
        if basin_selection is None:
            raise ValueError("basin_path_closure renderer requires basin_preserving_medoids selection")
        basin_masses = {
            int(key): float(value)
            for key, value in basin_selection.diagnostics.get("basin_masses", {}).items()
        }
        basin_render = render_basin_path_closure_indices(
            nodes,
            anchors,
            distance_matrix=distance_matrix,
            rho=rho,
            max_context_tokens=cfg.pamae.max_context_tokens,
            max_context_nodes=cfg.pamae.max_context_nodes,
            node_to_basin=basin_selection.node_to_basin,
            covered_basin_masses=basin_masses,
        )
        context_indices = basin_render.indices
        basin_render_diagnostics = dict(basin_render.diagnostics)
    else:
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
    render_latency_ms = (time.perf_counter() - stage_start) * 1000.0
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
        **graph_diagnostics,
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
        **basin_diagnostics,
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
        "context_budget_policy": (
            "basin_path_closure"
            if renderer == "basin_path_closure"
            else "anchors_then_cell_top_rho_then_score_fill"
        ),
        **basin_render_diagnostics,
        "max_context_nodes_less_than_k": bool(node_budget_active and max_context_nodes < k),
        "stage_diagnostics": {
            **stage_diagnostics,
            "context_rendering": make_stage_metrics(
                stage="context_rendering",
                selected_node_ids=context_node_ids,
                gold_node_ids=example.gold_node_ids,
                context_node_ids=context_node_ids,
                rendered_node_ids=context_node_ids,
                token_count=final_context_tokens,
                latency_ms=render_latency_ms,
                extra={
                    **basin_render_diagnostics,
                    "answer_in_context": _answer_in_context(example, nodes, context_indices),
                    **_support_fact_extra(example, nodes, context_node_ids),
                },
            ),
        },
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
