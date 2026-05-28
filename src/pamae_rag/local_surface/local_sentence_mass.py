from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pamae_rag.graph.content_graph import normalize_content_text
from pamae_rag.local_surface.local_surface_graph import LocalSurfaceGraph
from pamae_rag.sentence_graph.sentence_splitter import extract_query_entities


@dataclass(frozen=True)
class LocalSentenceMass:
    sentence_mass: dict[str, float]
    query_anchor_entities: tuple[str, ...]
    diagnostics: dict[str, Any]


def _fallback_anchor_entities(graph: LocalSurfaceGraph, query: str) -> tuple[str, ...]:
    query_tokens = set(normalize_content_text(query).split())
    if not query_tokens:
        return tuple(graph.entity_ids[:1])
    scored: list[tuple[int, str]] = []
    for entity_id in graph.entity_ids:
        surface = entity_id.removeprefix("entity:").replace("_", " ")
        overlap = len(set(surface.split()) & query_tokens)
        if overlap:
            scored.append((-overlap, entity_id))
    if scored:
        return tuple(entity_id for _score, entity_id in sorted(scored))
    return tuple(graph.entity_ids[:1])


def _ppr(
    graph: LocalSurfaceGraph,
    anchors: tuple[str, ...],
    *,
    alpha: float,
    max_iters: int = 100,
    tolerance: float = 1e-12,
) -> tuple[dict[str, float], float]:
    nodes = tuple(sorted(graph.node_ids))
    if not nodes:
        return {}, 0.0
    node_set = set(nodes)
    active_anchors = tuple(anchor for anchor in anchors if anchor in node_set)
    if not active_anchors:
        return {node_id: 1.0 / len(nodes) for node_id in nodes}, 0.0
    restart = {node_id: 0.0 for node_id in nodes}
    for anchor in active_anchors:
        restart[anchor] += 1.0 / len(active_anchors)
    rank = dict(restart)
    residual = 0.0
    for _ in range(max_iters):
        new_rank = {node_id: alpha * restart[node_id] for node_id in nodes}
        dangling_mass = 0.0
        for node_id in nodes:
            neighbors = graph.adjacency.get(node_id, ())
            if not neighbors:
                dangling_mass += rank[node_id]
                continue
            share = (1.0 - alpha) * rank[node_id] / len(neighbors)
            for neighbor, _length, _edge_type in neighbors:
                new_rank[neighbor] += share
        if dangling_mass:
            for node_id in nodes:
                new_rank[node_id] += (1.0 - alpha) * dangling_mass * restart[node_id]
        residual = sum(abs(new_rank[node_id] - rank[node_id]) for node_id in nodes)
        rank = new_rank
        if residual <= tolerance:
            break
    return rank, residual


def local_query_sentence_mass(
    graph: LocalSurfaceGraph,
    query: str,
    *,
    expected_hops: int = 2,
) -> LocalSentenceMass:
    alpha = 1.0 / (float(expected_hops) + 1.0)
    query_entities = extract_query_entities(query)
    query_anchor_entities = tuple(
        entity.entity_id for entity in query_entities if entity.entity_id in set(graph.entity_ids)
    )
    local_anchor_missing = False
    if not query_anchor_entities:
        local_anchor_missing = True
        query_anchor_entities = _fallback_anchor_entities(graph, query)
    rank, residual = _ppr(graph, query_anchor_entities, alpha=alpha)
    sentence_mass = {sentence_id: rank.get(sentence_id, 0.0) for sentence_id in graph.sentence_ids}
    total = sum(sentence_mass.values())
    if total > 0:
        sentence_mass = {node_id: value / total for node_id, value in sentence_mass.items()}
    elif graph.sentence_ids:
        uniform = 1.0 / len(graph.sentence_ids)
        sentence_mass = {node_id: uniform for node_id in graph.sentence_ids}
    diagnostics = {
        "local_sentence_count": len(graph.sentence_ids),
        "local_fact_count": len(graph.facts),
        "local_entity_count": len(graph.entity_ids),
        "local_anchor_count": len(query_anchor_entities),
        "local_anchor_missing": local_anchor_missing,
        "local_query_anchor_entities": list(query_anchor_entities),
        "local_alpha": alpha,
        "local_residual_mass": residual,
    }
    return LocalSentenceMass(
        sentence_mass=sentence_mass,
        query_anchor_entities=query_anchor_entities,
        diagnostics=diagnostics,
    )


__all__ = ["LocalSentenceMass", "local_query_sentence_mass"]
