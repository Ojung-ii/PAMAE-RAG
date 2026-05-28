from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from pamae_rag.graph.content_graph import normalize_content_text
from pamae_rag.sentence_graph.sentence_graph_builder import SentenceGraphIndex
from pamae_rag.sentence_graph.sentence_splitter import extract_query_entities

try:  # pragma: no cover - optional speed path.
    from scipy.sparse import csr_matrix
except ModuleNotFoundError:  # pragma: no cover - focused unit env may not include SciPy.
    csr_matrix = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QueryAnchorResult:
    entity_ids: tuple[str, ...]
    anchor_fallback_used: bool
    query_anchor_entities: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SentenceMassResult:
    sentence_ids: tuple[str, ...]
    sentence_mass: np.ndarray
    active_sentence_ids: tuple[str, ...]
    active_sentence_mass: np.ndarray
    query_anchors: QueryAnchorResult
    diagnostics: dict[str, Any]


def alpha_from_expected_hops(expected_hops: int | float) -> float:
    hops = max(0.0, float(expected_hops))
    return 1.0 / (hops + 1.0)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _fallback_entities(index: SentenceGraphIndex, query: str) -> tuple[str, ...]:
    if not index.entity_ids:
        return tuple()
    query_tokens = _tokens(query)
    scored = []
    for entity_id in index.entity_ids:
        canonical = index.entity_canonicals.get(entity_id, "")
        entity_tokens = _tokens(canonical)
        overlap = len(query_tokens & entity_tokens)
        scored.append((overlap, canonical, entity_id))
    best_overlap = max((row[0] for row in scored), default=0)
    if best_overlap > 0:
        return tuple(row[2] for row in sorted(scored, key=lambda row: (-row[0], row[1], row[2]))[:3])

    ranked_sentences = sorted(
        index.sentence_nodes,
        key=lambda sentence: (-sentence.source_relevance, sentence.sentence_id),
    )
    for sentence in ranked_sentences:
        if sentence.entities:
            return tuple(sentence.entities[:3])
    return tuple(index.entity_ids[: min(3, len(index.entity_ids))])


def query_entity_anchors(index: SentenceGraphIndex, query: str) -> QueryAnchorResult:
    entity_by_canonical = index.entity_by_canonical
    requested: list[str] = []
    anchor_ids: list[str] = []
    for entity in extract_query_entities(query):
        canonical = normalize_content_text(entity.canonical)
        if not canonical:
            continue
        requested.append(canonical)
        entity_id = entity_by_canonical.get(canonical)
        if entity_id is not None and entity_id not in anchor_ids:
            anchor_ids.append(entity_id)

    fallback_used = False
    if not anchor_ids:
        fallback_used = True
        anchor_ids = list(_fallback_entities(index, query))
        requested = [index.entity_canonicals.get(entity_id, entity_id) for entity_id in anchor_ids]

    entity_ids = tuple(anchor_ids)
    diagnostics = {
        "anchor_count": len(entity_ids),
        "anchor_fallback_used": fallback_used,
        "query_anchor_entities": list(requested),
        "query_anchor_entity_ids": list(entity_ids),
    }
    return QueryAnchorResult(
        entity_ids=entity_ids,
        anchor_fallback_used=fallback_used,
        query_anchor_entities=tuple(requested),
        diagnostics=diagnostics,
    )


def _all_graph_node_ids(index: SentenceGraphIndex) -> tuple[str, ...]:
    return tuple(
        [
            *index.entity_ids,
            *index.sentence_ids,
            *(chunk.chunk_node_id for chunk in index.chunk_nodes),
        ]
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=np.float64)
    return values / total


def personalized_pagerank(
    index: SentenceGraphIndex,
    *,
    query_anchor_entity_ids: tuple[str, ...],
    alpha: float,
    include_chunk_parent_edges: bool = False,
    max_iters: int = 100,
    tol: float = 1e-12,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    node_ids = _all_graph_node_ids(index)
    if not node_ids:
        return {}, {"ppr_iterations": 0, "ppr_converged": True}
    pos = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = index.adjacency(include_chunk_parent_edges=include_chunk_parent_edges)

    restart = np.zeros(len(node_ids), dtype=np.float64)
    valid_anchors = [entity_id for entity_id in query_anchor_entity_ids if entity_id in pos]
    if valid_anchors:
        for entity_id in valid_anchors:
            restart[pos[entity_id]] += 1.0 / len(valid_anchors)
    else:
        sentence_positions = [pos[sentence_id] for sentence_id in index.sentence_ids if sentence_id in pos]
        if sentence_positions:
            for idx in sentence_positions:
                restart[idx] = 1.0 / len(sentence_positions)
        else:
            restart[:] = 1.0 / len(node_ids)

    transition = None
    if csr_matrix is not None:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for node_id, idx in pos.items():
            neighbors = [neighbor_id for neighbor_id, _length, _edge_type in adjacency.get(node_id, []) if neighbor_id in pos]
            if not neighbors:
                continue
            share = 1.0 / len(neighbors)
            for neighbor_id in neighbors:
                rows.append(idx)
                cols.append(pos[neighbor_id])
                data.append(share)
        if rows:
            transition = csr_matrix((data, (rows, cols)), shape=(len(node_ids), len(node_ids)), dtype=np.float64)

    rank = restart.copy()
    converged = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        if transition is not None:
            flowed = np.asarray(rank @ transition).ravel()
            dangling_mass = max(0.0, 1.0 - float(flowed.sum()))
        else:
            flowed = np.zeros_like(rank)
            dangling_mass = 0.0
            for node_id, idx in pos.items():
                neighbors = adjacency.get(node_id, [])
                if not neighbors:
                    dangling_mass += rank[idx]
                    continue
                share = rank[idx] / len(neighbors)
                for neighbor_id, _length, _edge_type in neighbors:
                    neighbor_pos = pos.get(neighbor_id)
                    if neighbor_pos is not None:
                        flowed[neighbor_pos] += share
        if dangling_mass:
            flowed += dangling_mass * restart
        new_rank = alpha * restart + (1.0 - alpha) * flowed
        if float(np.abs(new_rank - rank).sum()) <= tol:
            rank = new_rank
            converged = True
            break
        rank = new_rank
    rank = _normalise(rank)
    diagnostics = {
        "ppr_iterations": iterations,
        "ppr_converged": converged,
        "ppr_include_chunk_parent_edges": bool(include_chunk_parent_edges),
        "ppr_sparse_transition": transition is not None,
    }
    return {node_id: float(rank[idx]) for node_id, idx in pos.items()}, diagnostics


def sentence_mass_from_ppr(
    index: SentenceGraphIndex,
    query: str,
    *,
    expected_hops: int | float = 2,
    active_mass_eta: float = 0.995,
    include_chunk_parent_edges_in_ppr: bool = False,
) -> SentenceMassResult:
    if not 0.0 < active_mass_eta <= 1.0:
        raise ValueError("active_mass_eta must be in (0, 1]")
    anchors = query_entity_anchors(index, query)
    alpha = alpha_from_expected_hops(expected_hops)
    ppr, ppr_diagnostics = personalized_pagerank(
        index,
        query_anchor_entity_ids=anchors.entity_ids,
        alpha=alpha,
        include_chunk_parent_edges=include_chunk_parent_edges_in_ppr,
    )
    sentence_ids = index.sentence_ids
    sentence_mass = np.asarray([ppr.get(sentence_id, 0.0) for sentence_id in sentence_ids], dtype=np.float64)
    sentence_mass = _normalise(sentence_mass)

    order = sorted(range(len(sentence_ids)), key=lambda idx: (-float(sentence_mass[idx]), sentence_ids[idx]))
    active: list[int] = []
    cumulative = 0.0
    for idx in order:
        active.append(idx)
        cumulative += float(sentence_mass[idx])
        if cumulative >= active_mass_eta:
            break
    if not active and sentence_ids:
        active = [0]
        cumulative = float(sentence_mass[0])

    active_mass = sentence_mass[active] if active else np.zeros(0, dtype=np.float64)
    active_mass = _normalise(active_mass) if active_mass.size else active_mass
    active_sentence_ids = tuple(sentence_ids[idx] for idx in active)
    diagnostics = {
        "alpha": round(alpha, 6),
        "expected_hops": expected_hops,
        "active_sentence_universe_size": len(active_sentence_ids),
        "active_mass_eta": active_mass_eta,
        "residual_mass": max(0.0, 1.0 - cumulative),
        **anchors.diagnostics,
        **ppr_diagnostics,
    }
    return SentenceMassResult(
        sentence_ids=sentence_ids,
        sentence_mass=sentence_mass,
        active_sentence_ids=active_sentence_ids,
        active_sentence_mass=active_mass,
        query_anchors=anchors,
        diagnostics=diagnostics,
    )
