from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pamae_rag.local_surface.local_surface_graph import LocalSurfaceGraph, shortest_path


@dataclass(frozen=True)
class FactMediatedSelection:
    query_reachable_fact_ids: tuple[str, ...]
    fact_grounding_sentence_ids: tuple[str, ...]
    path_sentence_ids: tuple[str, ...]
    diagnostics: dict[str, Any]


def _neighbors_by_type(graph: LocalSurfaceGraph, node_id: str, edge_type: str) -> tuple[str, ...]:
    return tuple(
        neighbor
        for neighbor, _length, observed_type in graph.adjacency.get(str(node_id), ())
        if observed_type == edge_type
    )


def _path_entities_between_anchors_and_local_entities(
    graph: LocalSurfaceGraph,
    query_anchor_entities: Iterable[str],
) -> tuple[str, ...]:
    entity_set = set(graph.entity_ids)
    out: list[str] = []
    for anchor in sorted(dict.fromkeys(str(value) for value in query_anchor_entities)):
        for target in graph.entity_ids:
            path = shortest_path(graph, anchor, target)
            out.extend(node_id for node_id in path if node_id in entity_set)
    return tuple(dict.fromkeys(out))


def select_fact_mediated_sentences(
    graph: LocalSurfaceGraph,
    *,
    query_anchor_entities: Iterable[str],
    medoid_sentence_ids: Iterable[str],
) -> FactMediatedSelection:
    anchors = tuple(dict.fromkeys(str(value) for value in query_anchor_entities))
    medoids = tuple(dict.fromkeys(str(value) for value in medoid_sentence_ids))
    path_entities = _path_entities_between_anchors_and_local_entities(graph, anchors)
    candidate_facts: set[str] = set()
    for entity_id in (*anchors, *path_entities):
        for fact_id in _neighbors_by_type(graph, entity_id, "fact_entity"):
            candidate_facts.add(fact_id)
    for sentence_id in medoids:
        for fact_id in _neighbors_by_type(graph, sentence_id, "sentence_fact"):
            candidate_facts.add(fact_id)

    sentence_set = set(graph.sentence_ids)
    grounding_sentences: list[str] = []
    path_sentence_ids: list[str] = []
    for fact_id in sorted(candidate_facts):
        sentence_neighbors = [node_id for node_id in _neighbors_by_type(graph, fact_id, "sentence_fact") if node_id in sentence_set]
        if not sentence_neighbors:
            continue
        best_sentence = min(
            sentence_neighbors,
            key=lambda sentence_id: (
                min(
                    (
                        len(shortest_path(graph, anchor, sentence_id)) - 1
                        for anchor in anchors
                        if shortest_path(graph, anchor, sentence_id)
                    ),
                    default=10**9,
                ),
                sentence_id,
            ),
        )
        grounding_sentences.append(best_sentence)
        for anchor in anchors:
            path = shortest_path(graph, anchor, fact_id)
            path_sentence_ids.extend(node_id for node_id in path if node_id in sentence_set)
        path_to_sentence = shortest_path(graph, fact_id, best_sentence)
        path_sentence_ids.extend(node_id for node_id in path_to_sentence if node_id in sentence_set)

    fact_ids = tuple(sorted(candidate_facts))
    grounding = tuple(dict.fromkeys(grounding_sentences))
    path_sentences = tuple(dict.fromkeys(path_sentence_ids))
    return FactMediatedSelection(
        query_reachable_fact_ids=fact_ids,
        fact_grounding_sentence_ids=grounding,
        path_sentence_ids=path_sentences,
        diagnostics={
            "query_reachable_fact_count": len(fact_ids),
            "fact_grounding_sentence_count": len(grounding),
            "fact_path_sentence_count": len(path_sentences),
            "fact_mediated_uses_answer_or_gold": False,
            "fact_mediated_selection_rule": "deterministic_graph_closure",
        },
    )


__all__ = ["FactMediatedSelection", "select_fact_mediated_sentences"]
