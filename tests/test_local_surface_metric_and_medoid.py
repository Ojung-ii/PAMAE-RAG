from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.local_surface.local_metric_distance import distance_between, validate_triangle_inequality
from pamae_rag.local_surface.local_sentence_medoid import LocalMedoidConfig, select_local_sentence_medoids
from pamae_rag.local_surface.local_surface_graph import build_local_surface_graph


def _node(node_id: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.zeros(2),
        metadata={"title": node_id},
    )


def test_metric_distance_and_triangle_inequality() -> None:
    graph = build_local_surface_graph(
        [_node("c0", "Ada Lovelace wrote notes. London hosted Ada Lovelace.")],
        ["c0"],
    )
    left, right = graph.sentence_ids[:2]

    assert distance_between(graph, left, right) < float("inf")
    assert validate_triangle_inequality(graph, graph.sentence_ids) == 0


def test_local_medoid_objective_uses_sentence_medoids_and_tie_breaks() -> None:
    graph = build_local_surface_graph(
        [_node("c0", "Ada Lovelace wrote notes. Ada Lovelace visited London. Ada Lovelace left.")],
        ["c0"],
    )

    result = select_local_sentence_medoids(
        graph,
        "Where did Ada Lovelace visit?",
        config=LocalMedoidConfig(local_sentence_medoids=2),
    )

    assert len(result.selected_sentence_ids) == 2
    assert all(sentence_id.startswith("sent:") for sentence_id in result.selected_sentence_ids)
    assert result.diagnostics["ppr_used_only_as_mass"] is True
    assert result.diagnostics["local_objective_uses_graph_distance"] is True
    assert result.diagnostics["deterministic_tie_break"] == "node_id"
    assert result.diagnostics["local_objective_invalid"] is False
