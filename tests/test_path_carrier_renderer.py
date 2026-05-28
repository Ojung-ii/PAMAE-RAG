from __future__ import annotations

import inspect

import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.rendering.answer_carrier_oracle_renderers import ANSWER_CARRIER_ORACLE_RENDERERS
from pamae_rag.rendering.path_carrier_renderer import (
    METRIC_PATH_CARRIER,
    PATH_CARRIER_RENDERERS,
    render_metric_path_carrier_indices,
)


def _node(node_id: str, *, node_type: str = "chunk", tokens: int = 1) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=f"text for {node_id}",
        embedding=np.zeros(2),
        token_count=tokens,
        node_type=node_type,
    )


def _toy_nodes() -> tuple[EvidenceNode, ...]:
    return (
        _node("anchor", node_type="entity"),
        _node("e1", node_type="entity"),
        _node("c_bridge"),
        _node("e2", node_type="entity"),
        _node("c_medoid"),
    )


def _toy_distance() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0, 1.0, 2.0],
            [3.0, 2.0, 1.0, 0.0, 1.0],
            [4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )


def test_metric_path_carrier_renders_support_tree_chunks_not_entities() -> None:
    result = render_metric_path_carrier_indices(
        nodes=_toy_nodes(),
        selected_medoids=[4],
        query_anchors=[0],
        distance_matrix=_toy_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
    )

    assert result.indices == [4, 2]
    assert result.diagnostics["support_tree_chunk_ids"] == ["c_bridge", "c_medoid"]
    assert "anchor" not in result.diagnostics["path_carrier_order_node_ids"]
    assert "e1" not in result.diagnostics["path_carrier_order_node_ids"]
    assert "e2" not in result.diagnostics["path_carrier_order_node_ids"]


def test_metric_path_carrier_order_is_deterministic() -> None:
    kwargs = dict(
        nodes=_toy_nodes(),
        selected_medoids=[4],
        query_anchors=[0],
        distance_matrix=_toy_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
    )

    first = render_metric_path_carrier_indices(**kwargs)
    second = render_metric_path_carrier_indices(**kwargs)

    assert first.indices == second.indices
    assert first.diagnostics["path_carrier_order_node_ids"] == second.diagnostics["path_carrier_order_node_ids"]


def test_metric_path_carrier_has_no_score_mixing_or_answer_gold_calls() -> None:
    source = inspect.getsource(render_metric_path_carrier_indices).lower()
    for forbidden in ("bm25", "dense", "llm", "answer", "gold", "score ="):
        assert forbidden not in source


def test_metric_path_carrier_logs_budget_cutoff() -> None:
    nodes = (
        _node("anchor", node_type="entity"),
        _node("c_bridge", tokens=2),
        _node("c_medoid", tokens=1),
    )
    distance = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = render_metric_path_carrier_indices(
        nodes=nodes,
        selected_medoids=[2],
        query_anchors=[0],
        distance_matrix=distance,
        max_context_tokens=2,
        max_context_nodes=None,
        disconnected_distance=9.0,
    )

    assert result.indices == [2]
    assert result.diagnostics["budget_cutoff_count"] == 1
    assert result.diagnostics["budget_cutoff_node_ids"] == ["c_bridge"]


def test_path_carrier_oracles_are_not_adoption_renderers() -> None:
    assert "support_tree_answer_oracle" in ANSWER_CARRIER_ORACLE_RENDERERS
    assert ANSWER_CARRIER_ORACLE_RENDERERS.isdisjoint(PATH_CARRIER_RENDERERS)
    assert METRIC_PATH_CARRIER in PATH_CARRIER_RENDERERS
