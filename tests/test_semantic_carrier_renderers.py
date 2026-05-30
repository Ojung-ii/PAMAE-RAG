from __future__ import annotations

import inspect

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.runtime_profile import context_text_hash
from pamae_rag.semantic.angular_distance import angular_distance
from pamae_rag.semantic.embedding_store import EmbeddingStore
from pamae_rag.rendering.semantic_carrier_renderers import (
    SEMANTIC_ADOPTION_CANDIDATE_RENDERERS,
    SEMANTIC_ORACLE_RENDERERS,
    SHELL1_ANSWER_ORACLE,
    TREE_SHELL1_GRAPH_ORDER,
    TREE_SHELL1_SEMANTIC_QUERY_ORDER,
    render_semantic_carrier_indices,
    semantic_query_order_keys,
)


def _node(node_id: str, *, embedding: list[float], text: str | None = None, node_type: str = "chunk") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text or f"text {node_id}",
        embedding=np.asarray(embedding, dtype=float),
        token_count=1,
        node_type=node_type,
    )


def _example() -> QueryExample:
    return QueryExample(
        query_id="q",
        query="q",
        nodes=(
            _node("anchor", embedding=[1.0, 0.0], node_type="entity"),
            _node("c_tree", embedding=[1.0, 0.0]),
            _node("c_shell1_answer", embedding=[1.0, 0.0], text="answer lives here"),
            _node("c_shell1_other", embedding=[0.0, 1.0]),
            _node("c_shell2", embedding=[0.0, 1.0]),
        ),
        answer="answer",
        metadata={"query_embedding": [1.0, 0.0]},
    )


def _distance() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 2.0, 3.0],
            [2.0, 1.0, 2.0, 0.0, 3.0],
            [3.0, 2.0, 3.0, 3.0, 0.0],
        ]
    )


def test_semantic_renderer_draws_only_from_tree_union_shell1() -> None:
    result = render_semantic_carrier_indices(
        example=_example(),
        selected_medoids=[1],
        query_anchors=[0],
        distance_matrix=_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
        renderer_mode=TREE_SHELL1_GRAPH_ORDER,
    )

    rendered_ids = result.diagnostics["semantic_carrier_order_node_ids"]
    assert "c_tree" in rendered_ids
    assert "c_shell1_answer" in rendered_ids
    assert "c_shell1_other" in rendered_ids
    assert "c_shell2" not in rendered_ids
    assert result.diagnostics["graph_defined_pool_only"] is True


def test_semantic_query_order_is_lexicographic_not_weighted_mixing() -> None:
    result = render_semantic_carrier_indices(
        example=_example(),
        selected_medoids=[1],
        query_anchors=[0],
        distance_matrix=_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
        renderer_mode=TREE_SHELL1_SEMANTIC_QUERY_ORDER,
    )

    order = result.diagnostics["semantic_carrier_order_node_ids"]
    assert order.index("c_shell1_answer") < order.index("c_shell1_other")
    assert result.diagnostics["score_mixing_detected"] is False


def test_semantic_renderer_static_no_score_mixing_or_answer_gold_calls() -> None:
    source = inspect.getsource(render_semantic_carrier_indices).lower()
    for forbidden in ("bm25", "dense", "llm", "score ="):
        assert forbidden not in source
    non_oracle_source = source[: source.index("if renderer_mode == shell1_answer_oracle")]
    assert "answer_containing_chunk_ids" not in non_oracle_source


def test_semantic_oracle_is_not_adoption_candidate() -> None:
    assert SHELL1_ANSWER_ORACLE in SEMANTIC_ORACLE_RENDERERS
    assert SHELL1_ANSWER_ORACLE not in SEMANTIC_ADOPTION_CANDIDATE_RENDERERS


def test_b2_production_mode_keeps_context_and_drops_heavy_trace() -> None:
    kwargs = dict(
        example=_example(),
        selected_medoids=[1],
        query_anchors=[0],
        distance_matrix=_distance(),
        max_context_tokens=10,
        max_context_nodes=None,
        disconnected_distance=9.0,
        renderer_mode=TREE_SHELL1_SEMANTIC_QUERY_ORDER,
    )
    diagnostic = render_semantic_carrier_indices(**kwargs, include_trace=True)
    production = render_semantic_carrier_indices(**kwargs, include_trace=False)

    assert production.indices == diagnostic.indices
    assert context_text_hash([_example().nodes[idx].text for idx in production.indices]) == context_text_hash(
        [_example().nodes[idx].text for idx in diagnostic.indices]
    )
    assert "semantic_carrier_order_node_ids" in diagnostic.diagnostics
    assert "semantic_carrier_order_node_ids" not in production.diagnostics
    assert production.diagnostics["candidate_pool_size"] == diagnostic.diagnostics["candidate_pool_size"]


def test_b2_angular_order_matches_dot_monotonic_order() -> None:
    example = _example()
    store = EmbeddingStore.from_example(example)
    candidates = [2, 3]
    angular_order = sorted(
        candidates,
        key=lambda idx: (angular_distance(store.query_embedding, store.node_embedding(example.nodes[idx].node_id)), example.nodes[idx].node_id),
    )
    dot_order_keys = semantic_query_order_keys(store=store, candidate_indices=candidates, nodes=example.nodes)
    dot_order = sorted(candidates, key=lambda idx: (dot_order_keys[idx], example.nodes[idx].node_id))
    assert dot_order == angular_order
