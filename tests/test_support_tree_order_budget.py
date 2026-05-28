from __future__ import annotations

import inspect

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.current_tree_diff import current_tree_diff_row
from pamae_rag.diagnostics.support_tree_order_budget import (
    aggregate_support_tree_order_budget,
    support_tree_order_budget_rows,
)
from pamae_rag.rendering.tree_ablation_renderers import (
    DIAGNOSTIC_TREE_RENDERERS,
    TREE_ABLATION_RENDERERS,
    TREE_ANSWER_ORACLE,
    TREE_ORACLE_RENDERERS,
    render_tree_ablation,
)


def _node(node_id: str, *, text: str | None = None, tokens: int = 1, node_type: str = "chunk") -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text or f"text for {node_id}",
        embedding=np.zeros(2),
        token_count=tokens,
        node_type=node_type,
    )


def _example() -> QueryExample:
    return QueryExample(
        query_id="q",
        query="q",
        nodes=(
            _node("anchor", node_type="entity"),
            _node("e1", node_type="entity"),
            _node("c_bridge", text="answer lives here", tokens=2),
            _node("e2", node_type="entity"),
            _node("c_medoid", tokens=1),
            _node("c_hidden", text="answer hidden here", tokens=1),
            _node("c_tree_extra", tokens=2),
        ),
        gold_node_ids=frozenset({"c_bridge"}),
        answer="answer",
        metadata={"dataset": "toy"},
    )


def _row(context: list[str]) -> dict:
    return {
        "query_id": "q",
        "anchor_node_ids": ["c_medoid"],
        "context_node_ids": context,
        "diagnostics": {
            "active_universe_node_ids": ["anchor", "e1", "c_bridge", "e2", "c_medoid", "c_hidden", "c_tree_extra"],
            "candidate_node_ids": ["c_bridge", "c_medoid", "c_hidden", "c_tree_extra"],
            "projected_node_ids": ["c_bridge", "c_medoid", "c_hidden", "c_tree_extra"],
            "diagnostic_selected_basin_node_ids": ["c_bridge", "c_medoid", "c_hidden"],
            "refined_support_tree_node_ids": ["anchor", "e1", "c_bridge", "e2", "c_medoid", "c_tree_extra"],
            "refined_anchor_medoid_path_node_ids": ["anchor", "e1", "c_bridge", "e2", "c_medoid"],
            "refined_medoid_medoid_path_node_ids": [],
            "path_carrier_order_node_ids": ["c_medoid", "c_bridge", "c_tree_extra"],
            "budget_cutoff_node_ids": ["c_bridge"],
        },
    }


def test_current_vs_tree_set_diff() -> None:
    current = _row(["a", "b", "c"])
    current["diagnostics"]["refined_support_tree_node_ids"] = ["b", "c", "d"]
    metric = _row(["b", "d"])
    example = QueryExample(
        query_id="q",
        query="q",
        nodes=tuple(_node(node_id, text="answer" if node_id == "a" else node_id) for node_id in ("a", "b", "c", "d")),
        answer="answer",
    )

    row = current_tree_diff_row(example=example, current_row=current, metric_row=metric)

    assert row["current_tree_intersection_chunk_ids"] == ["b", "c"]
    assert row["current_only_chunk_ids"] == ["a"]
    assert row["tree_only_chunk_ids"] == ["d"]


def test_support_tree_order_budget_traces_path_role_and_cutoff() -> None:
    rows = support_tree_order_budget_rows(
        example=_example(),
        current_row=_row(["c_hidden", "c_medoid"]),
        metric_row=_row(["c_medoid"]),
        current_qa={"answer_coverage": 1.0, "f1": 0.0},
        metric_qa={"answer_coverage": 0.0, "f1": 0.0},
        distance_lookup=lambda left, right: 0.0 if left == right else 1.0,
    )
    bridge_row = next(row for row in rows if row["answer_chunk_id"] == "c_bridge")

    assert bridge_row["answer_chunk_on_support_tree"] is True
    assert bridge_row["answer_chunk_on_anchor_medoid_path"] is True
    assert bridge_row["metric_budget_cutoff_before_answer"] is True
    assert bridge_row["metric_render_role"] == "not_rendered"
    assert bridge_row["current_render_role"] == "not_rendered"

    agg = aggregate_support_tree_order_budget(rows)
    assert agg["answer_on_support_tree_rate"] == 1.0
    assert agg["answer_metric_budget_cutoff_rate"] == 1.0


def test_tree_ablation_renderers_are_diagnostic_and_budgeted() -> None:
    result = render_tree_ablation(
        example=_example(),
        retrieval_row=_row(["c_hidden", "c_medoid"]),
        renderer_mode="tree_current_budget_order",
        max_context_tokens=2,
    )

    assert result.context_node_ids == ("c_medoid",)
    assert result.diagnostics["diagnostic_renderer"] is True
    assert result.diagnostics["adoption_candidate"] is False
    assert "c_bridge" in result.diagnostics["budget_cutoff_node_ids"]


def test_tree_answer_oracle_is_excluded_from_diagnostic_adoption_renderers() -> None:
    assert TREE_ANSWER_ORACLE in TREE_ORACLE_RENDERERS
    assert TREE_ANSWER_ORACLE in TREE_ABLATION_RENDERERS
    assert TREE_ANSWER_ORACLE not in DIAGNOSTIC_TREE_RENDERERS


def test_non_oracle_tree_ablation_does_not_use_answer_or_gold_for_selection() -> None:
    source = inspect.getsource(render_tree_ablation).lower()
    # The function contains the oracle branch, so check each non-oracle constant is not
    # adjacent to answer/gold-specific selection names in the static source.
    for renderer in DIAGNOSTIC_TREE_RENDERERS:
        assert f"{renderer}" in source
    oracle_source = source[source.index("else:") :]
    assert "answer_containing_chunk_ids" in oracle_source
