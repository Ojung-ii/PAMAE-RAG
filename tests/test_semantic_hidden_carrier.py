from __future__ import annotations

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.semantic_hidden_carrier import (
    aggregate_semantic_hidden_carrier,
    semantic_hidden_carrier_rows,
)


def _node(node_id: str, text: str, embedding: list[float]) -> EvidenceNode:
    return EvidenceNode(node_id=node_id, text=text, embedding=np.asarray(embedding, dtype=float), token_count=1)


def test_semantic_hidden_carrier_groups_current_only_and_tree_chunks() -> None:
    example = QueryExample(
        query_id="q",
        query="who",
        nodes=(
            _node("c_answer_current", "the answer is nile", [1.0, 0.0]),
            _node("c_non_answer_current", "other current text", [0.0, 1.0]),
            _node("c_answer_tree", "nile appears on tree", [1.0, 0.0]),
            _node("c_tree_other", "tree filler", [0.5, 0.5]),
            _node("c_projected_answer", "nile projected", [1.0, 0.0]),
            _node("c_shell_other", "shell filler", [0.0, 1.0]),
        ),
        gold_node_ids=frozenset({"c_answer_tree"}),
        answer="nile",
        metadata={"query_embedding": [1.0, 0.0]},
    )
    row = {
        "context_node_ids": ["c_answer_current", "c_non_answer_current"],
        "diagnostics": {
            "refined_support_tree_node_ids": ["c_answer_tree", "c_tree_other"],
            "projected_node_ids": ["c_projected_answer"],
        },
    }

    rows = semantic_hidden_carrier_rows(
        example=example,
        current_row=row,
        shell1_chunk_ids=["c_projected_answer", "c_shell_other"],
    )
    by_group = {item["group"]: item for item in rows}

    assert by_group["current_only_answer"]["chunk_id"] == "c_answer_current"
    assert by_group["current_only_non_answer"]["chunk_id"] == "c_non_answer_current"
    assert by_group["tree_answer"]["on_support_tree"] is True
    assert by_group["shell1_answer"]["chunk_id"] == "c_projected_answer"
    assert by_group["shell1_non_answer"]["chunk_id"] == "c_shell_other"
    assert by_group["projected_nonrendered_answer"]["current_rendered"] is False
    assert by_group["current_only_answer"]["d_ang_query_chunk"] == 0.0


def test_semantic_hidden_carrier_reports_missing_query_signal() -> None:
    example = QueryExample(
        query_id="q",
        query="who",
        nodes=(
            _node("c_answer_current", "the answer is nile", [1.0, 0.0]),
            _node("c_non_answer_current", "other current text", [0.0, 1.0]),
        ),
        answer="nile",
    )
    row = {
        "context_node_ids": ["c_answer_current", "c_non_answer_current"],
        "diagnostics": {"refined_support_tree_node_ids": []},
    }

    aggregate = aggregate_semantic_hidden_carrier(semantic_hidden_carrier_rows(example=example, current_row=row))

    assert aggregate["semantic_query_signal_available"] is False
    assert aggregate["semantic_separation_query"] is None
