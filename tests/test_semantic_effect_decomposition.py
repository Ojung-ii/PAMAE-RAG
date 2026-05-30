from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pamae_rag.diagnostics.semantic_effect_decomposition import (
    decomposition_deltas,
    prompt_protocol_status,
    same_query_sample,
)
from pamae_rag.qa.generator import COMMON_QA_PROMPTS, PROMPT_HASH, PROMPT_ID, PROMPT_TEXT, PROMPT_TEXT_EXACT_MATCH
from pamae_rag.semantic.compatible_embedding_cache import make_metadata


def test_common_qa_prompt_is_fixed_and_hashable() -> None:
    assert PROMPT_ID == "common_qa"
    assert PROMPT_TEXT == COMMON_QA_PROMPTS["common_qa"]
    assert PROMPT_TEXT_EXACT_MATCH is True
    assert PROMPT_HASH == hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()
    assert PROMPT_HASH == "31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d"


def test_prompt_protocol_status_requires_common_hash_and_exact_match() -> None:
    rows = [
        {
            "qa_prompt_name": "common_qa",
            "qa_prompt_hash": "abc",
            "qa_prompt_text_exact_match": True,
        },
        {
            "qa_prompt_name": "common_qa",
            "qa_prompt_hash": "abc",
            "qa_prompt_text_exact_match": True,
        },
    ]

    status = prompt_protocol_status(rows)

    assert status["qa_prompt_consistent"] is True
    assert status["qa_prompt_name"] == "common_qa"
    assert status["qa_prompt_hash"] == "abc"


def test_decomposition_deltas_use_requested_baselines() -> None:
    rows = [
        {
            "renderer_mode": "metric_path_carrier",
            "answer_in_context": 0.2,
            "rendered_recall": 0.3,
            "context_f1": 0.4,
            "qa_f1": 0.5,
            "avg_context_tokens": 100.0,
        },
        {
            "renderer_mode": "tree_shell1_graph_order",
            "answer_in_context": 0.4,
            "rendered_recall": 0.5,
            "context_f1": 0.6,
            "qa_f1": 0.7,
            "avg_context_tokens": 120.0,
        },
        {
            "renderer_mode": "tree_shell1_semantic_query_order",
            "answer_in_context": 0.45,
            "rendered_recall": 0.55,
            "context_f1": 0.62,
            "qa_f1": 0.72,
            "avg_context_tokens": 118.0,
        },
        {
            "renderer_mode": "tree_shell1_semantic_tree_order",
            "answer_in_context": 0.35,
            "rendered_recall": 0.45,
            "context_f1": 0.58,
            "qa_f1": 0.68,
            "avg_context_tokens": 115.0,
        },
    ]

    deltas = decomposition_deltas(rows)

    assert deltas["delta_shell_B1_minus_A1"]["answer_in_context"] == 0.2
    assert deltas["delta_query_semantic_B2_minus_B1"]["qa_f1"] == 0.020000000000000018
    assert deltas["delta_tree_semantic_B3_minus_B1"]["tokens"] == -5.0


def test_same_query_sample_checks_identical_order(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    rows = [{"query_id": "q1"}, {"query_id": "q2"}]
    left.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    right.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    status = same_query_sample([left, right])

    assert status["same_sample"] is True
    assert status["sample_size"] == 2


def test_nv_embed_metadata_compatibility_contract() -> None:
    metadata = make_metadata(
        model_id="nvidia/NV-Embed-v2",
        model_revision="3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
        embedding_dim=4096,
        dataset="2wikimultihopqa",
        pooling="nv_embed_encode",
    )

    assert metadata.model_id == "nvidia/NV-Embed-v2"
    assert metadata.embedding_dim == 4096
    assert metadata.normalized is True
    assert metadata.chunk_text_format == "Title: <title>\nText: <chunk_text>"
    assert metadata.query_text_format == "raw_question"
