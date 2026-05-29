from __future__ import annotations

from pamae_rag.diagnostics.b2_robustness import flattened_delta_means, paired_deltas


def test_paired_deltas_align_by_query_id_and_report_sign_counts() -> None:
    baseline = [
        {"query_id": "q1", "f1": 0.1, "answer_coverage": 0.0, "context_recall": 0.2, "context_f1": 0.3, "context_tokens": 10.0, "retrieval_ms": 5.0},
        {"query_id": "q2", "f1": 0.2, "answer_coverage": 1.0, "context_recall": 0.4, "context_f1": 0.5, "context_tokens": 20.0, "retrieval_ms": 6.0},
    ]
    candidate = [
        {"query_id": "q2", "f1": 0.3, "answer_coverage": 1.0, "context_recall": 0.7, "context_f1": 0.6, "context_tokens": 25.0, "retrieval_ms": 8.0},
        {"query_id": "q1", "f1": 0.1, "answer_coverage": 1.0, "context_recall": 0.1, "context_f1": 0.2, "context_tokens": 9.0, "retrieval_ms": 4.0},
    ]

    result = paired_deltas(candidate_rows=candidate, baseline_rows=baseline, bootstrap_samples=0)

    assert result["num_pairs"] == 2
    assert result["query_ids_match"] is True
    assert result["metrics"]["qa_f1"]["improved"] == 1
    assert result["metrics"]["qa_f1"]["tied"] == 1
    assert result["metrics"]["rendered_recall"]["regressed"] == 1
    assert flattened_delta_means(result)["answer_in_context_mean"] == 0.5


def test_paired_deltas_only_use_shared_queries() -> None:
    baseline = [{"query_id": "q1", "f1": 0.0}]
    candidate = [{"query_id": "q1", "f1": 1.0}, {"query_id": "q2", "f1": 1.0}]

    result = paired_deltas(candidate_rows=candidate, baseline_rows=baseline, bootstrap_samples=0)

    assert result["num_pairs"] == 1
    assert result["query_ids_match"] is False
    assert result["metrics"]["qa_f1"]["mean"] == 1.0
