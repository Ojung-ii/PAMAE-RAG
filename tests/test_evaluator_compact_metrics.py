from math import isclose

from pamae_rag.data.io import read_jsonl
from pamae_rag.eval.support_recall import evaluate_predictions


def test_evaluator_context_precision_f1():
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    prediction = {
        "query_id": example.query_id,
        "anchor_node_ids": ["q1_c1", "q1_n1"],
        "context_node_ids": ["q1_c1", "q1_c2", "q1_n1"],
        "diagnostics": {
            "final_context_tokens": 100,
            "node_budget_satisfied": True,
            "token_budget_satisfied": True,
        },
    }

    metrics = evaluate_predictions([example], {example.query_id: prediction}).to_json()

    assert isclose(metrics["mean_context_recall"], 2 / 3)
    assert isclose(metrics["mean_context_precision"], 2 / 3)
    assert isclose(metrics["mean_context_f1"], 2 / 3)
    assert isclose(metrics["mean_anchor_recall"], 1 / 3)
    assert isclose(metrics["mean_anchor_precision"], 1 / 2)
    assert isclose(metrics["mean_anchor_f1"], 0.4)


def test_evaluator_recall_per_node_and_per_token():
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    prediction = {
        "query_id": example.query_id,
        "anchor_node_ids": ["q1_c1"],
        "context_node_ids": ["q1_c1", "q1_c2"],
        "diagnostics": {
            "final_context_tokens": 100,
            "node_budget_satisfied": True,
            "token_budget_satisfied": True,
        },
    }

    metrics = evaluate_predictions([example], {example.query_id: prediction}).to_json()

    assert isclose(metrics["mean_context_recall_per_node"], (2 / 3) / 2)
    assert isclose(metrics["mean_context_recall_per_1k_tokens"], (2 / 3) / 0.1)
    assert metrics["avg_context_tokens"] == 100
    assert metrics["context_node_budget_satisfied_rate"] == 1.0
    assert metrics["context_token_budget_satisfied_rate"] == 1.0

