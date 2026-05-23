from pamae_rag.data.io import read_jsonl
from pamae_rag.eval.support_recall import evaluate_predictions


def test_evaluator_anchor_ids_fallback():
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    prediction = {
        "query_id": example.query_id,
        "anchor_ids": [next(iter(example.gold_node_ids))],
        "context_node_ids": list(example.gold_node_ids),
        "objective_before_refinement": 1.0,
        "objective_after_refinement": 0.5,
        "latency_ms": 12.0,
        "diagnostics": {"refinement_accepted": True},
    }

    metrics = evaluate_predictions([example], {example.query_id: prediction}).to_json()
    assert metrics["missing_anchor_key_count"] == 0
    assert metrics["anchor_non_empty_ratio"] == 1.0
    assert metrics["mean_anchor_hit"] == 1.0
    assert metrics["avg_latency_ms"] == 12.0


def test_evaluator_counts_missing_anchor_keys():
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    prediction = {
        "query_id": example.query_id,
        "context_node_ids": [],
        "diagnostics": {},
    }

    metrics = evaluate_predictions([example], {example.query_id: prediction}).to_json()
    assert metrics["missing_anchor_key_count"] == 1
    assert metrics["anchor_non_empty_ratio"] == 0.0

