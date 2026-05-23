from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.pipeline import run_query_pamae


def test_prediction_output_schema_contains_anchor_node_ids():
    cfg = load_config("configs/smoke.yaml")
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    row = run_query_pamae(example, cfg).to_json()

    required = {
        "query_id",
        "anchor_node_ids",
        "anchor_ids",
        "context_node_ids",
        "objective_before_refinement",
        "objective_after_refinement",
        "support_recall",
        "support_hit",
        "exact_phase1",
        "diagnostics",
        "latency_ms",
    }
    assert required <= row.keys()
    assert row["anchor_node_ids"] == row["anchor_ids"]
    assert row["anchor_node_ids"]

