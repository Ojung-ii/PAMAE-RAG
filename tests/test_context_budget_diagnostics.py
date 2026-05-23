from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.pipeline import run_query_pamae


def test_context_budget_diagnostics():
    cfg = load_config("configs/ablations/popqa_correctness_20_compact.yaml")
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]
    row = run_query_pamae(example, cfg).to_json()
    diagnostics = row["diagnostics"]

    assert diagnostics["max_context_nodes"] == 8
    assert diagnostics["max_context_tokens"] == 512
    assert diagnostics["final_context_nodes"] == len(row["context_node_ids"])
    assert diagnostics["final_context_tokens"] > 0
    assert diagnostics["node_budget_satisfied"]
    assert diagnostics["token_budget_satisfied"]
    assert not diagnostics["node_budget_exceeded_by_anchors"]
    assert diagnostics["context_budget_policy"] == "anchors_then_cell_top_rho_then_score_fill"

