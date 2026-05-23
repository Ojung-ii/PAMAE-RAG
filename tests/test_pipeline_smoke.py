from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.pipeline import run_query_pamae


def test_pipeline_smoke_runs():
    cfg = load_config("configs/smoke.yaml")
    examples = read_jsonl("data/smoke/examples.jsonl")
    result = run_query_pamae(examples[0], cfg)
    assert len(result.anchor_ids) == cfg.pamae.k
    assert result.context_node_ids
    assert result.objective_after_refinement <= result.objective_before_refinement + 1e-12
    assert result.support_recall is not None
