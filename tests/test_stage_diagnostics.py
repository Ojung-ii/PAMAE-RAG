from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.eval.stage_diagnostics import STAGE_NAMES
from pamae_rag.pipeline import run_query_pamae


def test_retrieval_records_stage_diagnostics():
    cfg = load_config("configs/smoke.yaml")
    example = read_jsonl("data/smoke/examples.jsonl", limit=1)[0]

    row = run_query_pamae(example, cfg).to_json()

    stages = row["diagnostics"]["stage_diagnostics"]
    retrieval_stages = set(STAGE_NAMES) - {"final_qa"}
    assert retrieval_stages <= set(stages)
    assert stages["content_graph_projection"]["status"] == "not_configured"
    assert stages["context_rendering"]["rendered_recall"] is not None
    assert "support_fact_count" in stages["context_rendering"]
