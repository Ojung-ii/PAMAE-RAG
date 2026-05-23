from pathlib import Path

from pamae_rag.config import load_config


def test_compact_configs_load():
    paths = sorted(Path("configs/ablations").glob("*_compact.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        assert cfg.pamae.max_context_nodes == 8
        assert cfg.pamae.max_context_tokens == 512


def test_compact_config_budget_fields():
    cfg = load_config("configs/ablations/popqa_correctness_20_compact.yaml")
    assert cfg.pamae.k == 2
    assert cfg.pamae.k_max == 2
    assert cfg.pamae.max_context_nodes == 8
    assert cfg.pamae.max_context_tokens == 512
    assert cfg.pamae.evidence_per_anchor == 1
