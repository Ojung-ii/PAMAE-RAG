from pathlib import Path

from pamae_rag.config import load_config


def test_ablation_config_loads():
    paths = sorted(Path("configs/ablations").glob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        assert cfg.pamae.retrieval_variant
        assert cfg.pamae.renderer
        assert cfg.pamae.relevance_mode
