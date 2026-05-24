from pathlib import Path

from pamae_rag.config import load_config


def test_rho_ablation_configs_load():
    paths = sorted(Path("configs/ablations_rho").glob("*.yaml"))
    assert len(paths) == 16
    modes = set()
    variants = set()
    for path in paths:
        cfg = load_config(path)
        modes.add(cfg.pamae.relevance_mode)
        variants.add(cfg.pamae.retrieval_variant)
        assert cfg.pamae.max_context_tokens == 512
        assert cfg.pamae.max_context_nodes == 8

    assert modes == {"current", "title_aware", "entity_title_aware", "hybrid_title_semantic"}
    assert variants == {"top_rho", "sample_full_validation_refine_cell_renderer"}
