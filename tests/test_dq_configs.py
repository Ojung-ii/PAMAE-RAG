from pathlib import Path

from pamae_rag.config import load_config


def test_distance_mode_config_loads():
    paths = sorted(Path("configs/ablations_dq").glob("*.yaml"))
    assert len(paths) == 10
    modes = set()
    for path in paths:
        cfg = load_config(path)
        modes.add(cfg.pamae.distance_mode)
        assert cfg.pamae.relevance_mode == "title_aware"
        assert cfg.pamae.max_context_tokens == 512
        assert cfg.pamae.max_context_nodes == 8
        if cfg.pamae.distance_mode != "semantic":
            assert cfg.pamae.graph.enabled is True
            assert cfg.pamae.graph.edge_lengths.same_canonical_title >= 0
    assert modes == {"semantic", "graph_sp", "hybrid_sem_graph"}
