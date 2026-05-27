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


def test_dq_connectivity_config_loads():
    paths = sorted(Path("configs/ablations_dq_connectivity").glob("*.yaml"))
    assert len(paths) == 10
    modes = set()
    for path in paths:
        cfg = load_config(path)
        modes.add(cfg.pamae.graph.backbone.mode)
        assert cfg.pamae.distance_mode == "graph_sp"
        assert cfg.pamae.graph.enabled is True
        assert cfg.pamae.graph.backbone.k in {4, 8}
        assert cfg.pamae.graph.backbone.max_edges_per_node == 32
    assert modes == {"knn", "mutual_knn", "none"}


def test_dq_connectivity_retrieval_config_loads():
    paths = sorted(Path("configs/ablations_dq_connectivity_retrieval").glob("*.yaml"))
    assert len(paths) == 12
    modes = set()
    retrieval_variants = set()
    for path in paths:
        cfg = load_config(path)
        modes.add(cfg.pamae.distance_mode)
        retrieval_variants.add(cfg.pamae.retrieval_variant)
        assert cfg.pamae.relevance_mode == "title_aware"
        assert cfg.pamae.max_context_tokens == 512
        assert cfg.pamae.max_context_nodes == 8
    assert modes == {"graph_sp", "hybrid_sem_graph", "semantic"}
    assert retrieval_variants == {"sample_full_validation_refine_cell_renderer", "top_rho"}


def test_content_graph_configs_load():
    paths = sorted(Path("configs/content_graph").glob("*.yaml"))
    assert len(paths) == 2
    for path in paths:
        cfg = load_config(path)
        assert cfg.pamae.graph.source == "content"
        assert cfg.pamae.relevance_mode == "current"
        assert cfg.pamae.max_context_tokens == 512
        assert cfg.pamae.max_context_nodes == 8
