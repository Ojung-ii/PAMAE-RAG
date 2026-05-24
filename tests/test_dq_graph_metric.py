import numpy as np

from pamae_rag.config import AppConfig, GraphConfig, PamaeConfig
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.query_graph import (
    build_minimal_query_graph,
    canonical_title,
    extract_query_spans,
)
from pamae_rag.pipeline import run_query_pamae


class TrapDict(dict):
    def get(self, key, default=None):
        if key in {"gold_node_ids", "is_supporting", "possible_answers", "obj", "o_wiki_title"}:
            raise AssertionError(f"gold leakage key accessed: {key}")
        return super().get(key, default)


def _node(node_id: str, title: str, text: str, embedding=None) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.asarray(embedding if embedding is not None else [1.0, 0.0], dtype=np.float64),
        relevance=1.0,
        token_count=8,
        metadata=TrapDict({"title": title, "gold_node_ids": ["blocked"], "obj": "blocked"}),
    )


def test_canonical_title():
    assert canonical_title("George Rankin (politician)") == "george rankin"
    assert canonical_title("List of George Rankin works") == "george rankin works"


def test_extract_query_spans():
    spans = extract_query_spans("Who was George Rankin's party in 'Ontario'?")
    assert "george rankin" in spans
    assert "ontario" in spans
    assert "who" not in spans


def test_query_graph_no_gold_fields():
    nodes = (
        _node("a", "George Rankin", "George Rankin served in Ontario."),
        _node("b", "Ontario", "Ontario mentions George Rankin."),
    )
    graph = build_minimal_query_graph(
        nodes,
        "Who was George Rankin in Ontario?",
        edge_lengths={"same_canonical_title": 0.25, "title_mention": 0.5, "shared_query_span": 0.75},
        max_edges_per_node=4,
    )
    assert graph.num_edges > 0


def test_query_graph_edges_nonnegative():
    nodes = (
        _node("a", "George Rankin", "George Rankin served in Ontario."),
        _node("b", "George Rankin", "Another George Rankin node."),
        _node("c", "Ontario", "Ontario mentions George Rankin."),
    )
    graph = build_minimal_query_graph(
        nodes,
        "Who was George Rankin in Ontario?",
        edge_lengths={"same_canonical_title": 0.25, "title_mention": 0.5, "shared_query_span": 0.75},
        max_edges_per_node=4,
    )
    assert graph.num_edges > 0
    assert all(edge.length >= 0 for edge in graph.edges)


def test_graph_distance_symmetric_zero_diag():
    nodes = (
        _node("a", "George Rankin", "George Rankin served in Ontario.", [1.0, 0.0]),
        _node("b", "George Rankin", "Another George Rankin node.", [0.9, 0.1]),
    )
    semantic = np.array([[0.0, 0.4], [0.4, 0.0]])
    result = build_graph_aware_distance_matrix(
        nodes,
        "Who was George Rankin?",
        semantic,
        distance_mode="graph_sp",
        distance_weights={"semantic": 0.0, "graph": 1.0},
        graph_config=GraphConfig(enabled=True),
    )
    assert np.allclose(result.distance_matrix, result.distance_matrix.T)
    assert np.allclose(np.diag(result.distance_matrix), 0.0)


def test_graph_distance_disconnected_finite():
    nodes = (
        _node("a", "Alpha", "Alpha only.", [1.0, 0.0]),
        _node("b", "Beta", "Beta only.", [0.0, 1.0]),
    )
    semantic = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = build_graph_aware_distance_matrix(
        nodes,
        "Unrelated query",
        semantic,
        distance_mode="graph_sp",
        distance_weights={"semantic": 0.0, "graph": 1.0},
        graph_config=GraphConfig(enabled=True, disconnected_distance=2.5),
    )
    assert result.distance_matrix[0, 1] == 2.5
    assert np.isfinite(result.distance_matrix).all()


def test_hybrid_distance_nonnegative():
    nodes = (
        _node("a", "George Rankin", "George Rankin served in Ontario.", [1.0, 0.0]),
        _node("b", "Ontario", "Ontario mentions George Rankin.", [0.0, 1.0]),
    )
    semantic = np.array([[0.0, 0.8], [0.8, 0.0]])
    result = build_graph_aware_distance_matrix(
        nodes,
        "Who was George Rankin in Ontario?",
        semantic,
        distance_mode="hybrid_sem_graph",
        distance_weights={"semantic": 0.5, "graph": 0.5},
        graph_config=GraphConfig(enabled=True),
    )
    assert np.all(result.distance_matrix >= 0)


def test_pipeline_uses_distance_mode():
    nodes = (
        _node("a", "George Rankin", "George Rankin served in Ontario.", [1.0, 0.0]),
        _node("b", "Ontario", "Ontario mentions George Rankin.", [0.9, 0.1]),
        _node("c", "Rankin", "A surname list.", [0.0, 1.0]),
        _node("d", "Other", "Other text.", [0.1, 0.9]),
    )
    example = QueryExample(query_id="q", query="Who was George Rankin in Ontario?", nodes=nodes)
    cfg = AppConfig(
        pamae=PamaeConfig(
            retrieval_variant="sample_full_validation_refine_cell_renderer",
            renderer="cell_top_rho",
            relevance_mode="title_aware",
            distance_mode="graph_sp",
            graph=GraphConfig(enabled=True),
            k=2,
            k_max=2,
            num_samples=1,
            sample_size_per_k=3,
            sample_size_cap=4,
            max_context_tokens=64,
            max_context_nodes=4,
            evidence_per_anchor=1,
        )
    )
    result = run_query_pamae(example, cfg)
    assert result.diagnostics["distance_mode"] == "graph_sp"
    assert result.diagnostics["num_edges"] > 0
