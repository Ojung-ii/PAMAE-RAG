import numpy as np

from pamae_rag.config import AppConfig, PamaeConfig
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.rendering.basin_aware_renderer import render_basin_path_closure_indices


def _node(node_id: str, text: str = "text") -> EvidenceNode:
    return EvidenceNode(node_id, text, np.asarray([1.0]), token_count=1)


def test_basin_path_closure_renders_medoid_and_bridge():
    nodes = [_node("a"), _node("bridge"), _node("m")]
    distance = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    result = render_basin_path_closure_indices(
        nodes,
        [2],
        distance_matrix=distance,
        rho=np.asarray([0.2, 0.3, 0.5]),
        max_context_tokens=3,
        max_context_nodes=3,
        node_to_basin={0: 0, 1: 0, 2: 0},
        covered_basin_masses={0: 1.0},
    )

    assert result.indices == [2, 0, 1]
    assert result.diagnostics["renderer_mode"] == "basin_path_closure"
    assert result.diagnostics["rendered_bridge_chunk_count"] == 2


def test_pipeline_can_run_basin_path_closure_renderer():
    nodes = (
        EvidenceNode(
            "a",
            "Ada Lovelace worked with Charles Babbage.",
            np.asarray([1.0, 0.0]),
            relevance=0.4,
            token_count=5,
        ),
        EvidenceNode(
            "b",
            "Charles Babbage built engines.",
            np.asarray([0.9, 0.1]),
            relevance=0.3,
            token_count=4,
        ),
        EvidenceNode(
            "c",
            "A different note about mathematics.",
            np.asarray([0.0, 1.0]),
            relevance=0.2,
            token_count=5,
        ),
    )
    example = QueryExample(
        query_id="q",
        query="Who worked with Charles Babbage?",
        nodes=nodes,
        answer="Ada Lovelace",
    )
    cfg = AppConfig(
        pamae=PamaeConfig(
            retrieval_variant="basin_preserving_medoids",
            renderer="basin_path_closure",
            relevance_mode="current",
            k=2,
            k_max=2,
            num_samples=1,
            sample_size_per_k=3,
            sample_size_cap=3,
            max_context_tokens=64,
            max_context_nodes=3,
            evidence_per_anchor=1,
        )
    )

    result = run_query_pamae(example, cfg)

    assert result.diagnostics["renderer_mode"] == "basin_path_closure"
    assert "answer_in_context" in result.diagnostics["stage_diagnostics"]["context_rendering"]
