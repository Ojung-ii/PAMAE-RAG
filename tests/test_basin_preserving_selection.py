import numpy as np

from pamae_rag.config import AppConfig, PamaeConfig
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.objective.anchor_objective import anchor_objective
from pamae_rag.pamae.global_search import SearchResult
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.selection.basin_preserving import (
    assign_query_basins,
    eligible_basins,
    gold_ids_in_selected_basins,
    select_basin_preserving_medoids,
)


def test_basin_preserving_selection_covers_eligible_basins_lexicographically():
    distance = np.asarray(
        [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    rho = np.asarray([0.45, 0.05, 0.05, 0.45], dtype=np.float64)
    proposal = [1, 2]
    result = SearchResult(
        anchors=proposal,
        objective=anchor_objective(proposal, distance, rho),
        exact=True,
        num_combinations=1,
    )

    selected = select_basin_preserving_medoids(
        [result],
        candidates=[0, 1, 2, 3],
        k=2,
        distance_matrix=distance,
        rho=rho,
        token_costs=np.ones(4),
        token_weight=0.0,
        anchor_penalty=0.0,
        sample_sizes=[4],
        max_combinations=100,
        node_ids=["a", "b", "c", "d"],
        tau=1.0,
    )

    assert selected.diagnostics["selection_mode"] == "basin_preserving_medoids"
    assert selected.diagnostics["eligible_basin_count"] == 2
    assert selected.diagnostics["covered_basin_count"] == 2
    assert len(selected.anchors) == 2


def test_basin_helpers_use_sampling_expectation_and_gold_only_for_diagnostics():
    distance = np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
    basins = assign_query_basins([0, 1], [0], distance, ["a", "b"])

    assert basins == {0: 0, 1: 0}
    assert eligible_basins({0: 0.25}, sampling_budget=4, tau=1.0) == (0,)

    nodes = [type("Node", (), {"node_id": "a"})(), type("Node", (), {"node_id": "b"})()]
    assert gold_ids_in_selected_basins(
        gold_node_ids=frozenset({"b"}),
        nodes=nodes,
        node_to_basin=basins,
        covered_basin_ids=(0,),
    ) == ["b"]


def test_pipeline_can_run_basin_preserving_variant():
    nodes = (
        EvidenceNode("a", "Ada Lovelace worked with Charles Babbage.", np.asarray([1.0, 0.0]), relevance=0.4),
        EvidenceNode("b", "Charles Babbage built engines.", np.asarray([0.9, 0.1]), relevance=0.3),
        EvidenceNode("c", "A different note about mathematics.", np.asarray([0.0, 1.0]), relevance=0.2),
    )
    example = QueryExample(query_id="q", query="Who worked with Charles Babbage?", nodes=nodes)
    cfg = AppConfig(
        pamae=PamaeConfig(
            retrieval_variant="basin_preserving_medoids",
            renderer="cell_top_rho",
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

    assert result.diagnostics["selection_mode"] == "basin_preserving_medoids"
    assert result.diagnostics["eligible_basin_count"] >= 1
