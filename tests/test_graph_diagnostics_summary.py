import json

from scripts.summarize_graph_diagnostics import summarize


def test_graph_diagnostics_summary(tmp_path):
    run_dir = tmp_path / "hotpotqa" / "dq_connectivity" / "graph_knn4"
    run_dir.mkdir(parents=True)
    (run_dir / "graph_diagnostics.json").write_text(
        json.dumps(
            {
                "num_queries": 1,
                "avg_num_nodes": 3,
                "avg_num_edges": 2,
                "avg_symbolic_edges": 0,
                "avg_backbone_edges": 2,
                "avg_degree": 1.33,
                "max_degree_mean": 2,
                "avg_num_connected_components": 1,
                "avg_largest_component_ratio": 1.0,
                "avg_connected_pair_rate": 1.0,
                "avg_disconnected_pair_rate": 0.0,
                "gold_support_connected_rate": 1.0,
                "gold_support_same_component_rate": 1.0,
                "gold_support_avg_shortest_path_distance": 0.2,
                "backbone_missing_embedding_count": 0,
                "avg_edge_counts_by_type": {"semantic_knn": 2},
                "avg_edge_length_by_type": {"semantic_knn": 0.1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = summarize(tmp_path / "hotpotqa" / "dq_connectivity")
    assert rows[0]["dataset"] == "hotpotqa"
    assert rows[0]["variant"] == "graph_knn4"
    assert rows[0]["avg_backbone_edges"] == 2
