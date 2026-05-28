from __future__ import annotations

from pamae_rag.diagnostics.representative_taxonomy import (
    classify_representative_failure,
    summarize_representative_taxonomy,
)


def test_representative_taxonomy_separates_renderer_sparsity() -> None:
    payload = {
        "gold_rows": [
            {
                "gold_in_projected": True,
                "gold_in_selected_basin": True,
                "gold_on_existing_support_tree": False,
                "medoid_to_gold_path_exists": True,
                "gold_rendered": False,
                "render_budget_cutoff_before_gold": False,
                "answer_in_context": False,
            }
        ],
        "answer_trace": {"answer_chunk_in_projected": True},
    }

    assert classify_representative_failure(payload, {"f1": 0.0}) == "D_renderer_sparsity"


def test_representative_taxonomy_summarizes_path_gaps() -> None:
    result = summarize_representative_taxonomy(
        [
            {
                "failure_type": "D_renderer_sparsity",
                "mean_d_medoid_gold": 1.5,
                "mean_gold_distance_percentile_within_basin": 0.9,
                "gold_on_support_tree": False,
                "gold_path_exists_but_not_rendered": True,
                "answer_chunk_projected_but_not_rendered": True,
            },
            {
                "failure_type": "G_success",
                "mean_d_medoid_gold": 0.0,
                "mean_gold_distance_percentile_within_basin": 0.1,
                "gold_on_support_tree": True,
                "gold_path_exists_but_not_rendered": False,
                "answer_chunk_projected_but_not_rendered": False,
            },
        ]
    )

    assert result.representative_failure_counts["D_renderer_sparsity"] == 1
    assert result.representative_failure_counts["G_success"] == 1
    assert result.mean_d_medoid_gold == 0.75
    assert result.gold_on_support_tree_rate == 0.5
    assert result.gold_path_exists_but_not_rendered_rate == 0.5

