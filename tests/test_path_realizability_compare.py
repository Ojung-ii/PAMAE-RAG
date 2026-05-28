from __future__ import annotations

from scripts.compare_path_realizability_runs import _gate


def test_path_neighborhood_gate_requires_renderer_sparsity_decrease() -> None:
    ref = {
        "run": "current_content_current_renderer",
        "F1": 0.1,
        "oracle_gap": 0.1,
        "rendered_recall": 0.5,
        "answer_in_context": 0.5,
        "context_f1": 0.2,
        "C": 1,
        "D": 4,
        "oracle": {"oracle_dominance_valid": True},
    }
    row = {
        **ref,
        "run": "current_content_path_neighborhood_renderer",
        "F1": 0.11,
        "oracle_gap": 0.09,
        "D": 4,
    }

    decision, blockers = _gate(row, ref)

    assert decision == "STOP"
    assert "renderer_sparsity_not_reduced" in blockers

