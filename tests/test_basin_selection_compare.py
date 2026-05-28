import json
from pathlib import Path

from scripts.compare_basin_selection_runs import compare


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(root: Path, name: str, *, f1: float, type_b: int, answer: float = 1.0) -> Path:
    run = root / name
    _write(
        run / "qa_metrics.json",
        {
            "mean_f1": f1,
            "mean_context_f1": 0.5,
            "mean_answer_coverage": answer,
            "avg_retrieval_ms": 1.0,
            "avg_generation_ms": 1.0,
            "stage_diagnostics": {
                "context_rendering": {"mean": {"rendered_recall": 0.5}},
            },
        },
    )
    _write(run / "retrieval_metrics.json", {})
    _write(
        run / "failure_taxonomy.json",
        {
            "failure_type_counts": {"selection_miss": type_b},
            "rows": [{"selected_basin_hit": True}, {"selected_basin_hit": False}],
        },
    )
    return run


def test_compare_basin_selection_gates_type_b_reduction(tmp_path: Path):
    ref = _run(tmp_path, "current_content", f1=0.1, type_b=4)
    candidate = _run(tmp_path, "basin", f1=0.1, type_b=2)
    oracle = tmp_path / "oracle.json"
    _write(
        oracle,
        {
            "gold_support_f1": 0.2,
            "answer_containing_f1": 0.3,
            "answer_copy_f1": 0.8,
            "oracle_dominance_valid": True,
        },
    )

    result = compare([ref, candidate], oracle)

    assert result["risk_gates"][1]["decision"] == "PASS"
