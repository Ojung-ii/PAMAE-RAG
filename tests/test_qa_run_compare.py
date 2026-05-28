import json
from math import isclose
from pathlib import Path

from pamae_rag.eval.qa_run_compare import QARunSpec, compare_runs


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(
    f1: float,
    *,
    oracle: bool = False,
    recall: float = 1.0,
    answer_coverage: float | None = None,
    selected_answer_coverage: float | None = None,
) -> dict:
    if answer_coverage is None:
        answer_coverage = recall
    if selected_answer_coverage is None:
        selected_answer_coverage = recall
    return {
        "oracle": oracle,
        "generator_id": "g",
        "prompt_id": "p",
        "metric_id": "m",
        "mean_f1": f1,
        "mean_exact_match": 0.0,
        "mean_context_recall": recall,
        "mean_context_f1": recall,
        "mean_answer_coverage": answer_coverage,
        "mean_selected_answer_coverage": selected_answer_coverage,
        "avg_context_tokens": 10,
        "avg_generation_ms": 1,
        "stage_diagnostics": {},
    }


def test_compare_runs_detects_oracle_dominance_violation(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    content = tmp_path / "content.json"
    oracle = tmp_path / "oracle.json"
    _write(baseline, _metrics(0.2))
    _write(content, _metrics(0.6))
    _write(oracle, _metrics(0.5, oracle=True))

    summary = compare_runs(
        [
            QARunSpec("baseline", "legacy", baseline),
            QARunSpec("content", "content", content),
            QARunSpec("oracle", "direct_gold_context", oracle),
        ]
    )

    assert summary["qa_settings_consistent"] is True
    assert summary["oracle_context_complete"] is True
    assert summary["oracle_answer_coverage"] == 1.0
    assert summary["oracle_selected_answer_coverage"] == 1.0
    assert summary["oracle_dominance_valid"] is False
    assert summary["dominance_violations"] == ["content"]


def test_compare_runs_accepts_valid_oracle(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    oracle = tmp_path / "oracle.json"
    _write(baseline, _metrics(0.2))
    _write(oracle, _metrics(0.5, oracle=True))

    summary = compare_runs(
        [
            QARunSpec("baseline", "legacy", baseline),
            QARunSpec("oracle", "direct_gold_context", oracle),
        ]
    )

    assert summary["oracle_dominance_valid"] is True
    assert summary["rows"][0]["oracle_gap"] == 0.3


def test_compare_runs_adoption_gate_requires_gap_and_evidence(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    content = tmp_path / "content.json"
    oracle = tmp_path / "oracle.json"
    _write(baseline, _metrics(0.2, recall=0.8, answer_coverage=0.8, selected_answer_coverage=0.7))
    _write(content, _metrics(0.3, recall=0.7, answer_coverage=0.6, selected_answer_coverage=0.7))
    _write(oracle, _metrics(0.5, oracle=True))

    summary = compare_runs(
        [
            QARunSpec("baseline", "legacy", baseline),
            QARunSpec("content", "content", content),
            QARunSpec("oracle", "direct_gold_context", oracle),
        ]
    )

    check = summary["adoption_checks"][1]
    assert check["run"] == "content"
    assert check["adoption_gate_pass"] is False
    assert "answer_coverage_regression" in check["blockers"]
    assert "rendered_recall_regression" in check["blockers"]
    assert isclose(check["f1_delta_vs_reference"], 0.1)
    assert isclose(check["oracle_gap_delta_vs_reference"], -0.1)


def test_compare_runs_adoption_gate_accepts_qa_and_evidence_improvement(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    content = tmp_path / "content.json"
    oracle = tmp_path / "oracle.json"
    _write(baseline, _metrics(0.2, recall=0.6, answer_coverage=0.5, selected_answer_coverage=0.4))
    _write(content, _metrics(0.3, recall=0.7, answer_coverage=0.6, selected_answer_coverage=0.4))
    _write(oracle, _metrics(0.5, oracle=True))

    summary = compare_runs(
        [
            QARunSpec("baseline", "legacy", baseline),
            QARunSpec("content", "content", content),
            QARunSpec("oracle", "direct_gold_context", oracle),
        ]
    )

    check = summary["adoption_checks"][1]
    assert check["run"] == "content"
    assert check["adoption_gate_pass"] is True
    assert check["blockers"] == []
