import json
from pathlib import Path

from pamae_rag.eval.qa_run_compare import QARunSpec, compare_runs


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(f1: float, *, oracle: bool = False, recall: float = 1.0) -> dict:
    return {
        "oracle": oracle,
        "generator_id": "g",
        "prompt_id": "p",
        "metric_id": "m",
        "mean_f1": f1,
        "mean_exact_match": 0.0,
        "mean_context_recall": recall,
        "mean_context_f1": recall,
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
