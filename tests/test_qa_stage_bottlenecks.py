import json
from pathlib import Path

from scripts.analyze_qa_stage_bottlenecks import analyze


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(
    query_id: str,
    *,
    f1: float = 0.0,
    context_recall: float = 1.0,
    answer_coverage: float = 1.0,
    selected_answer_coverage: float = 1.0,
    projection: float | None = 1.0,
    local: float | None = 1.0,
    rendered: float | None = 1.0,
) -> dict:
    stages = {
        "local_refinement": {"gold_supporting_evidence_survival": local},
        "context_rendering": {"rendered_recall": rendered},
    }
    if projection is not None:
        stages["content_graph_projection"] = {
            "gold_supporting_evidence_survival": projection,
        }
    return {
        "query_id": query_id,
        "f1": f1,
        "context_recall": context_recall,
        "answer_coverage": answer_coverage,
        "selected_answer_coverage": selected_answer_coverage,
        "stage_diagnostics": stages,
    }


def test_analyze_qa_stage_bottlenecks_buckets_content_rows(tmp_path: Path):
    baseline_path = tmp_path / "baseline.jsonl"
    content_path = tmp_path / "content.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    _write_jsonl(
        baseline_path,
        [
            _row("q1", f1=0.1),
            _row("q2", f1=0.0),
            _row("q3", f1=0.3),
        ],
    )
    _write_jsonl(
        content_path,
        [
            _row("q1", f1=0.0, local=0.0, answer_coverage=1.0, selected_answer_coverage=0.0),
            _row("q2", f1=0.0, answer_coverage=1.0, selected_answer_coverage=0.0),
            _row("q3", f1=0.4, answer_coverage=1.0, selected_answer_coverage=1.0),
        ],
    )
    _write_jsonl(
        oracle_path,
        [
            _row("q1", f1=0.5),
            _row("q2", f1=0.5),
            _row("q3", f1=0.5),
        ],
    )

    result = analyze(baseline_path, content_path, oracle_path)

    summary = result["content_bucket_summary"]
    assert summary["local_refinement_loss"]["count"] == 1
    assert summary["answer_present_not_selected"]["count"] == 1
    assert summary["qa_partial_or_success"]["count"] == 1
    assert summary["answer_present_not_selected"]["mean_answer_coverage"] == 1.0
    assert summary["answer_present_not_selected"]["mean_selected_answer_coverage"] == 0.0
    assert result["transition_counts"]["qa_partial_or_success->local_refinement_loss"] == 1
