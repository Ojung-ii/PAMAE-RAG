from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_local_surface_runs import compare


def _write_run(root: Path, name: str, *, oracle: bool, qa: float = 0.5) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "local_surface_metrics.json").write_text(
        json.dumps(
            {
                "renderer_mode": name.removeprefix("entity_chunk_reference_"),
                "oracle_renderer": oracle,
                "answer_sentence_available_in_selected_chunks": 1.0,
                "gold_sentence_available_in_selected_chunks": 1.0,
                "answer_sentence_rendered": 1.0,
                "gold_sentence_rendered": 1.0,
                "answer_in_context": 1.0,
                "rendered_recall": 1.0,
                "context_f1": 1.0,
                "qa_f1": qa,
                "avg_context_tokens": 10.0,
                "triangle_inequality_violation_count": 0,
                "local_objective_invalid_count": 0,
            }
        ),
        encoding="utf-8",
    )


def test_compare_excludes_oracle_renderers_from_adoption(tmp_path: Path) -> None:
    ds = tmp_path / "2wikimultihopqa"
    _write_run(ds, "entity_chunk_reference_current_renderer", oracle=False, qa=0.4)
    _write_run(ds, "entity_chunk_reference_selected_chunk_answer_sentence_oracle", oracle=True, qa=1.0)

    summary = compare(tmp_path, ["2wikimultihopqa"])
    oracle_gate = next(
        gate
        for gate in summary["2wikimultihopqa"]["gates"]
        if gate["run"] == "entity_chunk_reference_selected_chunk_answer_sentence_oracle"
    )

    assert "oracle_renderer_excluded" in oracle_gate["blockers"]
    assert summary["_final"]["decision"] == "STOP"
