import json
from pathlib import Path

from scripts.analyze_relevance_alignment import analyze_examples, main


def _write_example(path: Path) -> None:
    row = {
        "query_id": "q1",
        "query": "Who was George Rankin?",
        "gold_node_ids": ["doc0"],
        "nodes": [
            {
                "node_id": "doc0",
                "text": "George Rankin was a representative.",
                "embedding": [1.0, 0.0],
                "relevance": 0.1,
                "token_count": 5,
                "metadata": {"title": "George Rankin"},
            },
            {
                "node_id": "doc1",
                "text": "A list of people named Rankin.",
                "embedding": [0.0, 1.0],
                "relevance": 0.9,
                "token_count": 5,
                "metadata": {"title": "Rankin"},
            },
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_relevance_alignment_script(tmp_path: Path):
    input_path = tmp_path / "examples.jsonl"
    output_json = tmp_path / "alignment.json"
    output_md = tmp_path / "alignment.md"
    _write_example(input_path)

    metrics = analyze_examples(input_path, relevance_mode="entity_title_aware")
    assert metrics["num_queries"] == 1
    assert metrics["gold_total"] == 1
    assert metrics["gold_top1_rate"] == 1.0

    main(
        [
            "--input",
            str(input_path),
            "--relevance-mode",
            "entity_title_aware",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["gold_top1_rate"] == 1.0
    assert "Relevance Alignment Diagnostic" in output_md.read_text(encoding="utf-8")
