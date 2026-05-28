import json
from pathlib import Path

from pamae_rag.diagnostics.oracle_runs import oracle_diagnostics


def test_oracle_diagnostics_separate_answer_copy(tmp_path: Path):
    input_path = tmp_path / "examples.jsonl"
    corpus_path = tmp_path / "corpus.json"
    row = {
        "query_id": "q1",
        "query": "Who invented the Difference Engine?",
        "answer": "Charles Babbage",
        "gold_node_ids": ["toy:doc:0"],
        "metadata": {"support_facts": [{"title": "Difference Engine", "sent_id": 0}]},
        "nodes": [
            {
                "node_id": "toy:doc:0",
                "text": "Charles Babbage invented the Difference Engine.",
                "embedding": [1.0],
                "token_count": 6,
                "metadata": {"title": "Difference Engine"},
            }
        ],
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    corpus_path.write_text(
        json.dumps([{"title": "Difference Engine", "text": "Charles Babbage invented the Difference Engine."}]),
        encoding="utf-8",
    )

    result = oracle_diagnostics(input_path, corpus_path=corpus_path, method_f1=0.0)

    assert result["answer_copy_f1"] > 0.0
    assert result["answer_containing_f1"] > 0.0
    assert result["gold_support_f1"] > 0.0
    assert result["oracle_dominance_valid"] is True
