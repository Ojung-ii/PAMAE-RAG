import json
from pathlib import Path

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.qa.runner import run_qa


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_run_qa_scores_retrieved_context(tmp_path: Path):
    example = QueryExample(
        query_id="q1",
        query="Where was Ada born?",
        answer="London",
        gold_node_ids=frozenset({"n1"}),
        nodes=(
            EvidenceNode(
                node_id="n1",
                text="Ada was born in London.",
                embedding=np.array([1.0, 0.0]),
                token_count=5,
            ),
            EvidenceNode(
                node_id="n2",
                text="Grace studied mathematics.",
                embedding=np.array([0.0, 1.0]),
                token_count=3,
            ),
        ),
    )
    input_path = tmp_path / "examples.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "qa.jsonl"
    metrics_path = tmp_path / "metrics.json"
    _write_jsonl(
        input_path,
        [
            {
                "query_id": example.query_id,
                "query": example.query,
                "answer": example.answer,
                "gold_node_ids": list(example.gold_node_ids),
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "text": node.text,
                        "embedding": node.embedding.tolist(),
                        "token_count": node.token_count,
                    }
                    for node in example.nodes
                ],
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [{"query_id": "q1", "context_node_ids": ["n1"], "latency_ms": 7.0}],
    )

    metrics = run_qa(input_path, prediction_path, output_path, metrics_path)

    assert metrics.num_queries == 1
    assert metrics.oracle is False
    assert metrics.mean_f1 > 0.0
    assert metrics.mean_context_recall == 1.0
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["prediction"] == "Ada was born in London."
    assert row["generation_ms"] >= 0.0
