from __future__ import annotations

import json
from pathlib import Path

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.raw_adapters import prepare_raw_qa_corpus_dataset


def test_prepare_raw_qa_corpus_dataset(tmp_path: Path):
    corpus = [
        {"title": "A", "text": "Ada studied mathematics at Example College."},
        {"title": "B", "text": "Example College is located in Example City."},
        {"title": "C", "text": "Noise document about another subject."},
    ]
    qa = [
        {
            "id": "q1",
            "question": "Where is the college where Ada studied?",
            "obj": "Example City",
            "possible_answers": '["Example City"]',
            "paragraphs": [
                {"title": "A", "text": corpus[0]["text"], "is_supporting": True},
                {"title": "B", "text": corpus[1]["text"], "is_supporting": True},
            ],
        }
    ]
    qa_path = tmp_path / "qa.json"
    corpus_path = tmp_path / "corpus.json"
    out_path = tmp_path / "examples.jsonl"
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    summary = prepare_raw_qa_corpus_dataset(
        qa_path,
        corpus_path,
        out_path,
        dataset_name="tiny",
        max_nodes_per_query=3,
        embedding_dim=8,
        max_features=100,
    )
    assert summary["num_examples"] == 1
    examples = read_jsonl(out_path)
    assert len(examples) == 1
    assert examples[0].query_id == "q1"
    assert len(examples[0].gold_node_ids) == 2
    assert all(node.embedding.size > 0 for node in examples[0].nodes)
