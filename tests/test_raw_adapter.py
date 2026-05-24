from __future__ import annotations

import json
from pathlib import Path

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.raw_adapters import prepare_raw_qa_corpus_dataset
from scripts.check_gold_universe import check_gold_universe


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


def test_hotpotqa_support_label_extraction_minimal_schema(tmp_path: Path):
    corpus = [
        {"title": "Alpha Page", "text": "Alpha Page first supporting sentence. Another alpha sentence."},
        {"title": "Beta Page", "text": "Beta Page second supporting sentence."},
        {"title": "Noise", "text": "Unrelated noise document."},
    ]
    qa = [
        {
            "_id": "hotpot-mini",
            "question": "What connects Alpha Page and Beta Page?",
            "answer": "Beta",
            "context": [
                ["Alpha Page", ["Alpha Page first supporting sentence.", "Another alpha sentence."]],
                ["Beta Page", ["Beta Page second supporting sentence."]],
                ["Noise", ["Unrelated noise document."]],
            ],
            "supporting_facts": [["Alpha Page", 0], ["Beta Page", 0]],
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
        dataset_name="hotpotqa",
        max_nodes_per_query=3,
        embedding_dim=8,
        max_features=100,
    )

    examples = read_jsonl(out_path)
    assert summary["total_support_paragraphs_seen"] == 2
    assert examples[0].gold_node_ids == frozenset({"hotpotqa:doc:0", "hotpotqa:doc:1"})
    assert examples[0].metadata["metadata"]["support_facts"][0]["sentence_id"] == 0


def test_2wiki_support_label_extraction_minimal_schema(tmp_path: Path):
    corpus = [
        {"title": "Lothair II", "text": "Lothair II had mother Ermengarde of Tours."},
        {"title": "Ermengarde of Tours", "text": "Ermengarde of Tours died on 20 March 851."},
        {"title": "Noise", "text": "Unrelated noise document."},
    ]
    qa = [
        {
            "_id": "2wiki-mini",
            "question": "When did Lothair II's mother die?",
            "answer": "20 March 851",
            "context": [
                ["Lothair II", ["Lothair II had mother Ermengarde of Tours."]],
                ["Ermengarde of Tours", ["Ermengarde of Tours died on 20 March 851."]],
            ],
            "evidences": [["Lothair II", "mother", "Ermengarde of Tours"]],
            "supporting_facts": [["Lothair II", 0], ["Ermengarde of Tours", 0]],
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
        dataset_name="2wikimultihopqa",
        max_nodes_per_query=3,
        embedding_dim=8,
        max_features=100,
    )

    examples = read_jsonl(out_path)
    assert summary["total_support_paragraphs_seen"] == 2
    assert examples[0].gold_node_ids == frozenset({"2wikimultihopqa:doc:0", "2wikimultihopqa:doc:1"})


def test_check_gold_universe_script(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    rows = [
        {
            "query_id": "q1",
            "nodes": [{"node_id": "d1"}, {"node_id": "d2"}],
            "gold_node_ids": ["d1", "d3"],
        },
        {
            "query_id": "q2",
            "nodes": [{"node_id": "d4"}],
            "gold_node_ids": [],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    metrics = check_gold_universe(path)
    assert metrics["num_queries"] == 2
    assert metrics["gold_total"] == 2
    assert metrics["gold_in_nodes"] == 1
    assert metrics["examples_with_no_gold"] == 1
    assert metrics["examples_missing_some_gold"] == 1
