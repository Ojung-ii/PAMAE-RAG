from pamae_rag.data.io import read_jsonl


def test_read_smoke_examples():
    examples = read_jsonl("data/smoke/examples.jsonl")
    assert len(examples) == 2
    assert examples[0].query_id == "q1"
    assert examples[0].gold_node_ids
    assert examples[0].nodes[0].embedding.shape[0] == 3
