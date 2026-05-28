import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.eval.support_facts import resolve_support_facts, support_fact_stage_metrics


def _node(node_id: str, title: str, text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        token_count=max(1, len(text.split())),
        metadata={"title": title},
    )


def test_resolve_support_facts_matches_title_and_sentence_id():
    nodes = [
        _node("n1", "Ada", "Ada wrote notes. Ada was born in London."),
        _node("n2", "Grace", "Grace studied mathematics."),
    ]

    resolved = resolve_support_facts(nodes, [{"title": "Ada", "sentence_id": 1}])

    assert len(resolved) == 1
    assert resolved[0].node_id == "n1"
    assert resolved[0].sentence == "Ada was born in London."


def test_support_fact_stage_metrics_separates_fact_survival_from_node_recall():
    nodes = [
        _node("n1", "Ada", "Ada wrote notes. Ada was born in London."),
        _node("n2", "Grace", "Grace studied mathematics."),
    ]
    metadata = {
        "support_facts": [
            {"title": "Ada", "sentence_id": 1},
            {"title": "Grace", "sentence_id": 0},
        ]
    }

    metrics = support_fact_stage_metrics(
        nodes=nodes,
        selected_node_ids=["n1"],
        metadata=metadata,
    )

    assert metrics["support_fact_count"] == 2
    assert metrics["support_fact_resolved_count"] == 2
    assert metrics["support_fact_surviving_count"] == 1
    assert metrics["support_fact_survival"] == 0.5
    assert metrics["support_fact_resolved_survival"] == 0.5
