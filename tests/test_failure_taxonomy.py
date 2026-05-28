import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.failure_taxonomy import summarize_taxonomy, taxonomy_rows


def _example(query_id: str) -> QueryExample:
    return QueryExample(
        query_id=query_id,
        query="Who?",
        nodes=(
            EvidenceNode("gold", "Gold evidence.", np.asarray([1.0])),
            EvidenceNode("other", "Other evidence.", np.asarray([0.0])),
        ),
        gold_node_ids=frozenset({"gold"}),
    )


def _qa_row(
    query_id: str,
    *,
    projected: list[str] | None,
    selected: list[str] | None,
    rendered: list[str] | None,
    answer_coverage: float = 0.0,
    f1: float = 0.0,
    exact_match: float = 0.0,
) -> dict:
    return {
        "query_id": query_id,
        "f1": f1,
        "exact_match": exact_match,
        "answer_coverage": answer_coverage,
        "context_node_ids": rendered or [],
        "stage_diagnostics": {
            "content_graph_projection": {"gold_supporting_node_ids": projected or []},
            "local_refinement": {"gold_supporting_node_ids": selected or []},
            "context_rendering": {"gold_supporting_node_ids": rendered or []},
        },
    }


def test_failure_taxonomy_classifies_stage_order():
    examples = [_example(f"q{i}") for i in range(5)]
    qa_rows = {
        "q0": _qa_row("q0", projected=[], selected=[], rendered=[]),
        "q1": _qa_row("q1", projected=["gold"], selected=[], rendered=[]),
        "q2": _qa_row("q2", projected=["gold"], selected=["gold"], rendered=[]),
        "q3": _qa_row(
            "q3",
            projected=["gold"],
            selected=["gold"],
            rendered=["gold"],
            answer_coverage=1.0,
        ),
        "q4": _qa_row(
            "q4",
            projected=["gold"],
            selected=["gold"],
            rendered=["gold"],
            f1=1.0,
            exact_match=1.0,
        ),
    }

    rows = taxonomy_rows(examples, qa_rows)
    assert [row["failure_type"] for row in rows] == [
        "projection_miss",
        "selection_miss",
        "rendering_miss",
        "qa_fail",
        "success",
    ]
    assert summarize_taxonomy(rows) == {
        "projection_miss": 1,
        "selection_miss": 1,
        "rendering_miss": 1,
        "qa_fail": 1,
        "success": 1,
    }
