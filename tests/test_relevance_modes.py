import numpy as np

from pamae_rag.config import load_config
from pamae_rag.data.schema import EvidenceNode
from pamae_rag.objective.relevance_mass import relevance_mass


class TrapDict(dict):
    def get(self, key, default=None):
        if key in {"obj", "o_wiki_title", "possible_answers", "is_supporting", "gold_node_ids"}:
            raise AssertionError(f"gold leakage key accessed: {key}")
        return super().get(key, default)


def _node(node_id: str, title: str, text: str, relevance: float) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        relevance=relevance,
        token_count=1,
        metadata={"title": title},
    )


def test_relevance_title_aware_no_gold_leakage():
    nodes = (
        _node("subject", "George Rankin", "George Rankin was a representative.", 0.01),
        _node("surname", "Rankin", "A list of people named Rankin.", 0.99),
    )
    metadata = TrapDict(
        {
            "subj": "George Rankin",
            "obj": "blocked",
            "o_wiki_title": "blocked",
            "possible_answers": ["blocked"],
            "gold_node_ids": ["subject"],
        }
    )

    title_aware = relevance_mass(
        nodes,
        mode="title_aware",
        query="Who was George Rankin?",
        query_metadata=metadata,
    )
    assert title_aware[0] > title_aware[1]


def test_diagnostic_subject_title_raises_subject_node_above_current_without_gold_leakage():
    nodes = (
        _node("subject", "George Rankin", "George Rankin was a representative.", 0.01),
        _node("surname", "Rankin", "A list of people named Rankin.", 0.99),
    )
    metadata = TrapDict({"subj": "George Rankin", "possible_answers": ["blocked"]})

    current = relevance_mass(nodes, mode="current", query="Who was George Rankin?")
    diagnostic = relevance_mass(
        nodes,
        mode="diagnostic_subject_title",
        query="Who was George Rankin?",
        query_metadata=metadata,
    )

    assert diagnostic[0] > current[0]
    assert diagnostic[0] > diagnostic[1]


def test_diagnostic_subject_title_mode_not_default():
    assert load_config("configs/default.yaml").pamae.relevance_mode == "current"
    assert (
        load_config("configs/ablations/popqa_relevance_diagnostic_subject_title.yaml").pamae.relevance_mode
        == "diagnostic_subject_title"
    )

