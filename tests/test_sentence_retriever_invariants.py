import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index
from pamae_rag.sentence_graph.sentence_retriever import (
    SentenceRetrieverConfig,
    retrieve_sentence_medoids,
)


def _node(node_id: str, text: str, relevance: float = 1.0) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        relevance=relevance,
        token_count=max(1, len(text.split())),
        metadata={"title": node_id, "corpus_index": node_id},
    )


def test_sentence_retriever_selects_only_sentence_medoids_and_refines_monotone():
    index = build_sentence_graph_index(
        [
            _node("Ada", "Ada Lovelace met Charles Babbage. Charles Babbage built engines.", 2.0),
            _node("Grace", "Grace Hopper wrote COBOL. COBOL influenced programmers.", 1.0),
        ],
        graph_variant=ENTITY_SENTENCE,
    )
    result = retrieve_sentence_medoids(
        index,
        "Which engine did Ada Lovelace discuss with Charles Babbage?",
        config=SentenceRetrieverConfig(k=2, num_samples=3, sample_size_per_k=4),
        seed=7,
    )

    assert result.selected_sentence_ids
    assert all(sentence_id.startswith("sent:") for sentence_id in result.selected_sentence_ids)
    assert result.diagnostics["ppr_used_only_for_sentence_mass"] is True
    assert result.phi_after_refine <= result.phi_before_refine + 1e-12
    assert result.diagnostics["objective_increase_count"] == 0
    assert result.diagnostics["triangle_inequality_violation_count"] == 0


def test_sentence_retriever_uses_deterministic_anchor_fallback_without_dense_retrieval():
    index = build_sentence_graph_index(
        [_node("Ada", "Ada Lovelace met Charles Babbage.")],
        graph_variant=ENTITY_SENTENCE,
    )
    result = retrieve_sentence_medoids(
        index,
        "lowercase query with no named anchor",
        config=SentenceRetrieverConfig(k=1, num_samples=2, sample_size_per_k=2),
        seed=3,
    )

    assert result.anchor_fallback_used is True
    assert result.query_anchor_entity_ids
    assert result.selected_sentence_ids
