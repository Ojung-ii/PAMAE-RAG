import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE
from pamae_rag.sentence_graph.sentence_diagnostics import (
    aggregate_sentence_traces,
    build_sentence_diagnostic_trace,
    sentence_mapping_diagnostics,
)
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index
from pamae_rag.sentence_graph.sentence_renderer import SENTENCE_PATH, render_sentence_context
from pamae_rag.sentence_graph.sentence_retriever import (
    SentenceRetrieverConfig,
    retrieve_sentence_medoids,
)


def _example() -> QueryExample:
    node = EvidenceNode(
        node_id="n1",
        text="Ada wrote notes. Ada was born in London.",
        embedding=np.array([1.0, 0.0]),
        relevance=1.0,
        metadata={"title": "Ada", "corpus_index": "n1"},
    )
    return QueryExample(
        query_id="q1",
        query="Where was Ada born?",
        nodes=(node,),
        gold_node_ids=frozenset({"n1"}),
        answer="London",
        metadata={"dataset": "toy", "support_facts": [{"title": "Ada", "sentence_id": 1}]},
    )


def test_sentence_mapping_diagnostics_maps_support_and_answer_sentence():
    example = _example()
    index = build_sentence_graph_index(example.nodes, graph_variant=ENTITY_SENTENCE)

    diagnostics = sentence_mapping_diagnostics(index, example)

    assert diagnostics["gold_support_sentence_mapping_rate"] == 1.0
    assert diagnostics["answer_containing_sentence_found_rate"] == 1.0


def test_sentence_trace_and_aggregate_include_projection_rendering_and_objective():
    example = _example()
    index = build_sentence_graph_index(example.nodes, graph_variant=ENTITY_SENTENCE)
    retrieval = retrieve_sentence_medoids(
        index,
        example.query,
        config=SentenceRetrieverConfig(k=1, num_samples=2, sample_size_per_k=2),
        seed=11,
    )
    render = render_sentence_context(
        index,
        retrieval,
        renderer_mode=SENTENCE_PATH,
        max_context_tokens=64,
    )

    trace = build_sentence_diagnostic_trace(
        example=example,
        index=index,
        retrieval=retrieval,
        render=render,
        graph_variant=ENTITY_SENTENCE,
        renderer_mode=SENTENCE_PATH,
        qa_f1=1.0,
    )
    summary = aggregate_sentence_traces([trace])

    assert trace["gold_sentence_projected"] is True
    assert trace["answer_sentence_found"] is True
    assert "phi_before_refine" in trace
    assert summary["qa_f1"] == 1.0
    assert summary["objective_increase_count"] == 0
